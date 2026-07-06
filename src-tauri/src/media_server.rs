use axum::{
	Router,
	body::Body,
	extract::Query,
	http::{HeaderMap, StatusCode, header},
	response::Response,
	routing::get,
};
use log::{error, info};
use std::path::Path;
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tokio::io::{AsyncReadExt, AsyncSeekExt};
use tokio::sync::Notify;
use tokio::time::{Duration, timeout};

const MAX_CHUNK_SIZE: u64 = 4 * 1024 * 1024; // 4 MB

#[derive(serde::Deserialize)]
struct FileQuery {
	path: String,
}

fn content_type_for(ext: &str) -> &'static str {
	match ext {
		"mp4" => "video/mp4",
		"webm" => "video/webm",
		"mov" => "video/quicktime",
		"avi" => "video/x-msvideo",
		"mkv" => "video/x-matroska",
		_ => "application/octet-stream",
	}
}

// ── Transcode session for progressive in-memory streaming ───────────────────

struct TranscodeSession {
	buffer: Mutex<Vec<u8>>,
	complete: AtomicBool,
	notify: Notify,
	abort: AtomicBool,
	/// Cached fragment offsets with the buffer length at time of parsing.
	frag_cache: Mutex<(Vec<usize>, usize)>,
	/// Cached duration (probed once, never changes).
	duration_secs: Mutex<Option<f64>>,
}

impl TranscodeSession {
	fn new() -> Self {
		Self {
			buffer: Mutex::new(Vec::new()),
			complete: AtomicBool::new(false),
			notify: Notify::new(),
			abort: AtomicBool::new(false),
			frag_cache: Mutex::new((Vec::new(), 0)),
			duration_secs: Mutex::new(None),
		}
	}

	/// Get fragment offsets, re-parsing only if the buffer has grown.
	/// Holds the buffer lock during parsing (fast linear scan of MP4 box headers).
	fn get_fragment_offsets(&self) -> Vec<usize> {
		let buffer = self.buffer.lock().unwrap();
		let buf_len = buffer.len();
		let mut cache = self.frag_cache.lock().unwrap();
		if cache.1 == buf_len {
			return cache.0.clone();
		}
		let offsets = parse_mp4_fragment_offsets(&buffer);
		*cache = (offsets.clone(), buf_len);
		offsets
	}

	fn get_duration(&self) -> f64 {
		self.duration_secs.lock().unwrap().unwrap_or(1.0)
	}
}

/// Parse top-level MP4 boxes and return byte offsets of each moof box.
/// Note: only handles 32-bit box sizes (sufficient for FFmpeg fMP4 output < 4 GB).
fn parse_mp4_fragment_offsets(buffer: &[u8]) -> Vec<usize> {
	let mut offsets = Vec::new();
	let mut pos = 0;
	while pos + 8 <= buffer.len() {
		let size = u32::from_be_bytes([buffer[pos], buffer[pos + 1], buffer[pos + 2], buffer[pos + 3]]) as usize;
		if size < 8 {
			break;
		}
		if &buffer[pos + 4..pos + 8] == b"moof" {
			offsets.push(pos);
		}
		if pos + size > buffer.len() {
			break;
		}
		pos += size;
	}
	offsets
}

/// Global active transcode session: (source_path, session).
/// Only one video is loaded at a time.
static ACTIVE_TRANSCODE: Mutex<Option<(String, Arc<TranscodeSession>)>> = Mutex::new(None);

/// Get the existing session for `path`, or start a new transcode.
fn get_or_start_transcode(path: &str) -> Arc<TranscodeSession> {
	let mut guard = ACTIVE_TRANSCODE.lock().unwrap();

	if let Some((ref existing_path, ref session)) = *guard {
		if existing_path == path {
			return Arc::clone(session);
		}
		session.abort.store(true, Ordering::Release);
		session.notify.notify_waiters();
	}

	let session = Arc::new(TranscodeSession::new());
	*guard = Some((path.to_string(), Arc::clone(&session)));

	let session_clone = Arc::clone(&session);
	let path_owned = path.to_string();
	tauri::async_runtime::spawn(async move {
		run_transcode(&path_owned, &session_clone).await;
	});

	session
}

/// Abort and drop the active transcode session.
pub fn cleanup_transcode() {
	let mut guard = ACTIVE_TRANSCODE.lock().unwrap();
	if let Some((_, ref session)) = *guard {
		session.abort.store(true, Ordering::Release);
		session.notify.notify_waiters();
	}
	*guard = None;
}

/// Spawn FFmpeg and feed chunks into the session buffer.
async fn run_transcode(path: &str, session: &TranscodeSession) {
	// Probe duration once and cache it
	let dur = crate::get_duration(path).await.unwrap_or(0.0);
	*session.duration_secs.lock().unwrap() = Some(dur);

	let vcodec = crate::get_video_codec(path).await.unwrap_or_default();
	let can_remux = crate::is_browser_compatible_video_codec(&vcodec);

	let mut args = vec!["-i", path];
	if can_remux {
		args.extend_from_slice(&["-c", "copy"]);
	} else {
		args.extend_from_slice(&["-c:v", "libx264", "-preset", "ultrafast", "-crf", "28", "-profile:v", "baseline", "-c:a", "aac", "-b:a", "128k"]);
	}
	args.extend_from_slice(&["-movflags", "frag_keyframe+empty_moov+default_base_moof", "-f", "mp4", "pipe:1"]);

	info!("Transcode start: codec={}, strategy={}, duration={:.1}s", vcodec, if can_remux { "remux" } else { "encode" }, dur);
	let t_start = std::time::Instant::now();

	let mut child = match tokio::process::Command::new(crate::ffmpeg_path()).args(&args).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn() {
		Ok(c) => c,
		Err(e) => {
			error!("Transcode spawn failed: {}", e);
			session.complete.store(true, Ordering::Release);
			session.notify.notify_waiters();
			return;
		}
	};

	let stderr_handle = child.stderr.take();

	if let Some(mut stdout) = child.stdout.take() {
		let mut buf = [0u8; 64 * 1024];
		loop {
			if session.abort.load(Ordering::Acquire) {
				let _ = child.kill().await;
				info!("Transcode aborted");
				break;
			}
			match timeout(Duration::from_secs(30), stdout.read(&mut buf)).await {
				Ok(Ok(0)) => break,
				Ok(Ok(n)) => {
					session.buffer.lock().unwrap().extend_from_slice(&buf[..n]);
					session.notify.notify_waiters();
				}
				Ok(Err(_)) => break,
				Err(_) => {
					error!("Transcode read timeout (30s), killing FFmpeg");
					let _ = child.kill().await;
					break;
				}
			}
		}
	}

	let status = child.wait().await;
	session.complete.store(true, Ordering::Release);
	session.notify.notify_waiters();

	let size = session.buffer.lock().unwrap().len();
	let elapsed = t_start.elapsed().as_secs_f64();

	match status {
		Ok(s) if s.success() => info!("Transcode done: {:.1}s, {} bytes", elapsed, size),
		Ok(s) => {
			let mut stderr_buf = String::new();
			if let Some(mut stderr) = stderr_handle {
				let _ = stderr.read_to_string(&mut stderr_buf).await;
			}
			error!("Transcode failed (exit {}): {}", s, stderr_buf.lines().last().unwrap_or("unknown error"));
		}
		Err(e) => error!("Transcode wait error: {}", e),
	}
}

// ── HTTP handlers ───────────────────────────────────────────────────────────

async fn serve_video(Query(query): Query<FileQuery>, headers: HeaderMap) -> Result<Response, StatusCode> {
	if !crate::is_path_registered(&query.path) {
		return Err(StatusCode::FORBIDDEN);
	}

	let file_path = Path::new(&query.path);
	let ext = file_path.extension().and_then(|e| e.to_str()).map(|e| e.to_lowercase()).unwrap_or_default();
	if !crate::ALLOWED_VIDEO_EXTENSIONS.contains(&ext.as_str()) {
		return Err(StatusCode::FORBIDDEN);
	}

	if crate::needs_transcode(&ext) {
		return serve_transcoded(&query.path, &headers).await;
	}

	cleanup_transcode();
	serve_direct(&query.path, &ext, &headers).await
}

/// Serve transcoded content from the in-memory buffer with range-request support.
async fn serve_transcoded(path: &str, headers: &HeaderMap) -> Result<Response, StatusCode> {
	let session = get_or_start_transcode(path);
	let buf_len = session.buffer.lock().unwrap().len();
	let complete = session.complete.load(Ordering::Acquire);

	// Range request — serve from the in-memory buffer (used by hls.js for segments)
	if let Some(range_value) = headers.get(header::RANGE)
		&& buf_len > 0
	{
		let range_str = range_value.to_str().map_err(|_| StatusCode::BAD_REQUEST)?;
		let buffer = session.buffer.lock().unwrap();
		let total = buffer.len() as u64;
		let (start, end) = parse_range(range_str, total).ok_or(StatusCode::RANGE_NOT_SATISFIABLE)?;
		let data = buffer[start as usize..=end as usize].to_vec();
		let total_str = if complete { total.to_string() } else { "*".to_string() };

		return Response::builder()
			.status(StatusCode::PARTIAL_CONTENT)
			.header(header::CONTENT_TYPE, "video/mp4")
			.header(header::CONTENT_LENGTH, data.len())
			.header(header::CONTENT_RANGE, format!("bytes {}-{}/{}", start, end, total_str))
			.header(header::ACCEPT_RANGES, "bytes")
			.body(Body::from(data))
			.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR);
	}

	// Transcode complete, no range — serve full buffer
	if complete && buf_len > 0 {
		let buffer = session.buffer.lock().unwrap();
		let total = buffer.len() as u64;
		let data = buffer.clone();
		return Response::builder()
			.status(StatusCode::OK)
			.header(header::CONTENT_TYPE, "video/mp4")
			.header(header::CONTENT_LENGTH, total)
			.header(header::ACCEPT_RANGES, "bytes")
			.body(Body::from(data))
			.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR);
	}

	// Transcode in progress, no data yet — client should retry
	Err(StatusCode::SERVICE_UNAVAILABLE)
}

/// Serve a file directly from disk with range-request support.
async fn serve_direct(path: &str, ext: &str, headers: &HeaderMap) -> Result<Response, StatusCode> {
	let file_size = tokio::fs::metadata(path).await.map_err(|_| StatusCode::NOT_FOUND)?.len();
	let content_type = content_type_for(ext);

	if let Some(range_value) = headers.get(header::RANGE) {
		let range_str = range_value.to_str().map_err(|_| StatusCode::BAD_REQUEST)?;
		let (start, end) = parse_range(range_str, file_size).ok_or(StatusCode::RANGE_NOT_SATISFIABLE)?;

		let chunk_end = end.min(start + MAX_CHUNK_SIZE - 1);
		let to_read = (chunk_end - start + 1) as usize;

		let mut file = tokio::fs::File::open(path).await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
		file.seek(std::io::SeekFrom::Start(start)).await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

		let mut buf = vec![0u8; to_read];
		let mut read = 0;
		while read < to_read {
			match file.read(&mut buf[read..]).await {
				Ok(0) => break,
				Ok(n) => read += n,
				Err(_) => return Err(StatusCode::INTERNAL_SERVER_ERROR),
			}
		}
		buf.truncate(read);
		let actual_end = start + read as u64 - 1;

		Response::builder()
			.status(StatusCode::PARTIAL_CONTENT)
			.header(header::CONTENT_TYPE, content_type)
			.header(header::CONTENT_LENGTH, read)
			.header(header::CONTENT_RANGE, format!("bytes {}-{}/{}", start, actual_end, file_size))
			.header(header::ACCEPT_RANGES, "bytes")
			.body(Body::from(buf))
			.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
	} else {
		let file = tokio::fs::File::open(path).await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
		let stream = tokio_util::io::ReaderStream::new(file);

		Response::builder()
			.status(StatusCode::OK)
			.header(header::CONTENT_TYPE, content_type)
			.header(header::CONTENT_LENGTH, file_size)
			.header(header::ACCEPT_RANGES, "bytes")
			.body(Body::from_stream(stream))
			.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
	}
}

fn parse_range(range: &str, file_size: u64) -> Option<(u64, u64)> {
	let range = range.strip_prefix("bytes=")?;
	let (start_str, end_str) = range.split_once('-')?;

	if start_str.is_empty() {
		let suffix: u64 = end_str.parse().ok()?;
		Some((file_size.checked_sub(suffix)?, file_size - 1))
	} else if end_str.is_empty() {
		let start: u64 = start_str.parse().ok()?;
		(start < file_size).then_some((start, file_size - 1))
	} else {
		let start: u64 = start_str.parse().ok()?;
		let end: u64 = end_str.parse().ok()?;
		(start <= end && start < file_size).then_some((start, end.min(file_size - 1)))
	}
}

/// Generate an HLS (m3u8) playlist from the in-memory fMP4 transcode buffer.
/// Uses byte-range addressing so hls.js fetches segments via Range requests
/// to the /video endpoint. Supports live (EVENT) during transcode and VOD after.
async fn serve_hls_playlist(Query(query): Query<FileQuery>) -> Result<Response, StatusCode> {
	if !crate::is_path_registered(&query.path) {
		return Err(StatusCode::FORBIDDEN);
	}

	let file_path = Path::new(&query.path);
	let ext = file_path.extension().and_then(|e| e.to_str()).map(|e| e.to_lowercase()).unwrap_or_default();
	if !crate::needs_transcode(&ext) {
		return Err(StatusCode::NOT_FOUND);
	}

	let session = get_or_start_transcode(&query.path);
	let offsets = session.get_fragment_offsets();
	let buf_len = session.buffer.lock().unwrap().len();
	let complete = session.complete.load(Ordering::Acquire);
	let dur = session.get_duration();

	// Percent-encode the path for use in playlist URIs
	let encoded_path: String = query
		.path
		.bytes()
		.map(|b| match b {
			b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'/' => (b as char).to_string(),
			_ => format!("%{:02X}", b),
		})
		.collect();
	let video_uri = format!("/video?path={}", encoded_path);

	// Init segment: everything before the first moof
	let init_end = match offsets.first() {
		Some(&offset) if offset > 0 => offset,
		_ => return Err(StatusCode::SERVICE_UNAVAILABLE), // No fragments yet
	};

	// Number of servable fragments (skip last if transcode is still running — may be incomplete)
	let num_frags = if complete { offsets.len() } else { offsets.len().saturating_sub(1) };
	if num_frags == 0 {
		return Err(StatusCode::SERVICE_UNAVAILABLE);
	}

	// Fragment duration: actual calculation for VOD, fixed 10s estimate for EVENT.
	// During EVENT, dividing total duration by partial fragment count gives wrong values.
	let frag_dur = if complete { dur / offsets.len() as f64 } else { 10.0 };
	let target_dur = (frag_dur.ceil() as u64).max(1);

	let mut m3u8 = format!(
		"#EXTM3U\n#EXT-X-VERSION:7\n#EXT-X-TARGETDURATION:{}\n#EXT-X-PLAYLIST-TYPE:{}\n#EXT-X-MEDIA-SEQUENCE:0\n#EXT-X-MAP:URI=\"{}\",BYTERANGE=\"{}@0\"\n",
		target_dur,
		if complete { "VOD" } else { "EVENT" },
		video_uri,
		init_end,
	);

	for i in 0..num_frags {
		let frag_start = offsets[i];
		let frag_end = if i + 1 < offsets.len() { offsets[i + 1] } else { buf_len };
		let frag_len = frag_end - frag_start;
		m3u8.push_str(&format!("#EXTINF:{:.6},\n#EXT-X-BYTERANGE:{}@{}\n{}\n", frag_dur, frag_len, frag_start, video_uri));
	}

	if complete {
		m3u8.push_str("#EXT-X-ENDLIST\n");
	}

	Response::builder()
		.status(StatusCode::OK)
		.header(header::CONTENT_TYPE, "application/vnd.apple.mpegurl")
		.header(header::CACHE_CONTROL, "no-cache")
		.body(Body::from(m3u8))
		.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

/// Binds synchronously, spawns async serving. Returns the port immediately.
pub fn start() -> u16 {
	let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("Failed to bind media server");
	let port = listener.local_addr().unwrap().port();
	listener.set_nonblocking(true).unwrap();

	tauri::async_runtime::spawn(async move {
		let listener = tokio::net::TcpListener::from_std(listener).expect("Failed to convert listener");
		let app = Router::new()
			.route("/video", get(serve_video))
			.route("/hls", get(serve_hls_playlist))
			.layer(axum::middleware::from_fn(|req, next: axum::middleware::Next| async {
				let mut res = next.run(req).await;
				// CORS: allow Tauri WebView (different port on localhost) to fetch
				res.headers_mut().insert(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*".parse().unwrap());
				res
			}));
		info!("Media server listening on 127.0.0.1:{}", port);
		axum::serve(listener, app).await.expect("Media server error");
	});

	port
}
