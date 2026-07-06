// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
mod media_server;

use log::{debug, error, info, warn};
use std::collections::HashSet;
use std::path::Path;
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicU16, Ordering};
use std::sync::{Mutex, OnceLock};
use tauri::Emitter;
use tauri_plugin_log::{Target, TargetKind};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, BufReader};
use tokio::process::Command;

static MEDIA_SERVER_PORT: AtomicU16 = AtomicU16::new(0);
static CANCEL_FLAG: AtomicBool = AtomicBool::new(false);
static ALLOWED_PATHS: Mutex<Option<HashSet<String>>> = Mutex::new(None);

#[tauri::command]
fn get_media_server_port() -> u16 {
	MEDIA_SERVER_PORT.load(Ordering::Relaxed)
}

/// Register a file path so the media server is allowed to serve it.
#[tauri::command]
fn register_media_path(path: String) {
	let mut guard = ALLOWED_PATHS.lock().unwrap();
	guard.get_or_insert_with(HashSet::new).insert(path);
}

/// Check whether a path has been registered for media serving.
pub fn is_path_registered(path: &str) -> bool {
	let guard = ALLOWED_PATHS.lock().unwrap();
	guard.as_ref().is_some_and(|set| set.contains(path))
}

#[tauri::command]
fn cancel_extraction() {
	CANCEL_FLAG.store(true, Ordering::Release);
	info!("Extraction cancel requested");
}

const ALLOWED_FORMATS: &[&str] = &["aac", "mp3", "ogg"];
pub const ALLOWED_VIDEO_EXTENSIONS: &[&str] = &["mp4", "mov", "avi", "mkv", "webm"];
const WAVEFORM_BINS: usize = 240;

#[derive(Debug, thiserror::Error)]
enum AppError {
	#[error("File not found: {0}")]
	FileNotFound(String),
	#[error("Unsupported video format: {0}")]
	UnsupportedVideoFormat(String),
	#[error("Unsupported output format: {0}")]
	UnsupportedOutputFormat(String),
	#[error("Invalid time format (expected HH:MM:SS): {0}")]
	InvalidTimeFormat(String),
	#[error("Minutes and seconds must be 0-59: {0}")]
	InvalidTimeRange(String),
	#[error("Start time must be before end time")]
	StartAfterEnd,
	#[error("FFmpeg failed: {0}")]
	FfmpegFailed(String),
	#[error("Cancelled")]
	Cancelled,
	#[error("{0}")]
	Io(String),
}

impl serde::Serialize for AppError {
	fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
	where
		S: serde::Serializer,
	{
		serializer.serialize_str(&self.to_string())
	}
}

/// Find a command binary by checking common installation paths.
/// On macOS, .app bundles launched from Finder have a restricted PATH that
/// doesn't include Homebrew paths, so we search known locations explicitly.
fn find_command(name: &str) -> String {
	let candidates: &[&str] = if cfg!(target_os = "macos") {
		&["/opt/homebrew/bin", "/usr/local/bin"]
	} else if cfg!(target_os = "linux") {
		&["/usr/local/bin", "/usr/bin"]
	} else {
		&[]
	};

	for dir in candidates {
		let full = format!("{}/{}", dir, name);
		if Path::new(&full).exists() {
			return full;
		}
	}

	name.to_string()
}

pub(crate) fn ffmpeg_path() -> &'static str {
	static PATH: OnceLock<String> = OnceLock::new();
	PATH.get_or_init(|| find_command("ffmpeg"))
}

pub(crate) fn ffprobe_path() -> &'static str {
	static PATH: OnceLock<String> = OnceLock::new();
	PATH.get_or_init(|| find_command("ffprobe"))
}

fn validate_extension(ext: &str) -> bool {
	ALLOWED_VIDEO_EXTENSIONS.contains(&ext)
}

fn validate_path(path: &str) -> Result<(), AppError> {
	let p = Path::new(path);
	if !p.exists() {
		return Err(AppError::FileNotFound(path.to_string()));
	}
	match p.extension().and_then(|e| e.to_str()).map(|e| e.to_lowercase()) {
		Some(ext) if validate_extension(&ext) => Ok(()),
		_ => Err(AppError::UnsupportedVideoFormat(path.to_string())),
	}
}

fn validate_format(format: &str) -> Result<(), AppError> {
	if ALLOWED_FORMATS.contains(&format) {
		Ok(())
	} else {
		Err(AppError::UnsupportedOutputFormat(format.to_string()))
	}
}

fn validate_time(time: &str) -> Result<(), AppError> {
	hms_to_seconds(time)?;
	Ok(())
}

fn hms_to_seconds(hms: &str) -> Result<f64, AppError> {
	let parts: Vec<&str> = hms.split(':').collect();
	if parts.len() != 3 {
		return Err(AppError::InvalidTimeFormat(hms.to_string()));
	}
	let h: u32 = parts[0].parse().map_err(|_| AppError::InvalidTimeFormat(hms.to_string()))?;
	let m: u32 = parts[1].parse().map_err(|_| AppError::InvalidTimeFormat(hms.to_string()))?;
	let s: u32 = parts[2].parse().map_err(|_| AppError::InvalidTimeFormat(hms.to_string()))?;
	if m >= 60 || s >= 60 {
		return Err(AppError::InvalidTimeRange(hms.to_string()));
	}
	Ok(h as f64 * 3600.0 + m as f64 * 60.0 + s as f64)
}

/// Formats that Chromium's <video> element cannot play natively.
pub(crate) fn needs_transcode(ext: &str) -> bool {
	matches!(ext, "avi" | "mkv")
}

/// Video codecs that Chromium can decode natively.
pub(crate) fn is_browser_compatible_video_codec(codec: &str) -> bool {
	matches!(codec, "h264" | "vp8" | "vp9" | "av1")
}

/// Probe the video codec of a file using ffprobe.
pub(crate) async fn get_video_codec(path: &str) -> Option<String> {
	let output = Command::new(ffprobe_path())
		.args([
			"-v",
			"error",
			"-select_streams",
			"v:0",
			"-show_entries",
			"stream=codec_name",
			"-of",
			"default=noprint_wrappers=1:nokey=1",
			path,
		])
		.output()
		.await
		.ok()?;
	if !output.status.success() {
		return None;
	}
	let codec = String::from_utf8_lossy(&output.stdout).trim().to_string();
	if codec.is_empty() { None } else { Some(codec) }
}

fn codec_for_format(format: &str, source_codec: Option<&str>) -> &'static str {
	match format {
		"aac" => match source_codec {
			Some("aac") => "copy",
			_ => "aac",
		},
		"mp3" => "libmp3lame",
		"ogg" => "libopus",
		_ => unreachable!("validate_format must be called before codec_for_format"),
	}
}

async fn resolve_codec(format: &str, path: &str) -> &'static str {
	let source_codec = get_audio_codec(path).await;
	codec_for_format(format, source_codec.as_deref())
}

fn build_output_path(path: &str, format: &str, append: &str) -> Result<String, AppError> {
	let p = Path::new(path);
	let dir = p.parent().ok_or_else(|| AppError::Io("Cannot determine parent directory".to_string()))?;
	let stem = p.file_stem().and_then(|s| s.to_str()).ok_or_else(|| AppError::Io("Cannot determine filename".to_string()))?;
	let clean_append = append.replace(['/', '\\', '\0'], "");
	let suffix = if clean_append.is_empty() { String::new() } else { format!("-{}", clean_append) };
	let output = dir.join(format!("{}{}-audio.{}", stem, suffix, format));
	output.to_str().map(|s| s.to_string()).ok_or_else(|| AppError::Io("Invalid output path".to_string()))
}

async fn get_audio_codec(path: &str) -> Option<String> {
	let output = Command::new(ffprobe_path())
		.args([
			"-v",
			"error",
			"-select_streams",
			"a:0",
			"-show_entries",
			"stream=codec_name",
			"-of",
			"default=noprint_wrappers=1:nokey=1",
			path,
		])
		.output()
		.await;
	let output = match output {
		Ok(out) => out,
		Err(e) => {
			error!("Failed to run ffprobe: {}. Is FFmpeg/FFprobe installed and in your PATH?", e);
			return None;
		}
	};
	if !output.status.success() {
		return None;
	}
	let codec = String::from_utf8_lossy(&output.stdout).trim().to_string();
	if codec.is_empty() { None } else { Some(codec) }
}

pub(crate) async fn get_duration(path: &str) -> Option<f64> {
	let output = Command::new(ffprobe_path())
		.args(["-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", path])
		.output()
		.await;
	let output = match output {
		Ok(out) => out,
		Err(e) => {
			error!("Failed to get duration: {}", e);
			return None;
		}
	};
	if !output.status.success() {
		return None;
	}
	String::from_utf8_lossy(&output.stdout).trim().parse::<f64>().ok()
}

#[derive(serde::Serialize)]
struct VideoMetadata {
	duration_secs: f64,
	file_size_bytes: u64,
	audio_codec: Option<String>,
	needs_transcode: bool,
}

#[tauri::command]
async fn get_video_metadata(path: String) -> Result<VideoMetadata, AppError> {
	validate_path(&path)?;
	let duration_secs = get_duration(&path).await.unwrap_or(0.0);
	let file_size_bytes = tokio::fs::metadata(&path)
		.await
		.map(|m| m.len())
		.map_err(|e| AppError::Io(format!("Cannot read file metadata: {}", e)))?;
	let audio_codec = get_audio_codec(&path).await;
	let ext = Path::new(&path).extension().and_then(|e| e.to_str()).map(|e| e.to_lowercase()).unwrap_or_default();
	Ok(VideoMetadata {
		duration_secs,
		file_size_bytes,
		audio_codec,
		needs_transcode: needs_transcode(&ext),
	})
}

#[tauri::command]
async fn get_audio_waveform(path: String) -> Result<Vec<f32>, AppError> {
	validate_path(&path)?;

	let duration_secs = get_duration(&path).await.unwrap_or(0.0);
	if duration_secs <= 0.0 {
		return Ok(vec![0.0; WAVEFORM_BINS]);
	}

	// At 8 kHz mono s16le, total samples = duration * 8000.
	// Compute samples_per_bin so we can stream bin-by-bin.
	let total_samples = (duration_secs * 8000.0) as usize;
	let samples_per_bin = (total_samples / WAVEFORM_BINS).max(1);

	let mut child = Command::new(ffmpeg_path())
		.args(["-i", &path, "-vn", "-ac", "1", "-ar", "8000", "-f", "s16le", "pipe:1"])
		.stdout(Stdio::piped())
		.stderr(Stdio::null())
		.spawn()
		.map_err(|e| AppError::FfmpegFailed(format!("{}. Is FFmpeg installed?", e)))?;

	let mut peaks: Vec<f32> = Vec::with_capacity(WAVEFORM_BINS);
	let mut current_peak: u16 = 0;
	let mut samples_in_bin: usize = 0;

	if let Some(stdout) = child.stdout.take() {
		// Read in fixed-size chunks to avoid buffering the entire stream.
		let mut reader = BufReader::with_capacity(16 * 1024, stdout);
		let mut buf = [0u8; 8192];
		let mut leftover: Option<u8> = None;

		loop {
			let n = reader.read(&mut buf).await.map_err(|e| AppError::Io(format!("Failed to read FFmpeg output: {}", e)))?;
			if n == 0 {
				break;
			}

			let mut slice = &buf[..n];

			// If we had a leftover byte from the previous read, combine it
			if let Some(lo) = leftover.take()
				&& !slice.is_empty()
			{
				let sample = i16::from_le_bytes([lo, slice[0]]);
				current_peak = current_peak.max(sample.unsigned_abs());
				samples_in_bin += 1;
				if samples_in_bin >= samples_per_bin && peaks.len() < WAVEFORM_BINS {
					peaks.push(current_peak as f32);
					current_peak = 0;
					samples_in_bin = 0;
				}
				slice = &slice[1..];
			}

			// Process pairs of bytes as i16 samples
			let pairs = slice.len() / 2;
			for i in 0..pairs {
				let sample = i16::from_le_bytes([slice[i * 2], slice[i * 2 + 1]]);
				current_peak = current_peak.max(sample.unsigned_abs());
				samples_in_bin += 1;
				if samples_in_bin >= samples_per_bin && peaks.len() < WAVEFORM_BINS {
					peaks.push(current_peak as f32);
					current_peak = 0;
					samples_in_bin = 0;
				}
			}

			// Save leftover byte if odd number of bytes
			if slice.len() % 2 != 0 {
				leftover = Some(slice[slice.len() - 1]);
			}
		}
	}

	// Flush last partial bin
	if samples_in_bin > 0 && peaks.len() < WAVEFORM_BINS {
		peaks.push(current_peak as f32);
	}

	// Pad to WAVEFORM_BINS if stream was shorter than expected
	peaks.resize(WAVEFORM_BINS, 0.0);

	let _ = child.wait().await;

	// Normalize to 0.0–1.0
	let max_peak = peaks.iter().cloned().fold(0.0_f32, f32::max);
	if max_peak > 0.0 {
		for p in &mut peaks {
			*p /= max_peak;
		}
	}

	Ok(peaks)
}

#[tauri::command]
async fn extract_audio_range(app: tauri::AppHandle, path: String, start: String, end: String, format: String, append: String) -> Result<(), AppError> {
	validate_path(&path)?;
	validate_format(&format)?;
	validate_time(&start)?;
	validate_time(&end)?;

	let output = build_output_path(&path, &format, &append)?;
	let acodec = resolve_codec(&format, &path).await;
	let duration_secs = hms_to_seconds(&end)? - hms_to_seconds(&start)?;
	if duration_secs <= 0.0 {
		return Err(AppError::StartAfterEnd);
	}

	CANCEL_FLAG.store(false, Ordering::Release);
	info!("Extracting range: path={}, start={}, end={}, output={}", path, start, end, output);

	// Place -ss before -i for fast input seeking; use -t (duration) since
	// -to is relative to the input start when -ss precedes -i.
	let duration_str = format!("{:.3}", duration_secs);
	let args = vec!["-y", "-ss", &start, "-i", &path, "-t", &duration_str, "-vn", "-c:a", acodec, "-progress", "pipe:1", &output];

	run_ffmpeg_command(&args, duration_secs, &output, &app).await
}

#[tauri::command]
async fn extract_whole_audio(app: tauri::AppHandle, path: String, format: String, append: String) -> Result<(), AppError> {
	validate_path(&path)?;
	validate_format(&format)?;

	let output = build_output_path(&path, &format, &append)?;
	let acodec = resolve_codec(&format, &path).await;
	let duration_secs = get_duration(&path).await.unwrap_or(0.0);

	CANCEL_FLAG.store(false, Ordering::Release);
	info!("Extracting whole: path={}, output={}", path, output);

	let args = vec!["-y", "-i", &path, "-vn", "-c:a", acodec, "-progress", "pipe:1", &output];

	run_ffmpeg_command(&args, duration_secs, &output, &app).await
}

async fn run_ffmpeg_command(args: &[&str], duration_secs: f64, output_path: &str, app: &tauri::AppHandle) -> Result<(), AppError> {
	let mut child = Command::new(ffmpeg_path())
		.args(args)
		.stdout(Stdio::piped())
		.stderr(Stdio::piped())
		.spawn()
		.map_err(|e| AppError::FfmpegFailed(format!("{}. Is FFmpeg installed and in your PATH?", e)))?;

	// Take stderr before the progress loop so we can read it after waiting.
	let stderr_handle = child.stderr.take();
	let mut cancelled = false;

	// Read progress from stdout
	if let Some(stdout) = child.stdout.take() {
		let reader = BufReader::new(stdout);
		let mut lines = reader.lines();
		while let Ok(Some(line)) = lines.next_line().await {
			if CANCEL_FLAG.load(Ordering::Acquire) {
				info!("Cancel requested, killing FFmpeg");
				let _ = child.kill().await;
				cancelled = true;
				break;
			}
			if let Some(time_us_str) = line.strip_prefix("out_time_us=")
				&& duration_secs > 0.0
				&& let Ok(time_us) = time_us_str.parse::<f64>()
			{
				let progress = ((time_us / 1_000_000.0) / duration_secs * 100.0).min(100.0);
				let _ = app.emit("extraction-progress", progress);
			}
		}
	}

	if cancelled {
		let _ = tokio::fs::remove_file(output_path).await;
		return Err(AppError::Cancelled);
	}

	let status = child.wait().await.map_err(|e| AppError::FfmpegFailed(e.to_string()))?;

	debug!("FFmpeg exited: {}", status);
	if !status.success() {
		let mut stderr_buf = String::new();
		if let Some(mut stderr) = stderr_handle {
			let _ = stderr.read_to_string(&mut stderr_buf).await;
		}
		warn!("FFmpeg stderr: {}", stderr_buf);
		return Err(AppError::FfmpegFailed(stderr_buf));
	}

	let _ = app.emit("extraction-progress", 100.0_f64);
	info!("Extraction complete");
	Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
	let mut log_targets = vec![Target::new(TargetKind::Stdout)];
	if cfg!(debug_assertions) {
		log_targets.push(Target::new(TargetKind::Folder {
			path: std::env::current_dir().unwrap_or_default(),
			file_name: Some("audio-snip.log".into()),
		}));
	}

	tauri::Builder::default()
		.plugin(tauri_plugin_log::Builder::new().level(log::LevelFilter::Debug).targets(log_targets).build())
		.plugin(tauri_plugin_dialog::init())
		.plugin(tauri_plugin_opener::init())
		.setup(|_app| {
			let port = media_server::start();
			MEDIA_SERVER_PORT.store(port, Ordering::Relaxed);
			info!("Media server started on port {}", port);
			Ok(())
		})
		.invoke_handler(tauri::generate_handler![
			extract_audio_range,
			extract_whole_audio,
			cancel_extraction,
			get_media_server_port,
			register_media_path,
			get_video_metadata,
			get_audio_waveform,
		])
		.build(tauri::generate_context!())
		.expect("error while building tauri application")
		.run(|_app, event| {
			if let tauri::RunEvent::Exit = event {
				media_server::cleanup_transcode();
			}
		});
}

#[cfg(test)]
mod tests {
	use super::*;
	use std::sync::atomic::Ordering;

	/// UC-001 | BR-001: All supported video formats are accepted
	#[test]
	fn uc001_validate_extension_accepts_supported() {
		for ext in ALLOWED_VIDEO_EXTENSIONS {
			assert!(validate_extension(ext), "expected {ext} to be accepted");
		}
	}

	/// UC-001 | BR-001: Non-video formats are rejected
	#[test]
	fn uc001_validate_extension_rejects_unsupported() {
		for ext in ["txt", "mp3", "pdf", "jpg", "flac"] {
			assert!(!validate_extension(ext), "expected {ext} to be rejected");
		}
	}

	/// UC-001 | A4: Missing file is rejected
	#[test]
	fn uc001_validate_path_rejects_missing_file() {
		let result = validate_path("/nonexistent/video.mp4");
		assert!(result.is_err());
		assert!(result.unwrap_err().to_string().contains("File not found"));
	}

	#[test]
	fn uc001_validate_path_rejects_unsupported_extension() {
		let dir = tempfile::tempdir().unwrap();
		let path = dir.path().join("file.txt");
		std::fs::write(&path, "data").unwrap();
		let result = validate_path(path.to_str().unwrap());
		assert!(result.is_err());
		assert!(result.unwrap_err().to_string().contains("Unsupported video format"));
	}

	#[test]
	fn uc001_validate_path_accepts_valid_video() {
		let dir = tempfile::tempdir().unwrap();
		let path = dir.path().join("video.mp4");
		std::fs::write(&path, "data").unwrap();
		assert!(validate_path(path.to_str().unwrap()).is_ok());
	}

	// UC-002 | BR-002: Time Format
	#[test]
	fn uc002_validate_time_accepts_valid_hms() {
		assert!(validate_time("00:00:00").is_ok());
		assert!(validate_time("01:30:45").is_ok());
		assert!(validate_time("12:59:59").is_ok());
	}

	#[test]
	fn uc002_validate_time_rejects_two_parts() {
		let result = validate_time("10:30");
		assert!(result.is_err());
		assert!(result.unwrap_err().to_string().contains("expected HH:MM:SS"));
	}

	#[test]
	fn uc002_validate_time_rejects_non_numeric() {
		let result = validate_time("ab:cd:ef");
		assert!(result.is_err());
	}

	/// UC-002 | BR-002: Minutes and seconds must be 0-59
	#[test]
	fn uc002_validate_time_rejects_invalid_minutes() {
		let result = validate_time("00:60:00");
		assert!(result.is_err());
		assert!(result.unwrap_err().to_string().contains("0-59"));
	}

	/// UC-002 | BR-002: Minutes and seconds must be 0-59
	#[test]
	fn uc002_validate_time_rejects_invalid_seconds() {
		let result = validate_time("00:00:60");
		assert!(result.is_err());
		assert!(result.unwrap_err().to_string().contains("0-59"));
	}

	#[test]
	fn uc002_hms_to_seconds_zero() {
		assert_eq!(hms_to_seconds("00:00:00").unwrap(), 0.0);
	}

	#[test]
	fn uc002_hms_to_seconds_minutes_only() {
		assert_eq!(hms_to_seconds("00:05:30").unwrap(), 330.0);
	}

	#[test]
	fn uc002_hms_to_seconds_with_hours() {
		assert_eq!(hms_to_seconds("01:30:00").unwrap(), 5400.0);
	}

	// UC-002 | BR-003: Range Constraints
	#[test]
	fn uc002_range_start_before_end() {
		let start = hms_to_seconds("00:01:00").unwrap();
		let end = hms_to_seconds("00:05:00").unwrap();
		assert!(end > start);
	}

	/// UC-003 | BR-004: All supported output formats are accepted
	#[test]
	fn uc003_validate_format_accepts_supported() {
		for fmt in ALLOWED_FORMATS {
			assert!(validate_format(fmt).is_ok(), "expected {fmt} to be accepted");
		}
	}

	/// UC-003 | BR-004: Unsupported output formats are rejected
	#[test]
	fn uc003_validate_format_rejects_unsupported() {
		for fmt in ["wav", "flac", "wma", "m4a"] {
			assert!(validate_format(fmt).is_err(), "expected {fmt} to be rejected");
		}
	}

	// UC-003 | BR-005: Output Filename Convention
	#[test]
	fn uc003_build_output_path_basic() {
		let result = build_output_path("/tmp/video.mp4", "mp3", "").unwrap();
		assert_eq!(result, "/tmp/video-audio.mp3");
	}

	#[test]
	fn uc003_build_output_path_with_suffix() {
		let result = build_output_path("/tmp/video.mp4", "aac", "intro").unwrap();
		assert_eq!(result, "/tmp/video-intro-audio.aac");
	}

	#[test]
	fn uc003_build_output_path_empty_suffix() {
		let result = build_output_path("/home/user/my_video.mkv", "ogg", "").unwrap();
		assert_eq!(result, "/home/user/my_video-audio.ogg");
	}

	/// UC-003 | BR-005: Path separators in suffix are stripped to prevent path traversal
	#[test]
	fn uc003_build_output_path_sanitizes_path_separators() {
		let result = build_output_path("/tmp/video.mp4", "mp3", "foo/../../etc").unwrap();
		assert_eq!(result, "/tmp/video-foo....etc-audio.mp3");
	}

	#[test]
	fn uc003_build_output_path_sanitizes_backslash() {
		let result = build_output_path("/tmp/video.mp4", "mp3", "a\\b").unwrap();
		assert_eq!(result, "/tmp/video-ab-audio.mp3");
	}

	#[test]
	fn uc003_build_output_path_sanitizes_null_byte() {
		let result = build_output_path("/tmp/video.mp4", "mp3", "a\0b").unwrap();
		assert_eq!(result, "/tmp/video-ab-audio.mp3");
	}

	// UC-003 | BR-006: Codec Optimization
	#[test]
	fn uc003_codec_for_format_aac_copy_when_source_aac() {
		assert_eq!(codec_for_format("aac", Some("aac")), "copy");
	}

	#[test]
	fn uc003_codec_for_format_aac_encode_when_source_other() {
		assert_eq!(codec_for_format("aac", Some("mp3")), "aac");
		assert_eq!(codec_for_format("aac", None), "aac");
	}

	#[test]
	fn uc003_codec_for_format_mp3() {
		assert_eq!(codec_for_format("mp3", None), "libmp3lame");
		assert_eq!(codec_for_format("mp3", Some("aac")), "libmp3lame");
	}

	#[test]
	fn uc003_codec_for_format_ogg() {
		assert_eq!(codec_for_format("ogg", None), "libopus");
		assert_eq!(codec_for_format("ogg", Some("aac")), "libopus");
	}

	// UC-004 | BR-007: Partial File Cleanup
	#[test]
	fn uc004_cancel_flag_store_and_load() {
		CANCEL_FLAG.store(false, Ordering::Release);
		assert!(!CANCEL_FLAG.load(Ordering::Acquire));
		CANCEL_FLAG.store(true, Ordering::Release);
		assert!(CANCEL_FLAG.load(Ordering::Acquire));
		CANCEL_FLAG.store(false, Ordering::Release);
	}
}
