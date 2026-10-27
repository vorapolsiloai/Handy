//! The worker process: the only place transcribe.cpp native code runs. Owns
//! the model, session and stream, and serves [`Request`]s until the parent
//! closes its stdin.
//!
//! A dedicated thread reads stdin so the parent's writes never block, and
//! the worker exits the moment the parent goes away, even if hung.

use super::protocol::{
    read_message, write_message, DeviceInfo, DeviceSelector, LoadedInfo, Request, Response,
};
use super::{CPU_ONLY_FLAG, LOG_LEVEL_ENV};
use log::{debug, error, warn, LevelFilter, Log, Metadata, Record};
use std::fs::File;
use std::io::{self, BufReader, Write};
use std::path::Path;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};
use transcribe_cpp::{
    Backend, BackendMask, DeviceType, Feature, Model, ModelOptions, RunOptions, Session, Stream,
    StreamOptions,
};

/// Prefix on worker log lines so the parent can tell them apart from raw
/// native output (e.g. a `GGML_ASSERT` message right before an abort).
pub(super) const LOG_LINE_PREFIX: &str = "\u{1}";

/// A request as handed from the stdin reader to the main loop.
struct Incoming {
    request: Request,
    pcm: Vec<f32>,
}

pub fn run() -> i32 {
    // The worker owns the per-run engine scratch, so it needs the same glibc
    // tuning as the app (#1792). Before anything allocates.
    crate::memory::init_allocator();
    #[cfg(unix)]
    ignore_app_signals();
    #[cfg(target_os = "linux")]
    set_process_name();
    // Take the real stdout for the protocol before any native code runs, and
    // point fd 1 at stderr: ggml writes to stdout in places, and any stray
    // byte there would corrupt the protocol stream.
    let protocol_out = match take_stdout_for_protocol() {
        Ok(file) => file,
        Err(e) => {
            eprintln!("transcribe worker: failed to claim stdout: {e}");
            return 2;
        }
    };
    init_logger();

    let (requests_tx, requests) = mpsc::channel();
    if let Err(e) = thread::Builder::new()
        .name("transcribe-worker-in".into())
        .spawn(move || read_requests(requests_tx))
    {
        error!("Failed to start the request reader: {}", e);
        return 2;
    }

    transcribe_cpp::init_logging();
    // Registering a GPU backend runs its driver code, so this is where a
    // broken driver crashes or hangs, before the hello is answered.
    let cpu_only = std::env::args_os().any(|arg| arg == CPU_ONLY_FLAG);
    inject_fault("init", !cpu_only);
    let init = if cpu_only {
        transcribe_cpp::init_backends_with(None::<&Path>, BackendMask::CPU)
    } else {
        transcribe_cpp::init_backends_default()
    };
    // Init only fails when no compute device registered at all, which no
    // other worker could fix either: the hello reports it.
    let init_error = init
        .err()
        .map(|e| format!("Failed to initialize transcribe.cpp backends: {e}"));
    debug!(
        "transcribe.cpp allowed backends: {:#x}",
        transcribe_cpp::allowed_backends().bits()
    );

    let mut output = protocol_out;
    let mut session: Option<(Session, LoadedInfo)> = None;

    while let Ok(Incoming { request, pcm }) = requests.recv() {
        let finished_run = matches!(request, Request::Run { .. });
        let response = match request {
            Request::Hello { list_devices } => match &init_error {
                Some(message) => Response::Error(message.clone()),
                None => Response::Hello {
                    devices: list_devices.then(|| {
                        inject_fault("list", !cpu_only);
                        transcribe_cpp::devices()
                            .iter()
                            .map(DeviceInfo::from_device)
                            .collect()
                    }),
                },
            },
            Request::Load {
                path,
                backend,
                device,
            } => {
                // Free any previous model first to avoid holding two at once.
                session = None;
                match load(&path, backend, device) {
                    Ok(mut loaded) => {
                        if loaded.1.on_gpu {
                            warm_up(&mut loaded.0, &loaded.1);
                        }
                        let info = loaded.1.clone();
                        session = Some(loaded);
                        Response::Loaded(info)
                    }
                    Err(e) => Response::Error(e),
                }
            }
            Request::Run { options } => match session.as_mut() {
                Some((session, info)) => {
                    inject_fault("run", info.on_gpu);
                    match session.run(&pcm, &options) {
                        Ok(transcript) => Response::Transcript(transcript),
                        Err(e) => Response::Error(e.to_string()),
                    }
                }
                None => not_loaded(),
            },
            Request::StreamBegin { run, stream } => match session.as_mut() {
                Some((session, info)) => match session.stream(&run, &stream) {
                    Ok(stream) => {
                        if write_message(&mut output, &Response::Ok, None).is_err() {
                            return 1;
                        }
                        let on_gpu = info.on_gpu;
                        if let Err(e) = serve_stream(stream, &requests, &mut output, on_gpu) {
                            error!("Protocol write failed during stream: {}", e);
                            return 1;
                        }
                        crate::memory::trim_freed_memory();
                        continue;
                    }
                    Err(e) => Response::Error(e.to_string()),
                },
                None => not_loaded(),
            },
            Request::Feed | Request::Finalize { .. } | Request::StreamReset => {
                Response::Error("no active stream".to_string())
            }
        };
        if let Err(e) = write_message(&mut output, &response, None) {
            error!("Protocol write failed: {}", e);
            return 1;
        }
        // After the reply, so it never delays the transcript.
        if finished_run {
            drop(pcm);
            crate::memory::trim_freed_memory();
        }
    }
    // Unreachable in practice: the reader exits the process on EOF.
    0
}

/// The stdin reader. Forwards requests to the main loop, and ends the
/// process when the parent closes stdin (unload, quit, or the parent died).
/// Cancels never reach the worker: the parent kills it instead. `_exit`
/// skips C++ static destructors, so a model still alive at that point can't
/// trip ggml-metal's teardown asserts, and the OS reclaims its CPU and GPU
/// memory.
fn read_requests(requests: mpsc::Sender<Incoming>) -> ! {
    let mut input = BufReader::new(io::stdin().lock());
    loop {
        match read_message::<Request>(&mut input) {
            Ok(Some((request, pcm))) => {
                if requests.send(Incoming { request, pcm }).is_err() {
                    exit_now(1);
                }
            }
            Ok(None) => exit_now(0),
            Err(e) => {
                error!("Protocol read failed: {}", e);
                exit_now(1);
            }
        }
    }
}

fn exit_now(code: i32) -> ! {
    let _ = io::stderr().flush();
    // SAFETY: `_exit` only ends the process; nothing runs afterwards.
    unsafe { libc::_exit(code) }
}

/// Serve requests against an active stream until it is finalized or reset.
fn serve_stream(
    mut stream: Stream<'_>,
    requests: &mpsc::Receiver<Incoming>,
    output: &mut impl Write,
    on_gpu: bool,
) -> io::Result<()> {
    while let Ok(Incoming { request, pcm }) = requests.recv() {
        let (response, done) = match request {
            Request::Feed => {
                inject_fault("feed", on_gpu);
                match stream.feed(&pcm) {
                    Ok(update) => {
                        let text = (update.committed_changed || update.tentative_changed)
                            .then(|| stream.text());
                        (Response::Fed { update, text }, false)
                    }
                    Err(e) => (Response::Error(e.to_string()), false),
                }
            }
            Request::Finalize { want_language } => {
                inject_fault("finalize", on_gpu);
                match stream.finalize() {
                    Ok(update) => {
                        let language = if want_language {
                            stream.snapshot().language
                        } else {
                            None
                        };
                        let text = stream.text();
                        (
                            Response::Finalized {
                                update,
                                text,
                                language,
                            },
                            true,
                        )
                    }
                    // Finalize ends the stream even when it fails.
                    Err(e) => (Response::Error(e.to_string()), true),
                }
            }
            Request::StreamReset => {
                stream.reset();
                (Response::Ok, true)
            }
            _ => (
                Response::Error("a stream is active; finalize or reset it first".to_string()),
                false,
            ),
        };
        write_message(output, &response, None)?;
        if done {
            break;
        }
    }
    Ok(())
}

/// `0`/`off` skips the GPU warm-up after a model load; a comma-separated
/// list of seconds (e.g. `1,5.12,10`) replaces the default clip lengths.
const WARMUP_ENV: &str = "HANDY_GPU_WARMUP";

/// Silent clip lengths (seconds) run once right after a GPU model load.
/// GPU backends (ggml-vulkan in particular) build compute pipelines lazily,
/// per shape class, on the first graph that needs them. With a cold driver
/// shader cache (new build, driver update) that took 1.5-3.1 s on the first
/// ~1 s clip and 0.6-0.9 s more on the first ~5 s clip (Radeon 8060S,
/// parakeet/cohere Q8_0), and would otherwise land on the user's first
/// dictations. Longer clips (10-40 s) built nothing new.
const WARMUP_SECS: &[f32] = &[1.0, 5.12];

/// No new warm-up run starts after this much time: the load already
/// overlaps the user's speech, but a slow GPU must not hold the first
/// result back much longer. With a warm driver cache the default clips take
/// ~0.2-0.4 s; with a cold one they took up to 2.6 s, time the first
/// dictations would otherwise pay. The 1 s and 5.12 s clips build different
/// pipelines, so the budget must let both run on a cold cache.
const WARMUP_BUDGET: Duration = Duration::from_millis(4000);

/// The clip lengths to warm up with, or `None` when disabled.
fn warmup_lengths() -> Option<Vec<f32>> {
    parse_warmup_lengths(std::env::var(WARMUP_ENV).ok().as_deref())
}

fn parse_warmup_lengths(value: Option<&str>) -> Option<Vec<f32>> {
    let Some(value) = value else {
        return Some(WARMUP_SECS.to_vec());
    };
    let value = value.trim().to_ascii_lowercase();
    match value.as_str() {
        "0" | "off" | "false" | "no" => None,
        "" | "1" | "on" | "true" | "yes" => Some(WARMUP_SECS.to_vec()),
        list => {
            let secs: Vec<f32> = list
                .split(',')
                .filter_map(|s| s.trim().parse::<f32>().ok())
                .filter(|s| *s > 0.0 && *s <= 60.0)
                .collect();
            Some(if secs.is_empty() {
                WARMUP_SECS.to_vec()
            } else {
                secs
            })
        }
    }
}

/// Runs silent clips (and a short stream on streaming models) through the
/// freshly loaded session so the GPU's one-off pipeline builds happen during
/// the load, not the first transcription. Best-effort: any failure only logs.
fn warm_up(session: &mut Session, info: &LoadedInfo) {
    let Some(lengths) = warmup_lengths() else {
        log::info!("GPU warm-up skipped ({WARMUP_ENV})");
        return;
    };
    let started = Instant::now();
    let options = RunOptions::default();
    let mut runs = 0;
    for secs in lengths {
        if started.elapsed() >= WARMUP_BUDGET {
            break;
        }
        let pcm = vec![0.0f32; (secs * 16_000.0) as usize];
        let t = Instant::now();
        if let Err(e) = session.run(&pcm, &options) {
            warn!("GPU warm-up run ({secs}s) failed: {e}");
            return;
        }
        runs += 1;
        debug!("GPU warm-up {secs}s took {} ms", t.elapsed().as_millis());
    }
    if info.capabilities.supports_streaming && started.elapsed() < WARMUP_BUDGET {
        let t = Instant::now();
        match session.stream(&options, &StreamOptions::default()) {
            Ok(mut stream) => {
                let pcm = vec![0.0f32; 16_000];
                let result = stream
                    .feed(&pcm)
                    .and_then(|_| stream.feed(&pcm))
                    .and_then(|_| stream.finalize());
                if let Err(e) = result {
                    warn!("GPU warm-up stream failed: {e}");
                }
                debug!("GPU warm-up stream took {} ms", t.elapsed().as_millis());
            }
            Err(e) => warn!("GPU warm-up stream could not start: {e}"),
        }
    }
    log::info!(
        "GPU warm-up: {runs} run(s) in {} ms",
        started.elapsed().as_millis()
    );
}

fn not_loaded() -> Response {
    Response::Error("no model loaded".to_string())
}

fn load(
    path: &Path,
    backend: Backend,
    selector: DeviceSelector,
) -> Result<(Session, LoadedInfo), String> {
    let device = match selector {
        DeviceSelector::Auto => None,
        DeviceSelector::Key(key) => {
            let found = transcribe_cpp::devices().into_iter().find(|d| {
                let info = DeviceInfo::from_device(d);
                info.is_gpu() && info.key == key
            });
            if found.is_none() {
                warn!(
                    "Stored transcribe GPU device '{}' is no longer available; using automatic device selection",
                    key
                );
            }
            found
        }
        DeviceSelector::Index(index) => {
            let device = transcribe_cpp::devices()
                .into_iter()
                .find(|d| d.index == Some(index))
                .ok_or_else(|| {
                    format!("No compute device with index {index} (see --list-devices)")
                })?;
            if matches!(device.device_type, DeviceType::Accel | DeviceType::Unknown) {
                return Err(format!(
                    "Device index {index} ({}) cannot host a model",
                    device.kind
                ));
            }
            Some(device)
        }
    };

    inject_fault("load", backend != Backend::Cpu);
    let model = Model::load_with(path, &ModelOptions { backend, device })
        .map_err(|e| format!("Failed to load model: {e}"))?;
    let session = model
        .session()
        .map_err(|e| format!("Failed to create session: {e}"))?;
    let bound = model.device().ok().map(|d| DeviceInfo::from_device(&d));
    let backend_name = model.backend();
    let info = LoadedInfo {
        arch: model.arch(),
        variant: model.variant(),
        on_gpu: match &bound {
            Some(device) => device.is_gpu(),
            None => !backend_name.is_empty() && !backend_name.eq_ignore_ascii_case("cpu"),
        },
        backend: backend_name,
        device: bound
            .as_ref()
            .map(|d| d.label().to_string())
            .unwrap_or_else(|| "unknown".to_string()),
        capabilities: model
            .capabilities()
            .map_err(|e| format!("Failed to read model capabilities: {e}"))?,
        supports_initial_prompt: model.supports(Feature::InitialPrompt),
        supports_long_form: model.supports(Feature::LongForm),
        supports_vocabulary: model.supports(Feature::Vocabulary),
    };
    Ok((session, info))
}

/// Debug-build fault injection for exercising crash/hang recovery:
/// `HANDY_WORKER_FAULT=<abort|segv|hang>@<init|list|load|run|feed|finalize>`
/// (`abort` is what `GGML_ASSERT` ends in, `segv` a real invalid memory write)
/// (`init` is backend registration, before the hello). With `HANDY_WORKER_FAULT_ONCE=<marker
/// path>` it fires only in the first worker to reach that stage (the marker
/// file records that it fired). With `HANDY_WORKER_FAULT_GPU_ONLY=1` it fires
/// only on a GPU (backend init and device listing count as GPU work unless
/// the worker is CPU-only), so recovery on CPU can succeed.
#[cfg(debug_assertions)]
fn inject_fault(stage: &str, on_gpu: bool) {
    let Ok(spec) = std::env::var("HANDY_WORKER_FAULT") else {
        return;
    };
    let Some((kind, at)) = spec.split_once('@') else {
        return;
    };
    if at != stage {
        return;
    }
    if !on_gpu && std::env::var_os("HANDY_WORKER_FAULT_GPU_ONLY").is_some() {
        return;
    }
    if let Some(marker) = std::env::var_os("HANDY_WORKER_FAULT_ONCE") {
        if std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&marker)
            .is_err()
        {
            return;
        }
    }
    eprintln!("injected fault: {kind} at {stage}");
    match kind {
        "abort" => std::process::abort(),
        // SAFETY: deliberately crash the worker with a real invalid memory
        // write, as a faulting driver does: SIGSEGV on Unix, an access
        // violation on Windows (where raising SIGSEGV only exits with code
        // 3). On Unix, restore the default disposition first so Rust's
        // stack-overflow handler doesn't get involved.
        "segv" => unsafe {
            #[cfg(unix)]
            libc::signal(libc::SIGSEGV, libc::SIG_DFL);
            std::ptr::write_volatile(std::ptr::null_mut::<u32>(), 1);
        },
        "hang" => loop {
            std::thread::sleep(std::time::Duration::from_secs(3600));
        },
        _ => {}
    }
}

#[cfg(not(debug_assertions))]
fn inject_fault(_stage: &str, _on_gpu: bool) {}

/// Logs go to stderr as `\x01LEVEL\ttarget\tmessage` lines; the parent
/// re-logs them at the same level.
struct StderrLogger;

impl Log for StderrLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.level() <= log::max_level()
    }

    fn log(&self, record: &Record) {
        if !self.enabled(record.metadata()) {
            return;
        }
        let message = record.args().to_string();
        let mut stderr = io::stderr().lock();
        for line in message.lines() {
            let _ = writeln!(
                stderr,
                "{LOG_LINE_PREFIX}{}\t{}\t{}",
                record.level(),
                record.target(),
                line
            );
        }
    }

    fn flush(&self) {
        let _ = io::stderr().flush();
    }
}

fn init_logger() {
    static LOGGER: StderrLogger = StderrLogger;
    let level = std::env::var(LOG_LEVEL_ENV)
        .ok()
        .and_then(|v| v.parse::<LevelFilter>().ok())
        .unwrap_or(LevelFilter::Info);
    if log::set_logger(&LOGGER).is_ok() {
        log::set_max_level(level);
    }
}

/// SIGUSR2 toggles transcription in the app (and WebKitGTK uses SIGUSR1).
/// Aimed at Handy by name, e.g. the README's `pkill -USR2 -n handy`, one
/// can reach this worker instead, and its default action would kill it.
#[cfg(unix)]
fn ignore_app_signals() {
    // SAFETY: setting a signal's disposition to SIG_IGN has no preconditions.
    unsafe {
        libc::signal(libc::SIGUSR1, libc::SIG_IGN);
        libc::signal(libc::SIGUSR2, libc::SIG_IGN);
    }
}

/// Started from /proc/self/exe, the kernel names this process "exe". Name it
/// as Handy's, but not `handy`: `pkill`/`killall` match names case-sensitively,
/// so `pkill -USR2 -n handy` keeps reaching the app, never this newer process.
#[cfg(target_os = "linux")]
fn set_process_name() {
    // SAFETY: PR_SET_NAME copies a NUL-terminated name of up to 16 bytes.
    unsafe {
        libc::prctl(libc::PR_SET_NAME, c"Handy-worker".as_ptr());
    }
}

#[cfg(unix)]
fn take_stdout_for_protocol() -> io::Result<File> {
    use std::os::fd::FromRawFd;
    // SAFETY: plain fd duplication; the new fd is owned by the returned File.
    unsafe {
        let fd = libc::dup(libc::STDOUT_FILENO);
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        if libc::dup2(libc::STDERR_FILENO, libc::STDOUT_FILENO) < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(File::from_raw_fd(fd))
    }
}

#[cfg(windows)]
fn take_stdout_for_protocol() -> io::Result<File> {
    use std::os::windows::io::FromRawHandle;
    // Native code writes through the CRT's fd 1, so redirect that one. The
    // protocol uses the OS handle behind a CRT duplicate of the original.
    // SAFETY: CRT fd duplication; the duplicate's handle is owned by the File
    // for the rest of the process.
    unsafe {
        let fd = libc::dup(1);
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        if libc::dup2(2, 1) < 0 {
            return Err(io::Error::last_os_error());
        }
        let handle = libc::get_osfhandle(fd);
        if handle == -1 {
            return Err(io::Error::last_os_error());
        }
        Ok(File::from_raw_handle(handle as _))
    }
}

#[cfg(test)]
mod warmup_tests {
    use super::{parse_warmup_lengths, WARMUP_SECS};

    #[test]
    fn default_when_unset_or_on() {
        assert_eq!(parse_warmup_lengths(None), Some(WARMUP_SECS.to_vec()));
        assert_eq!(parse_warmup_lengths(Some("on")), Some(WARMUP_SECS.to_vec()));
        assert_eq!(parse_warmup_lengths(Some("")), Some(WARMUP_SECS.to_vec()));
    }

    #[test]
    fn disabled() {
        assert_eq!(parse_warmup_lengths(Some("0")), None);
        assert_eq!(parse_warmup_lengths(Some(" OFF ")), None);
    }

    #[test]
    fn custom_list_drops_invalid_entries() {
        assert_eq!(
            parse_warmup_lengths(Some("1, 5.12,x,-3,99,10")),
            Some(vec![1.0, 5.12, 10.0])
        );
        assert_eq!(parse_warmup_lengths(Some("x")), Some(WARMUP_SECS.to_vec()));
    }
}
