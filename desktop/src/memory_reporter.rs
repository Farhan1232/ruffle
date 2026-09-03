//! Periodic memory sampling, used to measure SWF retention over a play session.
//!
//! Enabled with `--memory-report <FILE>`. Every interval it writes one CSV row
//! combining the process' resident set size with Ruffle's own accounting of
//! what each still-loaded movie is keeping alive, so that growth in RSS can be
//! attributed to (or ruled out as) movies that were supposed to be unloaded.

use ruffle_core::Player;
use ruffle_core::memory_report::MemoryReport;
use std::alloc::{GlobalAlloc, Layout, System};
use std::fmt::Write as _;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// The system allocator with a running total of live bytes, so that the
/// memory report can say how much of the process is Rust heap as opposed to
/// graphics driver and other native memory.
pub struct CountingAllocator;

static LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            LIVE_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE_BYTES.fetch_sub(layout.size(), Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() {
            LIVE_BYTES.fetch_sub(layout.size(), Ordering::Relaxed);
            LIVE_BYTES.fetch_add(new_size, Ordering::Relaxed);
        }
        new_ptr
    }
}

/// Bytes currently allocated through the Rust allocator.
pub fn rust_heap_bytes() -> usize {
    LIVE_BYTES.load(Ordering::Relaxed)
}

/// Frame intervals seen since the last sample, in milliseconds, so that a
/// run can be judged on smoothness as well as on footprint.
#[derive(Default)]
struct FrameTimes {
    samples: Vec<f64>,
    last_frame: Option<Instant>,
}

impl FrameTimes {
    /// Records the gap since the previous frame. Called once per tick.
    fn record(&mut self, now: Instant) {
        if let Some(last) = self.last_frame {
            self.samples
                .push(now.duration_since(last).as_secs_f64() * 1000.0);
        }
        self.last_frame = Some(now);
    }

    /// `(count, mean, p50, p95, p99, max, frames over 33 ms, over 100 ms)`,
    /// draining what has been recorded so far.
    fn drain_stats(&mut self) -> (usize, f64, f64, f64, f64, f64, usize, usize) {
        if self.samples.is_empty() {
            return (0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0);
        }
        let long = self.samples.iter().filter(|ms| **ms > 33.0).count();
        let very_long = self.samples.iter().filter(|ms| **ms > 100.0).count();
        let sum: f64 = self.samples.iter().sum();
        let count = self.samples.len();
        self.samples
            .sort_by(|a, b| a.partial_cmp(b).expect("Frame times are finite"));
        let at = |q: f64| {
            let i = ((count as f64 - 1.0) * q).round() as usize;
            self.samples[i]
        };
        let stats = (
            count,
            sum / count as f64,
            at(0.50),
            at(0.95),
            at(0.99),
            self.samples[count - 1],
            long,
            very_long,
        );
        self.samples.clear();
        stats
    }
}

pub struct MemoryReporter {
    output: BufWriter<File>,
    interval: Duration,
    started: Instant,
    last_sample: Option<Instant>,
    /// Retained bytes at the first sample, so the log states growth directly.
    baseline_retained: Option<usize>,
    frame_times: FrameTimes,
    /// Cumulative created-texture bytes at the previous sample, for the churn
    /// rate between the two.
    last_created_bytes: Option<u64>,
    last_elapsed: f64,
    /// The header names one group of columns per texture kind, and only the
    /// renderer knows those names, so it is written with the first sample.
    header_written: bool,
    /// Running maxima across samples, so a row states the high-water mark
    /// beside the current value.
    peak_gpu_texture_bytes: usize,
    peak_rss: usize,
}

impl MemoryReporter {
    pub fn new(path: &Path, interval: Duration) -> Result<Self, std::io::Error> {
        let output = BufWriter::new(File::create(path)?);
        tracing::info!(
            "Memory instrumentation {} active",
            ruffle_core::memory_report::INSTRUMENTATION_VERSION
        );

        Ok(Self {
            output,
            interval,
            started: Instant::now(),
            last_sample: None,
            baseline_retained: None,
            frame_times: FrameTimes::default(),
            last_created_bytes: None,
            last_elapsed: 0.0,
            header_written: false,
            peak_gpu_texture_bytes: 0,
            peak_rss: 0,
        })
    }

    /// Takes a sample if the interval has elapsed. Cheap to call every frame.
    pub fn maybe_sample(&mut self, player: &mut Player) {
        let now = Instant::now();
        self.frame_times.record(now);
        if let Some(last) = self.last_sample
            && now.duration_since(last) < self.interval
        {
            return;
        }
        self.last_sample = Some(now);

        let report = player.mutate_with_update_context(MemoryReport::capture);
        let elapsed = now.duration_since(self.started).as_secs_f64();
        let rss = resident_set_size().unwrap_or(0);
        self.peak_rss = self.peak_rss.max(rss);
        self.peak_gpu_texture_bytes = self.peak_gpu_texture_bytes.max(report.gpu_texture_bytes);

        if !self.header_written {
            self.header_written = true;
            if let Err(e) = writeln!(
                self.output,
                "rss_bytes,peak_rss_bytes,{},peak_gpu_texture_bytes_sampled,rust_heap_bytes,\
                 frames,frame_ms_mean,frame_ms_p50,frame_ms_p95,frame_ms_p99,frame_ms_max,\
                 frames_over_33ms,frames_over_100ms",
                MemoryReport::csv_header_for(report.texture_kind_names())
            ) {
                tracing::error!("Could not write memory report header: {e}");
            }
        }

        let rust_heap = rust_heap_bytes();
        let (frames, mean, p50, p95, p99, max, long, very_long) = self.frame_times.drain_stats();
        if let Err(e) = writeln!(
            self.output,
            "{},{},{},{},{},{},{mean:.2},{p50:.2},{p95:.2},{p99:.2},{max:.2},{long},{very_long}",
            rss,
            self.peak_rss,
            report.to_csv_row(elapsed),
            self.peak_gpu_texture_bytes,
            rust_heap,
            frames,
        )
        .and_then(|_| self.output.flush())
        {
            tracing::error!("Could not write memory report: {e}");
        }

        let retained = report.swf_bytes + report.bitmap_decoded_bytes;
        let baseline = *self.baseline_retained.get_or_insert(retained);
        let mut kind_breakdown = String::new();
        for (i, name) in report.texture_kind_names().iter().enumerate() {
            let bytes = report
                .texture_kind_live_bytes
                .get(i)
                .copied()
                .unwrap_or_default();
            let count = report
                .texture_kind_live_counts
                .get(i)
                .copied()
                .unwrap_or_default();
            let _ = write!(
                kind_breakdown,
                " [{name} {} MiB/{count}]",
                bytes / (1024 * 1024)
            );
        }
        let churn_mib_per_s = self
            .last_created_bytes
            .replace(report.texture_bytes_created)
            .map(|previous| {
                let span = elapsed - self.last_elapsed;
                if span > 0.0 {
                    (report.texture_bytes_created.saturating_sub(previous) as f64)
                        / (1024.0 * 1024.0)
                        / span
                } else {
                    0.0
                }
            })
            .unwrap_or(0.0);
        self.last_elapsed = elapsed;

        tracing::info!(
            "memory @{elapsed:.0}s: rss {} MiB (rust heap {} MiB), {} movies retaining {} MiB \
             (+{} MiB since first sample), {} pending loaders, {} class aliases, \
             gc {} MiB / {} objects (+{} MiB external), gpu {} textures {} MiB + buffers {} MiB, \
             {} meshes {} MiB, textures live {} MiB (peak {} MiB){}, \
             pools idle: main {} MiB / {} sizes, offscreen {} MiB / {} sizes, buffers {} MiB, \
             pool reuse {}/{}, churn {} MiB/s, \
             frame ms mean {:.1} p95 {:.1} p99 {:.1} max {:.1} ({} over 33 ms){}{}",
            rss / (1024 * 1024),
            rust_heap / (1024 * 1024),
            report.movies.len(),
            retained / (1024 * 1024),
            retained.saturating_sub(baseline) / (1024 * 1024),
            report.pending_loaders,
            report.class_aliases,
            report.gc_allocation / (1024 * 1024),
            report.gc_objects,
            report.gc_external_bytes / (1024 * 1024),
            report.gpu_textures,
            report.gpu_texture_bytes / (1024 * 1024),
            report.gpu_buffer_bytes / (1024 * 1024),
            report.meshes,
            report.mesh_bytes / (1024 * 1024),
            report.tracked_texture_bytes / (1024 * 1024),
            report.peak_texture_bytes / (1024 * 1024),
            kind_breakdown,
            report.main_pool_idle_bytes / (1024 * 1024),
            report.main_pool_size_classes,
            report.offscreen_pool_idle_bytes / (1024 * 1024),
            report.offscreen_pool_size_classes,
            report.buffer_pool_idle_bytes / (1024 * 1024),
            report.pool_reuses,
            report.pool_misses,
            churn_mib_per_s,
            mean,
            p95,
            p99,
            max,
            long,
            report.top_pool_classes(),
            report.top_movies(5),
        );
    }
}

/// Resident set size of this process, in bytes.
#[cfg(target_os = "linux")]
fn resident_set_size() -> Option<usize> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|line| line.starts_with("VmRSS:"))?;
    let kb: usize = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb * 1024)
}

/// Working set size of this process, in bytes.
#[cfg(target_os = "windows")]
fn resident_set_size() -> Option<usize> {
    use windows_sys::Win32::System::ProcessStatus::{
        GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    let mut counters: PROCESS_MEMORY_COUNTERS = unsafe { std::mem::zeroed() };
    counters.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
    let ok = unsafe {
        GetProcessMemoryInfo(
            GetCurrentProcess(),
            &mut counters,
            std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        )
    };
    (ok != 0).then_some(counters.WorkingSetSize)
}

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
fn resident_set_size() -> Option<usize> {
    None
}
