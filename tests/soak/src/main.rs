//! Plays a SWF headlessly for a while and reports what the renderer costs.
//!
//! The desktop player only renders when its window is on screen, which makes it
//! useless for an unattended soak: on a machine whose display is covered, or
//! under Xvfb, the movie ticks along and nothing is ever drawn. This runs the
//! same player over an offscreen target instead, so every frame is really
//! rendered, and writes the same columns the `--memory-report` CSV does plus
//! the frame times.
//!
//! Measurement only. It is not part of the player.

use anyhow::{Result, anyhow};
use clap::Parser;
use ruffle_core::backend::locale::DeterministicLocaleBackend;
use ruffle_core::limits::ExecutionLimit;
use ruffle_core::memory_report::MemoryReport;
use ruffle_core::tag_utils::movie_from_path;
use ruffle_core::{Player, PlayerBuilder};
use ruffle_render_wgpu::backend::{
    WgpuRenderBackend, create_wgpu_instance, request_adapter_and_device,
};
use ruffle_render_wgpu::descriptors::Descriptors;
use ruffle_render_wgpu::target::TextureTarget;
use ruffle_render_wgpu::wgpu;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Parser)]
struct Opt {
    /// The SWF to play.
    swf: PathBuf,
    /// How long to play it for.
    #[arg(long, default_value_t = 60)]
    seconds: u64,
    /// Viewport width; the client plays at 1920x985.
    #[arg(long, default_value_t = 1920)]
    width: u32,
    #[arg(long, default_value_t = 985)]
    height: u32,
    /// Seconds between samples.
    #[arg(long, default_value_t = 5.0)]
    interval: f64,
    /// Where to write the CSV.
    #[arg(long)]
    csv: Option<PathBuf>,
    /// Play at the movie's frame rate rather than as fast as the renderer can.
    #[arg(long)]
    realtime: bool,
}

/// Frame intervals since the last sample.
#[derive(Default)]
struct FrameTimes(Vec<f64>);

impl FrameTimes {
    fn record(&mut self, ms: f64) {
        self.0.push(ms);
    }

    /// `(count, mean, p50, p95, p99, max, over 33 ms, over 100 ms)`.
    fn drain(&mut self) -> (usize, f64, f64, f64, f64, f64, usize, usize) {
        if self.0.is_empty() {
            return (0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0);
        }
        let long = self.0.iter().filter(|ms| **ms > 33.0).count();
        let very_long = self.0.iter().filter(|ms| **ms > 100.0).count();
        let sum: f64 = self.0.iter().sum();
        let count = self.0.len();
        self.0.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        let at = |q: f64| self.0[((count as f64 - 1.0) * q).round() as usize];
        let stats = (
            count,
            sum / count as f64,
            at(0.50),
            at(0.95),
            at(0.99),
            self.0[count - 1],
            long,
            very_long,
        );
        self.0.clear();
        stats
    }
}

fn resident_set_size() -> usize {
    let Ok(status) = std::fs::read_to_string("/proc/self/status") else {
        return 0;
    };
    status
        .lines()
        .find(|line| line.starts_with("VmRSS:"))
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|kb| kb.parse::<usize>().ok())
        .map(|kb| kb * 1024)
        .unwrap_or(0)
}

fn private_bytes() -> usize {
    let Ok(rollup) = std::fs::read_to_string("/proc/self/smaps_rollup") else {
        return 0;
    };
    let field = |name: &str| -> usize {
        rollup
            .lines()
            .find(|line| line.starts_with(name))
            .and_then(|line| line.split_whitespace().nth(1))
            .and_then(|kb| kb.parse::<usize>().ok())
            .map(|kb| kb * 1024)
            .unwrap_or(0)
    };
    field("Private_Clean:") + field("Private_Dirty:")
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn,avm_trace=info")),
        )
        .init();
    let opt = Opt::parse();

    let instance = create_wgpu_instance(
        wgpu::Backends::all(),
        wgpu::BackendOptions::default(),
        None,
    );
    let (adapter, device, queue) = futures::executor::block_on(request_adapter_and_device(
        wgpu::Backends::all(),
        &instance,
        None,
        wgpu::PowerPreference::HighPerformance,
    ))
    .map_err(|e| anyhow!(e.to_string()))?;
    eprintln!("adapter: {:?}", adapter.get_info());
    let descriptors = Arc::new(Descriptors::new(instance, adapter, device, queue));

    let movie = movie_from_path(&opt.swf, None).map_err(|e| anyhow!(e.to_string()))?;
    let frame_rate = movie.frame_rate().to_f64().max(1.0);
    let target = TextureTarget::new(&descriptors.device, (opt.width, opt.height))
        .map_err(|e| anyhow!(e.to_string()))?;
    let player: Arc<Mutex<Player>> = PlayerBuilder::new()
        .with_renderer(
            WgpuRenderBackend::new(descriptors, target).map_err(|e| anyhow!(e.to_string()))?,
        )
        .with_locale(DeterministicLocaleBackend::default())
        .with_movie(movie)
        .with_autoplay(true)
        .with_viewport_dimensions(opt.width, opt.height, 1.0)
        .build();

    let mut csv = opt
        .csv
        .as_ref()
        .map(|path| Ok::<_, anyhow::Error>(BufWriter::new(File::create(path)?)))
        .transpose()?;
    let mut header_written = false;

    let started = Instant::now();
    let deadline = started + Duration::from_secs(opt.seconds);
    let mut next_sample = started;
    let mut frame_times = FrameTimes::default();
    let mut peak_rss = 0usize;
    let mut peak_private = 0usize;
    let frame_interval = Duration::from_secs_f64(1.0 / frame_rate);
    let mut next_frame = started;

    while Instant::now() < deadline {
        let frame_started = Instant::now();
        {
            let mut player = player.lock().expect("player lock");
            player.preload(&mut ExecutionLimit::none());
            player.tick(ruffle_core::FloatDuration::from_millis(1000.0 / frame_rate));
            player.render();
        }
        frame_times.record(frame_started.elapsed().as_secs_f64() * 1000.0);

        let now = Instant::now();
        if now >= next_sample {
            next_sample = now + Duration::from_secs_f64(opt.interval);
            let elapsed = now.duration_since(started).as_secs_f64();
            let rss = resident_set_size();
            let private = private_bytes();
            peak_rss = peak_rss.max(rss);
            peak_private = peak_private.max(private);
            let report = player
                .lock()
                .expect("player lock")
                .mutate_with_update_context(MemoryReport::capture);
            let (frames, mean, p50, p95, p99, max, long, very_long) = frame_times.drain();

            if let Some(csv) = csv.as_mut() {
                if !header_written {
                    header_written = true;
                    writeln!(
                        csv,
                        "rss_bytes,peak_rss_bytes,private_bytes,peak_private_bytes,{},\
                         rust_heap_bytes,frames,frame_ms_mean,frame_ms_p50,frame_ms_p95,\
                         frame_ms_p99,frame_ms_max,frames_over_33ms,frames_over_100ms",
                        MemoryReport::csv_header_for(report.texture_kind_names())
                    )?;
                }
                writeln!(
                    csv,
                    "{rss},{peak_rss},{private},{peak_private},{},0,{frames},{mean:.2},{p50:.2},\
                     {p95:.2},{p99:.2},{max:.2},{long},{very_long}",
                    report.to_csv_row(elapsed)
                )?;
                csv.flush()?;
            }

            println!(
                "@{elapsed:6.0}s rss {:5} MiB private {:5} MiB | textures {:4} live {:6} MiB \
                 (peak {:6} MiB) | pool_main {:4} / {:6} MiB, {} sizes | created {:9} \
                 | frame mean {mean:6.1} p95 {p95:6.1} p99 {p99:6.1} max {max:6.1} \
                 ({long}/{frames} over 33 ms)",
                rss / (1024 * 1024),
                private / (1024 * 1024),
                report.tracked_textures,
                report.tracked_texture_bytes / (1024 * 1024),
                report.peak_texture_bytes / (1024 * 1024),
                report.texture_kind_live_counts.get(3).copied().unwrap_or_default(),
                report
                    .texture_kind_live_bytes
                    .get(3)
                    .copied()
                    .unwrap_or_default()
                    / (1024 * 1024),
                report.main_pool_size_classes,
                report.textures_created,
            );
        }

        if opt.realtime {
            next_frame += frame_interval;
            if let Some(wait) = next_frame.checked_duration_since(Instant::now()) {
                std::thread::sleep(wait);
            }
        }
    }
    Ok(())
}
