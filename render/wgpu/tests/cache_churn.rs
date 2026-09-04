//! What a room of animating avatars costs in cache and offscreen textures.
//!
//! The client's authenticated 43-minute session built 621,413 `cacheAsBitmap`
//! textures and 758,880 offscreen ones - about 1.4 million textures and 207 GiB
//! of allocation traffic - while never holding more than about 112 MB at once.
//! That is not memory in use; it is churn, and it is what a driver's allocator,
//! its fragmentation and its deferred frees actually see.
//!
//! This reproduces the shape of it deterministically. An AdventureQuest Worlds
//! avatar is a cached display object whose bounds breathe by a pixel or two
//! every frame as its animation plays, with equipment and effects on top, and
//! that breathing is the whole hypothesis: a cache keyed on the exact pixel size
//! of its picture rebuilds its texture whenever the picture moves, and a filter
//! chain sized on that texture rebuilds its scratch space with it.
//!
//! Run it with the numbers visible:
//!
//! ```text
//! cargo test --release -p ruffle_render_wgpu --test cache_churn -- --nocapture
//! ```

use ruffle_render::backend::{BitmapCacheEntry, RenderBackend, ViewportDimensions};
use ruffle_render::bitmap::{Bitmap, BitmapFormat, BitmapHandle, PixelRegion, PixelSnapping};
use ruffle_render::cache_capacity::{
    capacity_fits, capacity_for, set_capacity_reuse_enabled, set_granularity,
};
use ruffle_render::commands::{CommandHandler, CommandList};
use ruffle_render::filters::Filter;
use ruffle_render::transform::Transform;
use ruffle_render_wgpu::backend::{
    WgpuRenderBackend, create_wgpu_instance, request_adapter_and_device,
};
use ruffle_render_wgpu::buffer_pool::{PoolKind, PoolTelemetry, pool_telemetry};
use ruffle_render_wgpu::cache_pool::{CachePoolStats, cache_pool_counters};
use ruffle_render_wgpu::descriptors::Descriptors;
use ruffle_render_wgpu::target::TextureTarget;
use ruffle_render_wgpu::texture_churn;
use ruffle_render_wgpu::tuning;
use ruffle_render_wgpu::wgpu;
use std::num::NonZeroU32;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};
use swf::{
    BlurFilter, BlurFilterFlags, Color, DropShadowFilter, DropShadowFilterFlags, Fixed8, Fixed16,
    GlowFilter, GlowFilterFlags, Rectangle, Twips,
};

const VIEWPORT: (u32, u32) = (1920, 985);

/// The counters and the GPU are process-wide, so these run one at a time.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn exclusive() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn descriptors() -> Option<Arc<Descriptors>> {
    let instance =
        create_wgpu_instance(wgpu::Backends::all(), wgpu::BackendOptions::default(), None);
    let (adapter, device, queue) = futures::executor::block_on(request_adapter_and_device(
        wgpu::Backends::all(),
        &instance,
        None,
        Default::default(),
    ))
    .ok()?;
    Some(Arc::new(Descriptors::new(instance, adapter, device, queue)))
}

/// What the graphics allocator owns: what is in use, and what it is holding on
/// to in order to serve it.
///
/// The gap between the two is the point. A driver's allocator suballocates from
/// large blocks, and a workload that creates and destroys a million textures
/// grows that reserve to its high-water mark and fragments it. That memory is
/// resident, is charged to the process, and does not appear in any count of the
/// textures Ruffle is currently holding - which is the shape of the ~2.2 GB the
/// client's Windows run could not account for.
#[derive(Copy, Clone, Debug, Default)]
struct Allocator {
    allocated: u64,
    reserved: u64,
    blocks: usize,
}

fn allocator(descriptors: &Descriptors) -> Allocator {
    descriptors
        .device
        .generate_allocator_report()
        .map(|report| Allocator {
            allocated: report.total_allocated_bytes,
            reserved: report.total_reserved_bytes,
            blocks: report.blocks.len(),
        })
        .unwrap_or_default()
}

fn build_backend(descriptors: Arc<Descriptors>) -> WgpuRenderBackend<TextureTarget> {
    let target = TextureTarget::new(&descriptors.device, VIEWPORT).expect("texture target");
    let mut backend =
        WgpuRenderBackend::new(descriptors, target).expect("backend on the test adapter");
    backend.set_viewport_dimensions(ViewportDimensions {
        width: VIEWPORT.0,
        height: VIEWPORT.1,
        scale_factor: 1.0,
    });
    backend
}

fn test_bitmap(backend: &mut WgpuRenderBackend<TextureTarget>) -> BitmapHandle {
    let pixels = vec![255u8; 64 * 64 * 4];
    backend
        .register_bitmap(Bitmap::new(64, 64, BitmapFormat::Rgba, pixels))
        .expect("bitmap registration")
}

/// What a cached avatar wears. AQW objects are not uniform: a plain name plate
/// has no filter at all, a player has a glow, a spell effect has a blur, and
/// equipment adds a drop shadow.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Dress {
    Plain,
    Glow,
    Blur,
    Shadow,
}

impl Dress {
    fn of(index: usize) -> Self {
        match index % 4 {
            0 => Dress::Plain,
            1 => Dress::Glow,
            2 => Dress::Blur,
            _ => Dress::Shadow,
        }
    }

    fn filters(self) -> Vec<Filter> {
        match self {
            Dress::Plain => vec![],
            Dress::Glow => vec![Filter::GlowFilter(GlowFilter {
                color: Color::WHITE,
                blur_x: Fixed16::from_f32(4.0),
                blur_y: Fixed16::from_f32(4.0),
                strength: Fixed8::ONE,
                flags: GlowFilterFlags::from_passes(1),
            })],
            Dress::Blur => vec![Filter::BlurFilter(BlurFilter {
                blur_x: Fixed16::from_f32(6.0),
                blur_y: Fixed16::from_f32(6.0),
                flags: BlurFilterFlags::from_passes(1),
            })],
            Dress::Shadow => vec![Filter::DropShadowFilter(DropShadowFilter {
                color: Color::BLACK,
                blur_x: Fixed16::from_f32(4.0),
                blur_y: Fixed16::from_f32(4.0),
                angle: Fixed16::from_f32(45.0),
                distance: Fixed16::from_f32(4.0),
                strength: Fixed8::ONE,
                flags: DropShadowFilterFlags::from_passes(1),
            })],
        }
    }
}

/// How an actor's bounds move.
///
/// Real animation loops, so a periodic wobble is the honest common case - but a
/// loop means the sizes recur, and a pool keyed on the exact size gets to reuse
/// them. `Wandering` is the adversarial case: bounds that drift and almost never
/// repeat, which is what a long session with equipment changes, zoom and
/// combined effects actually produces, and which is the only way to see whether
/// the offscreen pool is meeting genuinely new sizes.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Motion {
    Breathing,
    Wandering,
}

/// One cached display object: an avatar, its equipment, and whatever effect it
/// is playing.
struct Actor {
    base: (u32, u32),
    dress: Dress,
    /// The texture it is holding, and how big that texture physically is.
    texture: Option<(BitmapHandle, (u32, u32))>,
    /// Counted here rather than globally so the benchmark can attribute them.
    builds: u64,
    build_pixels: u64,
}

impl Actor {
    fn new(index: usize) -> Self {
        // Avatars, name plates and equipment overlays are not all one size.
        let base = match index % 5 {
            0 => (147, 196),
            1 => (120, 150),
            2 => (172, 210),
            3 => (96, 96),
            _ => (160, 184),
        };
        Self {
            base,
            dress: Dress::of(index),
            texture: None,
            builds: 0,
            build_pixels: 0,
        }
    }

    /// The bounds the animation gives it this frame.
    ///
    /// A few pixels of breathing, out of phase between actors, exactly the
    /// 147x196 / 151x198 / 149x197 pattern the brief asks for. Equipment
    /// changes step the base size every so often, and a skill burst briefly
    /// makes the object much bigger, the way a spell effect does.
    fn logical_size(&self, index: usize, frame: usize, motion: Motion) -> (u32, u32) {
        let phase = frame + index * 7;
        let (wobble_w, wobble_h) = match motion {
            Motion::Breathing => (
                [0u32, 4, 2, 6, 0, 3][phase % 6],
                [0u32, 2, 1, 3, 2, 0][(phase + index) % 6],
            ),
            Motion::Wandering => {
                // A cheap deterministic hash, so the walk never repeats within a
                // run and is identical between runs.
                let mut h = (phase as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
                h ^= h >> 29;
                h = h.wrapping_mul(0xbf58_476d_1ce4_e5b9);
                h ^= h >> 32;
                ((h % 37) as u32, ((h >> 8) % 29) as u32)
            }
        };

        // An equipment change every 240 frames, which is a few seconds of play.
        let equipment = ((frame / 240 + index) % 3) as u32 * 8;
        // A skill burst: one actor in eight, for twelve frames out of every
        // hundred and eighty.
        let bursting = index % 8 == 0 && (frame % 180) < 12;
        let burst = if bursting { 90 } else { 0 };

        (
            self.base.0 + wobble_w + equipment + burst,
            self.base.1 + wobble_h + equipment + burst,
        )
    }

    /// What the cache texture has to hold: the picture plus whatever the filter
    /// chain grows it by, which is how `render_base` sizes it.
    fn cache_size(&self, logical: (u32, u32)) -> (u32, u32) {
        let mut rect = Rectangle {
            x_min: Twips::ZERO,
            x_max: Twips::from_pixels_i32(logical.0 as i32),
            y_min: Twips::ZERO,
            y_max: Twips::from_pixels_i32(logical.1 as i32),
        };
        for filter in self.dress.filters() {
            rect = filter.calculate_dest_rect(rect);
        }
        let width = (rect.x_max.to_pixels().ceil() - rect.x_min.to_pixels().floor()) as u32;
        let height = (rect.y_max.to_pixels().ceil() - rect.y_min.to_pixels().floor()) as u32;
        (width.max(1), height.max(1))
    }

    /// Keeps the texture it has if the policy says it still fits, and builds one
    /// otherwise. This is `BitmapCache::update`'s decision, made with the same
    /// shared policy so the benchmark measures the real thing.
    fn texture_for(
        &mut self,
        backend: &mut WgpuRenderBackend<TextureTarget>,
        wanted: (u32, u32),
    ) -> BitmapHandle {
        if let Some((handle, physical)) = &self.texture
            && capacity_fits(*physical, wanted)
        {
            return handle.clone();
        }
        let physical = capacity_for(wanted.0, wanted.1);
        let handle = backend
            .create_empty_texture(
                NonZeroU32::new(physical.0).expect("non-zero width"),
                NonZeroU32::new(physical.1).expect("non-zero height"),
            )
            .expect("cache texture");
        self.builds += 1;
        self.build_pixels += u64::from(physical.0) * u64::from(physical.1);
        self.texture = Some((handle.clone(), physical));
        handle
    }
}

/// What one run of the room cost.
struct Churn {
    frames: usize,
    actors: usize,
    cache_builds: u64,
    cache_build_pixels: u64,
    textures_created: u64,
    texture_bytes_created: u64,
    live_textures: usize,
    live_texture_bytes: usize,
    offscreen: PoolTelemetry,
    cache_pool: CachePoolStats,
    allocator_before: Allocator,
    allocator_after: Allocator,
    cpu_times: Vec<Duration>,
    frame_times: Vec<Duration>,
}

impl Churn {
    fn mean_ms(times: &[Duration]) -> f64 {
        times.iter().map(|d| d.as_secs_f64()).sum::<f64>() * 1000.0 / times.len() as f64
    }

    fn percentile_ms(times: &[Duration], p: f64) -> f64 {
        let mut sorted: Vec<f64> = times.iter().map(|d| d.as_secs_f64() * 1000.0).collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).expect("frame times are finite"));
        sorted[((sorted.len() as f64 - 1.0) * p).round() as usize]
    }

    fn per_second(&self, count: u64) -> f64 {
        // At the client's frame budget, so the rates are comparable with the
        // session's own numbers rather than with this laptop's frame rate.
        count as f64 * 24.0 / self.frames as f64
    }

    fn report(&self, label: &str) {
        println!("\n{label}: {} actors, {} frames", self.actors, self.frames);
        println!(
            "  cache textures built   {:>10}  ({:.1}/s at 24fps, {:.2} per actor-frame)",
            self.cache_builds,
            self.per_second(self.cache_builds),
            self.cache_builds as f64 / (self.frames * self.actors) as f64
        );
        println!(
            "  cache bytes built      {:>10.1} MB ({:.1} MB/s at 24fps)",
            self.cache_build_pixels as f64 * 4.0 / 1_048_576.0,
            self.per_second(self.cache_build_pixels * 4) / 1_048_576.0
        );
        println!(
            "  all textures created   {:>10}  ({:.1}/s), {:.1} MB total",
            self.textures_created,
            self.per_second(self.textures_created),
            self.texture_bytes_created as f64 / 1_048_576.0
        );
        println!(
            "  live at the end        {:>10} textures, {:.1} MB",
            self.live_textures,
            self.live_texture_bytes as f64 / 1_048_576.0
        );
        let misses = self.offscreen.misses_total();
        println!(
            "  offscreen pool         {:>10} hits, {} misses ({:.1}% hit rate)",
            self.offscreen.hits,
            misses,
            100.0 * self.offscreen.hits as f64 / (self.offscreen.hits + misses).max(1) as f64
        );
        for (name, (count, bytes)) in ruffle_render_wgpu::buffer_pool::POOL_MISS_NAMES.iter().zip(
            self.offscreen
                .misses
                .iter()
                .zip(self.offscreen.miss_bytes.iter()),
        ) {
            if *count > 0 {
                println!(
                    "      {name:<22} {count:>8}  {:.1} MB",
                    *bytes as f64 / 1_048_576.0
                );
            }
        }
        println!(
            "      size classes seen  {:>8}, live {}",
            self.offscreen.size_classes_seen, self.offscreen.live_size_classes
        );
        println!(
            "      evicted by budget  {:>8}  {:.1} MB",
            self.offscreen.evictions,
            self.offscreen.evicted_bytes as f64 / 1_048_576.0
        );
        println!(
            "  cache texture pool     {:>10} takes, {} recycled, {} allocated ({:.1}% recycled)",
            self.cache_pool.takes,
            self.cache_pool.hits,
            self.cache_pool.builds,
            100.0 * self.cache_pool.hit_rate(),
        );
        println!(
            "  gpu allocator          allocated {:.1} -> {:.1} MB, reserved {:.1} -> {:.1} MB, \
             blocks {} -> {} (gap at the end {:.1} MB)",
            self.allocator_before.allocated as f64 / 1_048_576.0,
            self.allocator_after.allocated as f64 / 1_048_576.0,
            self.allocator_before.reserved as f64 / 1_048_576.0,
            self.allocator_after.reserved as f64 / 1_048_576.0,
            self.allocator_before.blocks,
            self.allocator_after.blocks,
            self.allocator_after
                .reserved
                .saturating_sub(self.allocator_after.allocated) as f64
                / 1_048_576.0,
        );
        println!(
            "  frame ms   mean {:.1}  p95 {:.1}  p99 {:.1}   (cpu encode mean {:.1})",
            Self::mean_ms(&self.frame_times),
            Self::percentile_ms(&self.frame_times, 0.95),
            Self::percentile_ms(&self.frame_times, 0.99),
            Self::mean_ms(&self.cpu_times),
        );
    }
}

/// Plays the room for `frames` frames and reports what it cost.
fn run_room(
    backend: &mut WgpuRenderBackend<TextureTarget>,
    descriptors: &Descriptors,
    bitmap: &BitmapHandle,
    actors: usize,
    frames: usize,
    warmup: usize,
    motion: Motion,
) -> Churn {
    let mut cast: Vec<Actor> = (0..actors).map(Actor::new).collect();

    let mut cache_builds = 0;
    let mut cache_build_pixels = 0;
    let mut cpu_times = Vec::with_capacity(frames);
    let mut frame_times = Vec::with_capacity(frames);
    let mut created_before = 0;
    let mut bytes_before = 0;
    let mut offscreen_before = PoolTelemetry::default();
    let mut cache_pool_before = CachePoolStats::default();
    let mut allocator_before = Allocator::default();

    for frame in 0..(warmup + frames) {
        if frame == warmup {
            // Everything before this is the scene settling; the measurement is
            // of a room that is already running.
            for actor in &mut cast {
                actor.builds = 0;
                actor.build_pixels = 0;
            }
            let (created, _, bytes) = texture_churn();
            created_before = created;
            bytes_before = bytes;
            offscreen_before = pool_telemetry(PoolKind::Offscreen);
            cache_pool_before = cache_pool_counters();
            allocator_before = allocator(descriptors);
        }

        let mut entries = Vec::with_capacity(actors);
        for (index, actor) in cast.iter_mut().enumerate() {
            let logical = actor.logical_size(index, frame, motion);
            let wanted = actor.cache_size(logical);
            let handle = actor.texture_for(backend, wanted);
            let mut commands = CommandList::new();
            commands.render_bitmap(
                bitmap.clone(),
                Transform::default(),
                false,
                PixelSnapping::Never,
                PixelRegion::for_whole_size(logical.0.min(64), logical.1.min(64)),
            );
            entries.push(BitmapCacheEntry {
                handle,
                commands,
                clear: Color::from_rgba(0),
                filters: actor.dress.filters(),
                // The picture, which is what the cache is asked to hold; the
                // texture behind it may be rounded up.
                logical_width: wanted.0,
                logical_height: wanted.1,
            });
        }

        let start = Instant::now();
        backend.submit_frame(Color::BLACK, CommandList::new(), entries);
        let cpu = start.elapsed();
        backend.capture_frame().expect("capture must succeed");
        if frame >= warmup {
            cpu_times.push(cpu);
            frame_times.push(start.elapsed());
        }
    }

    for actor in &cast {
        cache_builds += actor.builds;
        cache_build_pixels += actor.build_pixels;
    }
    let (created, _, bytes) = texture_churn();
    let (live_textures, live_texture_bytes) = ruffle_render_wgpu::tracked_texture_totals();
    let offscreen_after = pool_telemetry(PoolKind::Offscreen);
    let cache_pool_after = cache_pool_counters();

    Churn {
        frames,
        actors,
        cache_builds,
        cache_build_pixels,
        textures_created: created - created_before,
        texture_bytes_created: bytes - bytes_before,
        live_textures,
        live_texture_bytes,
        offscreen: PoolTelemetry {
            hits: offscreen_after.hits - offscreen_before.hits,
            misses: offscreen_after
                .misses
                .iter()
                .zip(offscreen_before.misses.iter().chain(std::iter::repeat(&0)))
                .map(|(a, b)| a - b)
                .collect(),
            miss_bytes: offscreen_after
                .miss_bytes
                .iter()
                .zip(
                    offscreen_before
                        .miss_bytes
                        .iter()
                        .chain(std::iter::repeat(&0)),
                )
                .map(|(a, b)| a - b)
                .collect(),
            evictions: offscreen_after.evictions - offscreen_before.evictions,
            evicted_bytes: offscreen_after.evicted_bytes - offscreen_before.evicted_bytes,
            ..offscreen_after
        },
        cache_pool: CachePoolStats {
            takes: cache_pool_after.takes - cache_pool_before.takes,
            hits: cache_pool_after.hits - cache_pool_before.hits,
            builds: cache_pool_after.builds - cache_pool_before.builds,
            returns: cache_pool_after.returns - cache_pool_before.returns,
            evictions: cache_pool_after.evictions - cache_pool_before.evictions,
            evicted_bytes: cache_pool_after.evicted_bytes - cache_pool_before.evicted_bytes,
            ..cache_pool_after
        },
        allocator_before,
        allocator_after: allocator(descriptors),
        cpu_times,
        frame_times,
    }
}

fn crowds() -> Vec<usize> {
    match std::env::var("RUFFLE_CHURN_CROWDS") {
        Ok(list) => list
            .split(',')
            .map(|n| n.trim().parse().expect("crowd sizes are numbers"))
            .collect(),
        Err(_) => vec![25, 50, 100, 250],
    }
}

fn frames() -> usize {
    std::env::var("RUFFLE_CHURN_FRAMES")
        .ok()
        .and_then(|f| f.parse().ok())
        .unwrap_or(120)
}

/// The measurement the brief asks for: what a room of animating cached objects
/// allocates, with the capacity policy off and on, a few seconds apart on the
/// same machine in the same process.
#[test]
fn a_room_of_animating_avatars_reports_its_churn() {
    let _exclusive = exclusive();
    let Some(descriptors) = descriptors() else {
        eprintln!("no GPU adapter available; skipping");
        return;
    };
    let mut backend = build_backend(descriptors.clone());
    let bitmap = test_bitmap(&mut backend);
    let frames = frames();

    println!(
        "\n=== cache and offscreen churn, {frames} frames per run ===\
         \n(the client's session: 621,413 cache textures and 758,880 offscreen \
         textures in 43 minutes)"
    );

    for motion in [Motion::Breathing, Motion::Wandering] {
        println!("\n############ {motion:?} bounds ############");
        for actors in crowds() {
            // Three arms, run within a few seconds of each other on the same
            // machine in the same process, because this laptop's frame times drift
            // with its temperature over a run.
            set_capacity_reuse_enabled(false);
            tuning::set_cache_pool_enabled(false);
            let base = run_room(
                &mut backend,
                &descriptors,
                &bitmap,
                actors,
                frames,
                12,
                motion,
            );
            base.report(&format!("A: phase 1 behaviour, {actors} actors"));

            set_capacity_reuse_enabled(false);
            tuning::set_cache_pool_enabled(true);
            let pooled = run_room(
                &mut backend,
                &descriptors,
                &bitmap,
                actors,
                frames,
                12,
                motion,
            );
            pooled.report(&format!("B: recycled, exact size, {actors} actors"));

            set_capacity_reuse_enabled(true);
            set_granularity(ruffle_render::cache_capacity::DEFAULT_GRANULARITY);
            tuning::set_cache_pool_enabled(true);
            let both = run_room(
                &mut backend,
                &descriptors,
                &bitmap,
                actors,
                frames,
                12,
                motion,
            );
            both.report(&format!("C: recycled, capacity rounded, {actors} actors"));

            println!(
                "\n  >>> {actors} actors, textures really allocated per run: \
             A {} ({:.1} MB)  ->  B {} ({:.1} MB, {:.1}x)  ->  C {} ({:.1} MB, {:.1}x)",
                base.textures_created,
                base.texture_bytes_created as f64 / 1_048_576.0,
                pooled.textures_created,
                pooled.texture_bytes_created as f64 / 1_048_576.0,
                base.textures_created as f64 / pooled.textures_created.max(1) as f64,
                both.textures_created,
                both.texture_bytes_created as f64 / 1_048_576.0,
                base.textures_created as f64 / both.textures_created.max(1) as f64,
            );
            println!(
                "      live texture memory at the end: A {:.1} MB, B {:.1} MB, C {:.1} MB",
                base.live_texture_bytes as f64 / 1_048_576.0,
                pooled.live_texture_bytes as f64 / 1_048_576.0,
                both.live_texture_bytes as f64 / 1_048_576.0,
            );
        }
    }
    set_capacity_reuse_enabled(true);
    tuning::set_cache_pool_enabled(true);
}

/// Sweeps the rounding, because the brief asks for the bucket size to be
/// measured rather than assumed.
#[test]
#[ignore = "policy sweep; run explicitly"]
fn the_granularity_is_chosen_by_measurement() {
    let _exclusive = exclusive();
    let Some(descriptors) = descriptors() else {
        eprintln!("no GPU adapter available; skipping");
        return;
    };
    let mut backend = build_backend(descriptors.clone());
    let bitmap = test_bitmap(&mut backend);
    let frames = frames();

    println!("\n=== rounding sweep, 100 actors, {frames} frames ===");
    set_capacity_reuse_enabled(false);
    let exact = run_room(
        &mut backend,
        &descriptors,
        &bitmap,
        100,
        frames,
        12,
        Motion::Wandering,
    );
    exact.report("exact size");

    for granularity in [8u32, 16, 32, 64, 128] {
        set_capacity_reuse_enabled(true);
        set_granularity(granularity);
        let run = run_room(
            &mut backend,
            &descriptors,
            &bitmap,
            100,
            frames,
            12,
            Motion::Wandering,
        );
        run.report(&format!("granularity {granularity}"));
    }
    set_granularity(ruffle_render::cache_capacity::DEFAULT_GRANULARITY);
}
