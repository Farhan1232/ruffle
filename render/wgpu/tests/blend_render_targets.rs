//! What a crowded room costs the renderer.
//!
//! An AdventureQuest Worlds room is a few hundred avatars, each a display
//! object with a non-normal blend mode, and each of those is rendered through a
//! temporary render target before being composited back. This measures what
//! those targets ask for - the pixels the pool hands out for one frame, the
//! targets it has to build, and how long the frame takes - for scenes of the
//! sizes the client actually plays in.
//!
//! Run it with the numbers visible:
//!
//! ```text
//! cargo test --release -p ruffle_render_wgpu --test blend_render_targets -- --nocapture
//! ```

use ruffle_render::backend::{BitmapCacheEntry, RenderBackend, ViewportDimensions};
use ruffle_render::bitmap::{Bitmap, BitmapFormat, BitmapHandle, PixelSnapping};
use ruffle_render::commands::{CommandHandler, CommandList, RenderBlendMode};
use ruffle_render::filters::Filter;
use ruffle_render::matrix::Matrix;
use ruffle_render::transform::Transform;
use ruffle_render_wgpu::backend::{
    WgpuRenderBackend, create_wgpu_instance, request_adapter_and_device,
};
use ruffle_render_wgpu::buffer_pool::{PoolUsage, pool_usage};
use ruffle_render_wgpu::descriptors::Descriptors;
use ruffle_render_wgpu::target::TextureTarget;
use ruffle_render_wgpu::wgpu;
use ruffle_render_wgpu::{RenderStats, render_stats};
use std::num::NonZeroU32;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};
use swf::{BlendMode, Color, Fixed8, Fixed16, GlowFilter, GlowFilterFlags, Twips};

/// The client's window.
const VIEWPORT: (u32, u32) = (1920, 985);

/// An avatar with its equipment: bigger than a particle, far smaller than the
/// screen.
const OBJECT: (u32, u32) = (150, 200);

/// Room sizes to measure, from a quiet map to the worst the client has
/// reported.
///
/// Overridable, because the full-viewport renderer this is measured against
/// needs about 7 MiB of render target per object and cannot reach the top of
/// the list on a machine with 8 GiB of memory.
fn crowds() -> Vec<usize> {
    match std::env::var("RUFFLE_BENCH_CROWDS") {
        Ok(list) => list
            .split(',')
            .map(|n| n.trim().parse().expect("crowd sizes are numbers"))
            .collect(),
        Err(_) => vec![50, 100, 250, 500, 800],
    }
}

const WARMUP_FRAMES: usize = 3;
const MEASURED_FRAMES: usize = 10;

/// The counters these tests read are process-wide, and so is the GPU they
/// share, so they have to run one at a time whatever `--test-threads` says.
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

/// Scatters `count` blended objects over the viewport, the way a room full of
/// players is scattered over a map.
///
/// Deterministic: the same count always lays out the same scene, so two builds
/// can be compared.
fn crowd(bitmap: &BitmapHandle, count: usize, blend_mode: BlendMode) -> CommandList {
    let mut commands = CommandList::new();
    for i in 0..count {
        // A fixed low-discrepancy scatter, so objects overlap the way avatars
        // do without any two runs differing.
        let x = ((i as f64 * 0.7548776662) % 1.0) * (VIEWPORT.0 - OBJECT.0) as f64;
        let y = ((i as f64 * 0.5698402909) % 1.0) * (VIEWPORT.1 - OBJECT.1) as f64;

        let mut group = CommandList::new();
        group.render_bitmap(
            bitmap.clone(),
            Transform {
                matrix: Matrix::translate(Twips::from_pixels(x), Twips::from_pixels(y)),
                color_transform: Default::default(),
                perspective_projection: None,
            },
            false,
            PixelSnapping::Never,
            ruffle_render::bitmap::PixelRegion::for_whole_size(OBJECT.0, OBJECT.1),
        );
        commands.blend(group, RenderBlendMode::Builtin(blend_mode));
    }
    commands
}

struct Measurement {
    usage: PoolUsage,
    work: RenderStats,
    frames: usize,
    /// Time to walk the commands and encode the frame, without waiting for the
    /// GPU. This is what a stuttering frame costs the main thread.
    cpu_times: Vec<Duration>,
    /// Time until the frame is actually on screen, GPU included.
    frame_times: Vec<Duration>,
}

impl Measurement {
    fn per_frame_pixels(&self) -> u64 {
        self.usage.pixels / self.frame_times.len() as u64
    }

    fn per_frame_builds(&self) -> f64 {
        self.usage.builds as f64 / self.frame_times.len() as f64
    }

    fn reuse_rate(&self) -> f64 {
        if self.usage.takes == 0 {
            return 1.0;
        }
        1.0 - self.usage.builds as f64 / self.usage.takes as f64
    }

    fn mean_ms(times: &[Duration]) -> f64 {
        times.iter().map(|d| d.as_secs_f64()).sum::<f64>() * 1000.0 / times.len() as f64
    }

    fn percentile_ms(times: &[Duration], p: f64) -> f64 {
        let mut sorted: Vec<f64> = times.iter().map(|d| d.as_secs_f64() * 1000.0).collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).expect("frame times are finite"));
        let index = ((sorted.len() as f64 - 1.0) * p).round() as usize;
        sorted[index]
    }
}

/// Draws the scene a few times and reports what it cost.
fn measure(
    backend: &mut WgpuRenderBackend<TextureTarget>,
    bitmap: &BitmapHandle,
    count: usize,
    blend_mode: BlendMode,
) -> Measurement {
    for _ in 0..WARMUP_FRAMES {
        backend.submit_frame(Color::BLACK, crowd(bitmap, count, blend_mode), vec![]);
        backend.capture_frame().expect("capture must succeed");
    }

    let before = pool_usage();
    let work_before = render_stats();
    let mut cpu_times = Vec::with_capacity(MEASURED_FRAMES);
    let mut frame_times = Vec::with_capacity(MEASURED_FRAMES);
    for _ in 0..MEASURED_FRAMES {
        let commands = crowd(bitmap, count, blend_mode);
        let start = Instant::now();
        backend.submit_frame(Color::BLACK, commands, vec![]);
        cpu_times.push(start.elapsed());
        // The capture waits for the GPU, so the frame is really finished.
        backend.capture_frame().expect("capture must succeed");
        frame_times.push(start.elapsed());
    }

    let after = render_stats();
    Measurement {
        usage: pool_usage() - before,
        work: RenderStats {
            render_passes: after.render_passes - work_before.render_passes,
            bind_groups_created: after.bind_groups_created - work_before.bind_groups_created,
            bind_group_cache_hits: after.bind_group_cache_hits - work_before.bind_group_cache_hits,
            bind_group_cache_misses: after.bind_group_cache_misses
                - work_before.bind_group_cache_misses,
            fastpath_eligible: after.fastpath_eligible - work_before.fastpath_eligible,
            fastpath_used: after.fastpath_used - work_before.fastpath_used,
            batch_eligible: after.batch_eligible - work_before.batch_eligible,
            batch_used: after.batch_used - work_before.batch_used,
            destination_copies: after.destination_copies - work_before.destination_copies,
            destination_copy_pixels: after.destination_copy_pixels
                - work_before.destination_copy_pixels,
            complex_blends: after.complex_blends - work_before.complex_blends,
            complex_blend_passes: after.complex_blend_passes - work_before.complex_blend_passes,
            fallbacks: after
                .fallbacks
                .iter()
                .zip(&work_before.fallbacks)
                .map(|(a, b)| a - b)
                .collect(),
            ..after
        },
        frames: MEASURED_FRAMES,
        cpu_times,
        frame_times,
    }
}

fn build_backend(descriptors: Arc<Descriptors>) -> WgpuRenderBackend<TextureTarget> {
    let target = TextureTarget::new(&descriptors.device, VIEWPORT).expect("texture target");
    let mut backend = WgpuRenderBackend::new(descriptors, target).expect("render backend");
    backend.set_viewport_dimensions(ViewportDimensions {
        width: VIEWPORT.0,
        height: VIEWPORT.1,
        scale_factor: 1.0,
    });
    backend
}

fn test_bitmap(backend: &mut WgpuRenderBackend<TextureTarget>) -> BitmapHandle {
    let pixels = vec![200u8; (OBJECT.0 * OBJECT.1 * 4) as usize];
    backend
        .register_bitmap(Bitmap::new(OBJECT.0, OBJECT.1, BitmapFormat::Rgba, pixels))
        .expect("bitmap registration")
}

#[test]
fn a_crowded_room_does_not_ask_for_screen_sized_targets() {
    let _exclusive = exclusive();
    let Some(descriptors) = descriptors() else {
        eprintln!("no GPU adapter available; skipping");
        return;
    };
    let mut backend = build_backend(descriptors);
    let bitmap = test_bitmap(&mut backend);

    let screen_pixels = u64::from(VIEWPORT.0) * u64::from(VIEWPORT.1);

    for blend_mode in [BlendMode::Layer, BlendMode::Multiply] {
        println!(
            "\n{blend_mode:?} blends, {}x{} objects in a {}x{} viewport",
            OBJECT.0, OBJECT.1, VIEWPORT.0, VIEWPORT.1
        );
        println!(
            "{:>7} {:>10} {:>9} {:>8} {:>6} {:>7} {:>7} {:>7} {:>8} {:>7} {:>7} {:>7} {:>7}",
            "objects",
            "target MB",
            "passes/fr",
            "targets",
            "pages",
            "batch%",
            "copies",
            "copy MB",
            "blend/pass",
            "fast%",
            "cpu ms",
            "mean ms",
            "p99 ms"
        );
        for count in crowds() {
            let m = measure(&mut backend, &bitmap, count, blend_mode);
            let frames = m.frames as f64;
            println!(
                "{:>7} {:>10.1} {:>9.1} {:>8} {:>6} {:>6.0}% {:>7.1} {:>7.2} {:>8.1} {:>6.0}% {:>7.1} {:>7.1} {:>7.1}",
                count,
                m.per_frame_pixels() as f64 * 4.0 / (1024.0 * 1024.0),
                m.work.render_passes as f64 / frames,
                m.work.blend_targets_live,
                m.work.pages_last_frame,
                if m.work.batch_eligible > 0 {
                    100.0 * m.work.batch_used as f64 / m.work.batch_eligible as f64
                } else {
                    0.0
                },
                m.work.destination_copies as f64 / frames,
                m.work.destination_copy_pixels as f64 * 4.0 / (1024.0 * 1024.0) / frames,
                if m.work.complex_blend_passes > 0 {
                    m.work.complex_blends as f64 / m.work.complex_blend_passes as f64
                } else {
                    0.0
                },
                if m.work.fastpath_eligible > 0 {
                    100.0 * m.work.fastpath_used as f64 / m.work.fastpath_eligible as f64
                } else {
                    0.0
                },
                Measurement::mean_ms(&m.cpu_times),
                Measurement::mean_ms(&m.frame_times),
                Measurement::percentile_ms(&m.frame_times, 0.99),
            );

            // Bind groups are kept with the pooled texture they name, so a
            // steady scene must stop building them. If this creeps up, the
            // cache is missing - or worse, growing.
            assert!(
                m.work.bind_groups_created <= 2,
                "{count} {blend_mode:?} blends built {} bind groups over {} frames; \
                 the cache kept with the pooled targets is not holding",
                m.work.bind_groups_created,
                m.frames
            );

            // The point of the exercise: an object a fraction of the screen's
            // size must not be given the screen. Allow generous slack for the
            // frame's own target and the size classes' rounding, and still
            // catch any return to a target per object.
            let per_object = m.per_frame_pixels().saturating_sub(screen_pixels) / count as u64;
            assert!(
                per_object < screen_pixels / 8,
                "{count} {blend_mode:?} blends asked for {per_object} target pixels each, \
                 against a {screen_pixels}-pixel screen"
            );
        }
    }
}

/// A scene that keeps drawing the same thing should settle: the pool ends up
/// holding what the frame needs and stops building targets. Growth here would
/// mean the size classes had failed and every object was getting its own pool
/// key.
#[test]
fn a_steady_scene_stops_building_targets() {
    let _exclusive = exclusive();
    let Some(descriptors) = descriptors() else {
        eprintln!("no GPU adapter available; skipping");
        return;
    };
    let mut backend = build_backend(descriptors);
    let bitmap = test_bitmap(&mut backend);

    let m = measure(&mut backend, &bitmap, 250, BlendMode::Multiply);
    assert!(
        m.reuse_rate() > 0.99,
        "only {:.1}% of targets were re-used in a scene that never changes",
        m.reuse_rate() * 100.0
    );
}

/// A filtered `cacheAsBitmap` object, which is how AdventureQuest Worlds draws
/// a glowing name plate or a spell effect. Ruffle redraws one of these into its
/// cache texture whenever it changes, applying the filter through temporary
/// targets of its own.
fn cache_entries(
    bitmap: &BitmapHandle,
    caches: &[(BitmapHandle, u32, u32)],
) -> Vec<BitmapCacheEntry> {
    caches
        .iter()
        .map(|(handle, width, height)| BitmapCacheEntry {
            handle: handle.clone(),
            commands: {
                let mut commands = CommandList::new();
                commands.render_bitmap(
                    bitmap.clone(),
                    Transform::default(),
                    false,
                    PixelSnapping::Never,
                    ruffle_render::bitmap::PixelRegion::for_whole_size(*width, *height),
                );
                commands
            },
            clear: Color::from_rgba(0),
            filters: vec![Filter::GlowFilter(GlowFilter {
                color: Color::WHITE,
                blur_x: Fixed16::from_f32(4.0),
                blur_y: Fixed16::from_f32(4.0),
                strength: Fixed8::ONE,
                flags: GlowFilterFlags::from_passes(1),
            })],
        })
        .collect()
}

/// Cached, filtered objects are redrawn every frame, so the targets their
/// filters run through are wanted again a frame later. Building them fresh each
/// time is pure churn: a client session measured 1.86 million offscreen targets
/// created and the same number destroyed, for a pool never holding more than a
/// few megabytes.
#[test]
fn filtered_cached_objects_re_use_their_filter_targets() {
    let _exclusive = exclusive();
    let Some(descriptors) = descriptors() else {
        eprintln!("no GPU adapter available; skipping");
        return;
    };
    let mut backend = build_backend(descriptors);
    let bitmap = test_bitmap(&mut backend);

    // Cached objects are all different sizes, the way avatars, name plates and
    // spell effects are. That is what made the per-frame pool expensive: a
    // fresh pool has to build a target for every size it meets, every frame.
    let caches: Vec<(BitmapHandle, u32, u32)> = (0..64u32)
        .map(|i| {
            let width = 48 + (i % 16) * 13;
            let height = 40 + (i % 11) * 17;
            let handle = backend
                .create_empty_texture(
                    NonZeroU32::new(width).expect("non-zero"),
                    NonZeroU32::new(height).expect("non-zero"),
                )
                .expect("cache texture");
            (handle, width, height)
        })
        .collect();

    for _ in 0..WARMUP_FRAMES {
        backend.submit_frame(
            Color::BLACK,
            CommandList::new(),
            cache_entries(&bitmap, &caches),
        );
        backend.capture_frame().expect("capture must succeed");
    }

    let before = pool_usage();
    for _ in 0..MEASURED_FRAMES {
        backend.submit_frame(
            Color::BLACK,
            CommandList::new(),
            cache_entries(&bitmap, &caches),
        );
        backend.capture_frame().expect("capture must succeed");
    }
    let usage = pool_usage() - before;

    println!(
        "\n64 filtered cached objects: {} targets taken over {MEASURED_FRAMES} frames, \
         {} built ({:.1} per frame)",
        usage.takes,
        usage.builds,
        usage.builds as f64 / MEASURED_FRAMES as f64
    );

    assert!(
        usage.builds * 4 < usage.takes,
        "{} of {} filter targets had to be built - they are not surviving between frames",
        usage.builds,
        usage.takes
    );
}

/// A room shaped the way AdventureQuest Worlds' is, rather than the way the
/// fast path likes.
///
/// The benchmark above blends a single bitmap, which is exactly the shape the
/// direct path accepts, so it measures the best case. A real room is a mixture:
/// some blended objects are cached or filtered and so reach the renderer as one
/// `render_bitmap`, and some are containers whose children are drawn
/// individually and must still be composited through a target. This measures
/// what the mixture costs and, more to the point, what fraction of it the
/// direct path can take - because if that fraction is small, the remaining
/// per-target price is what has to be attacked next.
fn mixed_room(
    bitmap: &BitmapHandle,
    count: usize,
    cached_share: f64,
    complex_share: f64,
) -> CommandList {
    let mut commands = CommandList::new();
    for i in 0..count {
        let x = ((i as f64 * 0.7548776662) % 1.0) * (VIEWPORT.0 - OBJECT.0) as f64;
        let y = ((i as f64 * 0.5698402909) % 1.0) * (VIEWPORT.1 - OBJECT.1) as f64;
        let place = |dx: f64, dy: f64| Transform {
            matrix: Matrix::translate(Twips::from_pixels(x + dx), Twips::from_pixels(y + dy)),
            color_transform: Default::default(),
            perspective_projection: None,
        };
        let draw = |group: &mut CommandList, dx: f64, dy: f64| {
            group.render_bitmap(
                bitmap.clone(),
                place(dx, dy),
                false,
                PixelSnapping::Never,
                ruffle_render::bitmap::PixelRegion::for_whole_size(OBJECT.0, OBJECT.1),
            );
        };

        let fraction = (i % 100) as f64 / 100.0;
        let mut group = CommandList::new();
        if fraction < cached_share {
            // Cached or filtered: one bitmap, which the direct path can take.
            draw(&mut group, 0.0, 0.0);
        } else {
            // A container: body, equipment and a name plate, drawn separately.
            draw(&mut group, 0.0, 0.0);
            draw(&mut group, 8.0, 40.0);
            draw(&mut group, -6.0, 90.0);
        }
        let blend = if fraction < complex_share {
            BlendMode::Multiply
        } else {
            BlendMode::Layer
        };
        commands.blend(group, RenderBlendMode::Builtin(blend));
    }
    commands
}

#[test]
fn how_much_of_an_aqw_shaped_room_takes_the_direct_path() {
    let _exclusive = exclusive();
    let Some(descriptors) = descriptors() else {
        eprintln!("no GPU adapter available; skipping");
        return;
    };
    let mut backend = build_backend(descriptors);
    let bitmap = test_bitmap(&mut backend);

    // The client's worst windows hold 800-920 targets at once.
    println!("\n800 blended objects, mixtures of cached singles and multi-child containers");
    println!(
        "{:>8} {:>8} {:>10} {:>9} {:>8} {:>6} {:>7} {:>7} {:>8} {:>7} {:>8} {:>8}",
        "cached%",
        "complex%",
        "target MB",
        "passes/fr",
        "targets",
        "pages",
        "batch%",
        "copies",
        "blend/pass",
        "fast%",
        "mean ms",
        "p95 ms"
    );
    for (cached, complex) in [(1.0, 0.0), (0.6, 0.1), (0.3, 0.2), (0.0, 0.2), (0.0, 1.0)] {
        for _ in 0..WARMUP_FRAMES {
            backend.submit_frame(
                Color::BLACK,
                mixed_room(&bitmap, 800, cached, complex),
                vec![],
            );
            backend.capture_frame().expect("capture must succeed");
        }
        let before = pool_usage();
        let work_before = render_stats();
        let mut times = Vec::new();
        for _ in 0..MEASURED_FRAMES {
            let commands = mixed_room(&bitmap, 800, cached, complex);
            let start = Instant::now();
            backend.submit_frame(Color::BLACK, commands, vec![]);
            backend.capture_frame().expect("capture must succeed");
            times.push(start.elapsed());
        }
        let usage = pool_usage() - before;
        let after = render_stats();
        let frames = MEASURED_FRAMES as f64;
        let eligible = after.fastpath_eligible - work_before.fastpath_eligible;
        let used = after.fastpath_used - work_before.fastpath_used;
        let batch_eligible = after.batch_eligible - work_before.batch_eligible;
        let batch_used = after.batch_used - work_before.batch_used;
        let complex_blends = after.complex_blends - work_before.complex_blends;
        let complex_passes = after.complex_blend_passes - work_before.complex_blend_passes;
        println!(
            "{:>7.0}% {:>7.0}% {:>10.1} {:>9.1} {:>8} {:>6} {:>6.0}% {:>7.1} {:>8.1} {:>6.0}% {:>8.1} {:>8.1}",
            cached * 100.0,
            complex * 100.0,
            usage.pixels as f64 * 4.0 / (1024.0 * 1024.0) / frames,
            (after.render_passes - work_before.render_passes) as f64 / frames,
            after.blend_targets_live,
            after.pages_last_frame,
            if batch_eligible > 0 {
                100.0 * batch_used as f64 / batch_eligible as f64
            } else {
                0.0
            },
            (after.destination_copies - work_before.destination_copies) as f64 / frames,
            if complex_passes > 0 {
                complex_blends as f64 / complex_passes as f64
            } else {
                0.0
            },
            if eligible > 0 {
                100.0 * used as f64 / eligible as f64
            } else {
                0.0
            },
            Measurement::mean_ms(&times),
            Measurement::percentile_ms(&times, 0.95),
        );
    }
    println!("\nwhy the rest could not take the direct path:");
    let stats = render_stats();
    for (name, count) in ruffle_render_wgpu::render_stats::FALLBACK_NAMES
        .iter()
        .zip(&stats.fallbacks)
    {
        if *count > 0 {
            println!("  {name:24} {count:>10}");
        }
    }
    println!("why the rest could not share a page:");
    for (name, count) in ruffle_render_wgpu::render_stats::PAGE_FALLBACK_NAMES
        .iter()
        .zip(&stats.page_fallbacks)
    {
        if *count > 0 {
            println!("  {name:24} {count:>10}");
        }
    }
}

/// The same scenes with the batching on and off, measured against each other.
///
/// Frame times on a laptop drift with its temperature over the minutes a run
/// takes, so the two builds are measured a few frames apart rather than a
/// benchmark apart: each size is rendered the old way and the new way back to
/// back, and what is reported is the ratio between them.
#[test]
fn what_the_batching_is_worth() {
    let _exclusive = exclusive();
    let Some(descriptors) = descriptors() else {
        eprintln!("no GPU adapter available; skipping");
        return;
    };
    let mut backend = build_backend(descriptors);
    let bitmap = test_bitmap(&mut backend);

    for (blend_mode, cached, complex) in [
        (BlendMode::Layer, 1.0, 0.0),
        (BlendMode::Layer, 0.0, 0.0),
        (BlendMode::Multiply, 0.0, 0.0),
        (BlendMode::Layer, 0.6, 0.1),
        (BlendMode::Layer, 0.0, 0.2),
        (BlendMode::Layer, 0.0, 1.0),
    ] {
        println!(
            "\n{}",
            if cached == 1.0 && complex == 0.0 {
                "single cached bitmaps - the direct path's own case".to_string()
            } else if cached == 0.0 && complex == 0.0 {
                format!("{blend_mode:?} groups of one bitmap each")
            } else {
                format!(
                    "an AQW-shaped room: {:.0}% cached singles, {:.0}% complex blends",
                    cached * 100.0,
                    complex * 100.0
                )
            }
        );
        println!(
            "{:>7} | {:>7} {:>7} {:>8} {:>7} | {:>7} {:>7} {:>8} {:>7} | {:>6} {:>6}",
            "objects",
            "passes",
            "copies",
            "target MB",
            "mean ms",
            "passes",
            "copies",
            "target MB",
            "mean ms",
            "pass x",
            "time x"
        );
        for count in crowds() {
            let scene = || {
                if cached == 1.0 && complex == 0.0 {
                    crowd(&bitmap, count, BlendMode::Layer)
                } else if cached == 0.0 && complex == 0.0 {
                    crowd_of_pairs(&bitmap, count, blend_mode)
                } else {
                    mixed_room(&bitmap, count, cached, complex)
                }
            };
            let old = measure_scene(&mut backend, false, &scene);
            let new = measure_scene(&mut backend, true, &scene);
            println!(
                "{:>7} | {:>7.0} {:>7.0} {:>8.1} {:>7.1} | {:>7.0} {:>7.0} {:>8.1} {:>7.1} | {:>5.1}x {:>5.1}x",
                count,
                old.passes,
                old.copies,
                old.target_mb,
                old.mean_ms,
                new.passes,
                new.copies,
                new.target_mb,
                new.mean_ms,
                if new.passes > 0.0 {
                    old.passes / new.passes
                } else {
                    0.0
                },
                if new.mean_ms > 0.0 {
                    old.mean_ms / new.mean_ms
                } else {
                    0.0
                },
            );
        }
    }
}

struct Cost {
    passes: f64,
    copies: f64,
    target_mb: f64,
    mean_ms: f64,
}

fn measure_scene(
    backend: &mut WgpuRenderBackend<TextureTarget>,
    batched: bool,
    scene: &impl Fn() -> CommandList,
) -> Cost {
    ruffle_render_wgpu::tuning::set_blend_pages_enabled(batched);
    ruffle_render_wgpu::tuning::set_blend_batching_enabled(batched);
    for _ in 0..WARMUP_FRAMES {
        backend.submit_frame(Color::BLACK, scene(), vec![]);
        backend.capture_frame().expect("capture must succeed");
    }
    let pool_before = pool_usage();
    let before = render_stats();
    let mut times = Vec::with_capacity(MEASURED_FRAMES);
    for _ in 0..MEASURED_FRAMES {
        let commands = scene();
        let start = Instant::now();
        backend.submit_frame(Color::BLACK, commands, vec![]);
        backend.capture_frame().expect("capture must succeed");
        times.push(start.elapsed());
    }
    let after = render_stats();
    let usage = pool_usage() - pool_before;
    let frames = MEASURED_FRAMES as f64;
    ruffle_render_wgpu::tuning::set_blend_pages_enabled(true);
    ruffle_render_wgpu::tuning::set_blend_batching_enabled(true);
    Cost {
        passes: (after.render_passes - before.render_passes) as f64 / frames,
        copies: (after.destination_copies - before.destination_copies) as f64 / frames,
        target_mb: usage.pixels as f64 * 4.0 / (1024.0 * 1024.0) / frames,
        mean_ms: Measurement::mean_ms(&times),
    }
}

/// A crowd whose groups hold two drawables, so none of them can take the direct
/// single-drawable path and every one of them wants a target.
fn crowd_of_pairs(bitmap: &BitmapHandle, count: usize, blend_mode: BlendMode) -> CommandList {
    let mut commands = CommandList::new();
    for i in 0..count {
        let x = ((i as f64 * 0.7548776662) % 1.0) * (VIEWPORT.0 - OBJECT.0) as f64;
        let y = ((i as f64 * 0.5698402909) % 1.0) * (VIEWPORT.1 - OBJECT.1) as f64;
        let mut group = CommandList::new();
        for (dx, dy) in [(0.0, 0.0), (9.0, 7.0)] {
            group.render_bitmap(
                bitmap.clone(),
                Transform {
                    matrix: Matrix::translate(
                        Twips::from_pixels(x + dx),
                        Twips::from_pixels(y + dy),
                    ),
                    color_transform: Default::default(),
                    perspective_projection: None,
                },
                false,
                PixelSnapping::Never,
                ruffle_render::bitmap::PixelRegion::for_whole_size(OBJECT.0, OBJECT.1),
            );
        }
        commands.blend(group, RenderBlendMode::Builtin(blend_mode));
    }
    commands
}
