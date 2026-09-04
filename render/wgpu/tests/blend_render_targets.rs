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
use std::num::NonZeroU32;
use std::sync::Arc;
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
        Err(_) => vec![50, 100, 250, 500, 700],
    }
}

const WARMUP_FRAMES: usize = 3;
const MEASURED_FRAMES: usize = 10;

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

    Measurement {
        usage: pool_usage() - before,
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
            "{:>7}  {:>14}  {:>12}  {:>9}  {:>7}  {:>8}  {:>8}  {:>8}",
            "objects",
            "target px/frame",
            "target MB/fr",
            "builds/fr",
            "reuse",
            "cpu ms",
            "mean ms",
            "p95 ms"
        );
        for count in crowds() {
            let m = measure(&mut backend, &bitmap, count, blend_mode);
            println!(
                "{:>7}  {:>14}  {:>12.1}  {:>9.1}  {:>6.1}%  {:>8.1}  {:>8.1}  {:>8.1}",
                count,
                m.per_frame_pixels(),
                m.per_frame_pixels() as f64 * 4.0 / (1024.0 * 1024.0),
                m.per_frame_builds(),
                m.reuse_rate() * 100.0,
                Measurement::mean_ms(&m.cpu_times),
                Measurement::mean_ms(&m.frame_times),
                Measurement::percentile_ms(&m.frame_times, 0.95),
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
