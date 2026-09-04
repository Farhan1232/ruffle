//! A long run over the scenes a session actually moves through.
//!
//! The counters a benchmark reads say what one steady scene costs. What they do
//! not say is whether anything accumulates over a session that walks from a
//! crowded room to an empty one and back, through complex blends, masks and
//! filters - which is what a player does, and what left the pool holding
//! gigabytes before it was bounded.
//!
//! This runs that walk for as long as it is told to and insists that nothing
//! grows: the pages a frame takes, the render passes it encodes, the textures
//! the renderer owns and the bind groups it builds all have to look the same at
//! the end as they did once the first cycle had warmed everything up.
//!
//! Ignored by default because it is minutes long. Run it with:
//!
//! ```text
//! RUFFLE_SOAK_SECONDS=1200 cargo test --release -p ruffle_render_wgpu \
//!     --test blend_page_soak -- --ignored --nocapture
//! ```

use ruffle_render::backend::{BitmapCacheEntry, RenderBackend, ViewportDimensions};
use ruffle_render::bitmap::{Bitmap, BitmapFormat, BitmapHandle, PixelRegion, PixelSnapping};
use ruffle_render::commands::{CommandHandler, CommandList, RenderBlendMode};
use ruffle_render::filters::Filter;
use ruffle_render::matrix::Matrix;
use ruffle_render::transform::Transform;
use ruffle_render_wgpu::backend::{
    WgpuRenderBackend, create_wgpu_instance, request_adapter_and_device,
};
use ruffle_render_wgpu::buffer_pool::pool_usage;
use ruffle_render_wgpu::descriptors::Descriptors;
use ruffle_render_wgpu::target::TextureTarget;
use ruffle_render_wgpu::{render_stats, tracked_texture_totals, wgpu};
use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::{Duration, Instant};
use swf::{BlendMode, Color, Fixed8, Fixed16, GlowFilter, GlowFilterFlags, Twips};

const VIEWPORT: (u32, u32) = (1280, 720);
const OBJECT: (u32, u32) = (120, 160);

/// One trip through the kinds of scene a session meets, in order.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Phase {
    Crowd,
    Quiet,
    CrowdAgain,
    ComplexBlends,
    Masked,
    Filtered,
    Worst,
}

const CYCLE: &[Phase] = &[
    Phase::Crowd,
    Phase::Quiet,
    Phase::CrowdAgain,
    Phase::ComplexBlends,
    Phase::Masked,
    Phase::Filtered,
    Phase::Worst,
];

/// Frames spent in each phase before moving to the next.
const FRAMES_PER_PHASE: usize = 24;

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

fn place(i: usize, frame: usize) -> Matrix {
    // Everything drifts, so no frame is a repeat of the last one and the pool
    // is asked for a slightly different set of sizes as it goes.
    let x = ((i as f64 * 0.7548776662 + frame as f64 * 0.013) % 1.0)
        * (VIEWPORT.0 - OBJECT.0 - 40) as f64;
    let y = ((i as f64 * 0.5698402909 + frame as f64 * 0.007) % 1.0)
        * (VIEWPORT.1 - OBJECT.1 - 40) as f64;
    Matrix::translate(Twips::from_pixels(x), Twips::from_pixels(y))
}

fn draw(commands: &mut CommandList, bitmap: &BitmapHandle, matrix: Matrix) {
    commands.render_bitmap(
        bitmap.clone(),
        Transform {
            matrix,
            color_transform: Default::default(),
            perspective_projection: None,
        },
        false,
        PixelSnapping::Never,
        PixelRegion::for_whole_size(OBJECT.0, OBJECT.1),
    );
}

fn scene(bitmap: &BitmapHandle, phase: Phase, frame: usize) -> CommandList {
    let (count, complex_every, masked_every, singles) = match phase {
        Phase::Crowd => (400, 0, 0, false),
        Phase::Quiet => (12, 0, 0, true),
        Phase::CrowdAgain => (400, 5, 0, false),
        Phase::ComplexBlends => (300, 1, 0, false),
        Phase::Masked => (200, 4, 3, false),
        Phase::Filtered => (60, 3, 0, true),
        Phase::Worst => (700, 3, 7, false),
    };

    let mut commands = CommandList::new();
    for i in 0..count {
        let at = place(i, frame);
        let mut group = CommandList::new();
        if masked_every > 0 && i % masked_every == 0 {
            group.push_mask();
            group.draw_rect(
                Color::WHITE,
                Matrix::create_box(
                    90.0,
                    120.0,
                    Twips::from_pixels(10.0),
                    Twips::from_pixels(10.0),
                ) * at,
            );
            group.activate_mask();
            draw(&mut group, bitmap, at);
            group.deactivate_mask();
            group.draw_rect(
                Color::WHITE,
                Matrix::create_box(
                    90.0,
                    120.0,
                    Twips::from_pixels(10.0),
                    Twips::from_pixels(10.0),
                ) * at,
            );
            group.pop_mask();
        } else if singles {
            draw(&mut group, bitmap, at);
        } else {
            draw(&mut group, bitmap, at);
            draw(
                &mut group,
                bitmap,
                at * Matrix::translate(Twips::from_pixels(7.0), Twips::from_pixels(31.0)),
            );
            if i % 3 == 0 {
                draw(
                    &mut group,
                    bitmap,
                    at * Matrix::translate(Twips::from_pixels(-5.0), Twips::from_pixels(64.0)),
                );
            }
        }
        let mode = if complex_every > 0 && i % complex_every == 0 {
            match i % 4 {
                0 => BlendMode::Multiply,
                1 => BlendMode::Darken,
                2 => BlendMode::Overlay,
                _ => BlendMode::Difference,
            }
        } else if i % 7 == 0 {
            BlendMode::Add
        } else {
            BlendMode::Layer
        };
        commands.blend(group, RenderBlendMode::Builtin(mode));
    }
    commands
}

/// The filtered `cacheAsBitmap` objects the filter phase redraws, which is what
/// exercises the offscreen pool alongside the pages.
fn cache_entries(
    bitmap: &BitmapHandle,
    caches: &[(BitmapHandle, u32, u32)],
    phase: Phase,
    frame: usize,
) -> Vec<BitmapCacheEntry> {
    if phase != Phase::Filtered {
        return vec![];
    }
    caches
        .iter()
        .enumerate()
        .map(|(i, (handle, width, height))| BitmapCacheEntry {
            handle: handle.clone(),
            commands: {
                let mut commands = CommandList::new();
                commands.render_bitmap(
                    bitmap.clone(),
                    Transform {
                        matrix: Matrix::rotate(((i + frame) as f32) * 0.03),
                        color_transform: Default::default(),
                        perspective_projection: None,
                    },
                    false,
                    PixelSnapping::Never,
                    PixelRegion::for_whole_size(*width, *height),
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

/// What one trip round the cycle cost.
#[derive(Debug, Default, Clone, Copy)]
struct CycleCost {
    frames: u64,
    peak_pages: usize,
    peak_page_bytes: usize,
    peak_passes: u64,
    peak_pool_bytes: u64,
    bind_groups_built: u64,
    /// Bind groups built once each phase had been drawing the same scene for a
    /// while, which is what is left when the scene changes are taken out.
    bind_groups_built_settled: u64,
    /// Blended groups the cycle drew, which is what a bind group built per
    /// blend would be counted against.
    blends: u64,
    tracked_textures: usize,
    tracked_bytes: usize,
    slowest_ms: f64,
}

#[test]
#[ignore = "minutes long; run it deliberately"]
fn a_long_session_settles() {
    let Some(descriptors) = descriptors() else {
        eprintln!("no GPU adapter available; skipping");
        return;
    };
    let seconds: u64 = std::env::var("RUFFLE_SOAK_SECONDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(900);
    let deadline = Instant::now() + Duration::from_secs(seconds);

    let target = TextureTarget::new(&descriptors.device, VIEWPORT).expect("texture target");
    let mut backend = WgpuRenderBackend::new(descriptors, target).expect("render backend");
    backend.set_viewport_dimensions(ViewportDimensions {
        width: VIEWPORT.0,
        height: VIEWPORT.1,
        scale_factor: 1.0,
    });

    let pixels = vec![190u8; (OBJECT.0 * OBJECT.1 * 4) as usize];
    let bitmap = backend
        .register_bitmap(Bitmap::new(OBJECT.0, OBJECT.1, BitmapFormat::Rgba, pixels))
        .expect("bitmap registration");
    let caches: Vec<(BitmapHandle, u32, u32)> = (0..48u32)
        .map(|i| {
            let width = 40 + (i % 13) * 11;
            let height = 36 + (i % 9) * 15;
            let handle = backend
                .create_empty_texture(
                    NonZeroU32::new(width).expect("non-zero"),
                    NonZeroU32::new(height).expect("non-zero"),
                )
                .expect("cache texture");
            (handle, width, height)
        })
        .collect();

    let mut cycles: Vec<CycleCost> = Vec::new();
    let mut frame = 0usize;
    println!(
        "{:>6} {:>7} {:>7} {:>9} {:>9} {:>10} {:>9} {:>10} {:>9}",
        "cycle",
        "frames",
        "pages",
        "page MB",
        "passes",
        "pool MB",
        "bg built",
        "textures",
        "slowest"
    );
    while Instant::now() < deadline {
        let before = render_stats();
        let mut cost = CycleCost::default();
        for phase in CYCLE {
            let phase_start = render_stats();
            let mut settled_start = 0u64;
            for phase_frame in 0..FRAMES_PER_PHASE {
                if phase_frame == FRAMES_PER_PHASE - 8 {
                    settled_start = render_stats().bind_groups_created;
                }
                let commands = scene(&bitmap, *phase, frame);
                let entries = cache_entries(&bitmap, &caches, *phase, frame);
                let start = Instant::now();
                backend.submit_frame(Color::BLACK, commands, entries);
                backend.capture_frame().expect("capture must succeed");
                cost.slowest_ms = cost.slowest_ms.max(start.elapsed().as_secs_f64() * 1000.0);
                frame += 1;
                cost.frames += 1;

                let now = render_stats();
                cost.peak_pages = cost.peak_pages.max(now.pages_last_frame);
                cost.peak_page_bytes = cost.peak_page_bytes.max(now.page_bytes_last_frame);
                cost.peak_passes = cost.peak_passes.max(now.render_passes_last_frame);
            }
            let phase_end = render_stats();
            let over_the_phase = phase_end.bind_groups_created - phase_start.bind_groups_created;
            let once_settled = phase_end.bind_groups_created - settled_start;
            cost.bind_groups_built_settled += once_settled;
            if over_the_phase > 0 {
                println!(
                    "         {phase:?}: {over_the_phase} bind groups over the phase, \
                     {once_settled} once it had settled"
                );
            }
        }
        let after = render_stats();
        let (textures, bytes) = tracked_texture_totals();
        cost.bind_groups_built = after.bind_groups_created - before.bind_groups_created;
        cost.blends = after.fastpath_eligible - before.fastpath_eligible;
        cost.tracked_textures = textures;
        cost.tracked_bytes = bytes;
        cost.peak_pool_bytes = pool_usage().pixels * 4;
        cycles.push(cost);

        println!(
            "{:>6} {:>7} {:>7} {:>9.1} {:>9} {:>10.1} {:>9} {:>10} {:>8.1}ms",
            cycles.len(),
            cost.frames,
            cost.peak_pages,
            cost.peak_page_bytes as f64 / (1024.0 * 1024.0),
            cost.peak_passes,
            cost.tracked_bytes as f64 / (1024.0 * 1024.0),
            cost.bind_groups_built,
            cost.tracked_textures,
            cost.slowest_ms,
        );
        // Bind groups live on the pooled textures they name, so a cache that
        // was missing would build one per blend per frame. This scene drifts,
        // so a group's bounds cross a size class now and then and the pool
        // really does have to build a texture - and a bind group with it. Fifty
        // times fewer than one per blend is the difference between the two,
        // with room to spare.
        assert!(
            cost.bind_groups_built * 50 < cost.blends,
            "cycle {} built {} bind groups for {} blended groups; the cache kept with \
             the pooled textures is not holding",
            cycles.len(),
            cost.bind_groups_built,
            cost.blends
        );
    }

    assert!(
        cycles.len() >= 3,
        "only {} cycles finished; give the soak longer",
        cycles.len()
    );

    // The first cycle builds everything for the first time. What matters is
    // that the ones after it look like each other.
    let settled = &cycles[1..];
    let first = settled[0];
    let last = *settled.last().expect("settled cycles");

    println!(
        "\nsettled at cycle 2, finished at cycle {}: \
         {} -> {} pages, {:.1} -> {:.1} MB of texture, {} -> {} bind groups built",
        cycles.len(),
        first.peak_pages,
        last.peak_pages,
        first.tracked_bytes as f64 / (1024.0 * 1024.0),
        last.tracked_bytes as f64 / (1024.0 * 1024.0),
        first.bind_groups_built,
        last.bind_groups_built,
    );

    assert_eq!(
        first.peak_pages, last.peak_pages,
        "the busiest frame wanted {} pages at the start and {} at the end",
        first.peak_pages, last.peak_pages
    );
    assert_eq!(
        first.peak_passes, last.peak_passes,
        "the busiest frame encoded {} render passes at the start and {} at the end",
        first.peak_passes, last.peak_passes
    );
    assert!(
        last.tracked_bytes <= first.tracked_bytes + first.tracked_bytes / 20,
        "the renderer held {:.1} MB of texture after the first cycle and {:.1} MB at the end",
        first.tracked_bytes as f64 / (1024.0 * 1024.0),
        last.tracked_bytes as f64 / (1024.0 * 1024.0),
    );
    // Most of the bind groups still built go on two scene *changes*: the pool
    // gives up the sizes a scene has stopped asking for, and a size that comes
    // back comes back as new textures needing their bind groups again, which is
    // the price of handing that memory back. The rest is the drift crossing a
    // size class. Neither may grow, which is what comparing the two halves of a
    // long run says.
    let half = settled.len() / 2;
    let early: u64 = settled[..half].iter().map(|c| c.bind_groups_built).sum();
    let late: u64 = settled[half..].iter().map(|c| c.bind_groups_built).sum();
    let settled_total: u64 = settled.iter().map(|c| c.bind_groups_built_settled).sum();
    println!(
        "bind groups: {early} over the first {half} settled cycles, {late} over the last {}, \
         of which {settled_total} were built by scenes that had been drawing the same \
         thing for sixteen frames",
        settled.len() - half
    );
    assert!(
        late * 4 <= early * 5,
        "{early} bind groups over the first half of the run and {late} over the second; \
         something is building more of them as it goes"
    );
}
