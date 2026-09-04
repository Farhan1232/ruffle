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
use ruffle_render::cache_capacity::{capacity_fits, capacity_for};
use ruffle_render::commands::{CommandHandler, CommandList, RenderBlendMode};
use ruffle_render::filters::Filter;
use ruffle_render::matrix::Matrix;
use ruffle_render::transform::Transform;
use ruffle_render_wgpu::backend::{
    WgpuRenderBackend, create_wgpu_instance, request_adapter_and_device,
};
use ruffle_render_wgpu::buffer_pool::{PoolKind, pool_telemetry, pool_usage};
use ruffle_render_wgpu::cache_pool::cache_pool_counters;
use ruffle_render_wgpu::descriptors::Descriptors;
use ruffle_render_wgpu::target::TextureTarget;
use ruffle_render_wgpu::texture_churn;
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

/// What one trip round the cycle cost.
/// `(allocated, reserved, blocks)` as the graphics allocator sees it.
fn allocator_report(backend: &WgpuRenderBackend<TextureTarget>) -> (u64, u64, usize) {
    backend
        .descriptors()
        .device
        .generate_allocator_report()
        .map(|report| {
            (
                report.total_allocated_bytes,
                report.total_reserved_bytes,
                report.blocks.len(),
            )
        })
        .unwrap_or_default()
}

/// Cached display objects whose picture changes size every frame.
///
/// Each one keeps its texture while the capacity policy says the picture still
/// fits, and allocates a new one when it does not - which is exactly what
/// `BitmapCache::update` does, through the same shared policy.
struct BreathingCaches {
    actors: Vec<Option<(BitmapHandle, (u32, u32))>>,
    built: u64,
}

impl BreathingCaches {
    fn new(count: usize) -> Self {
        Self {
            actors: vec![None; count],
            built: 0,
        }
    }

    /// The picture's size this frame: a base per actor, a few pixels of
    /// breathing out of phase with its neighbours, and a step every so often
    /// for an equipment change.
    fn logical(index: usize, frame: usize) -> (u32, u32) {
        let base = (40 + (index as u32 % 13) * 11, 36 + (index as u32 % 9) * 15);
        let phase = frame + index * 7;
        let equipment = ((frame / 200 + index) % 3) as u32 * 9;
        (
            base.0 + [0u32, 4, 2, 6, 0, 3][phase % 6] + equipment,
            base.1 + [0u32, 2, 1, 3, 2, 0][(phase + index) % 6] + equipment,
        )
    }

    fn entries(
        &mut self,
        backend: &mut WgpuRenderBackend<TextureTarget>,
        bitmap: &BitmapHandle,
        frame: usize,
    ) -> Vec<BitmapCacheEntry> {
        let mut entries = Vec::with_capacity(self.actors.len());
        for index in 0..self.actors.len() {
            let logical = Self::logical(index, frame);
            let keep = matches!(&self.actors[index], Some((_, physical)) if capacity_fits(*physical, logical));
            if !keep {
                let physical = capacity_for(logical.0, logical.1);
                let handle = backend
                    .create_empty_texture(
                        NonZeroU32::new(physical.0).expect("non-zero"),
                        NonZeroU32::new(physical.1).expect("non-zero"),
                    )
                    .expect("cache texture");
                self.actors[index] = Some((handle, physical));
                self.built += 1;
            }
            let handle = self.actors[index]
                .as_ref()
                .expect("just allocated")
                .0
                .clone();
            let mut commands = CommandList::new();
            commands.render_bitmap(
                bitmap.clone(),
                Transform {
                    matrix: Matrix::rotate(((index + frame) as f32) * 0.03),
                    color_transform: Default::default(),
                    perspective_projection: None,
                },
                false,
                PixelSnapping::Never,
                PixelRegion::for_whole_size(logical.0.min(OBJECT.0), logical.1.min(OBJECT.1)),
            );
            entries.push(BitmapCacheEntry {
                handle,
                commands,
                clear: Color::from_rgba(0),
                filters: vec![Filter::GlowFilter(GlowFilter {
                    color: Color::WHITE,
                    blur_x: Fixed16::from_f32(4.0),
                    blur_y: Fixed16::from_f32(4.0),
                    strength: Fixed8::ONE,
                    flags: GlowFilterFlags::from_passes(1),
                })],
                logical_width: logical.0,
                logical_height: logical.1,
            });
        }
        entries
    }
}

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
    /// Phase 2: what the cycle really allocated, and what the graphics
    /// allocator is holding to serve it.
    cache_textures_built: u64,
    textures_created: u64,
    texture_bytes_created: u64,
    offscreen_misses: u64,
    cache_pool_takes: u64,
    cache_pool_hits: u64,
    allocator_allocated: u64,
    allocator_reserved: u64,
    allocator_blocks: usize,
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
    // Cached objects whose bounds breathe, the way an animating avatar's do.
    // The soak used to allocate these once and keep them for the whole run,
    // which is the one thing a real session never does: the churn phase 2 is
    // chasing is a cache being rebuilt because its picture moved by a pixel.
    let mut caches = BreathingCaches::new(48);

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
        let churn_before = texture_churn();
        let offscreen_before = pool_telemetry(PoolKind::Offscreen).misses_total();
        let cache_pool_before = cache_pool_counters();
        let built_before = caches.built;
        let mut cost = CycleCost::default();
        for phase in CYCLE {
            let phase_start = render_stats();
            let mut settled_start = 0u64;
            for phase_frame in 0..FRAMES_PER_PHASE {
                if phase_frame == FRAMES_PER_PHASE - 8 {
                    settled_start = render_stats().bind_groups_created;
                }
                let commands = scene(&bitmap, *phase, frame);
                let entries = if *phase == Phase::Filtered {
                    caches.entries(&mut backend, &bitmap, frame)
                } else {
                    vec![]
                };
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
        let churn_after = texture_churn();
        let cache_pool_after = cache_pool_counters();
        let allocator = allocator_report(&backend);
        cost.cache_textures_built = caches.built - built_before;
        cost.textures_created = churn_after.0 - churn_before.0;
        cost.texture_bytes_created = churn_after.2 - churn_before.2;
        cost.offscreen_misses =
            pool_telemetry(PoolKind::Offscreen).misses_total() - offscreen_before;
        cost.cache_pool_takes = cache_pool_after.takes - cache_pool_before.takes;
        cost.cache_pool_hits = cache_pool_after.hits - cache_pool_before.hits;
        cost.allocator_allocated = allocator.0;
        cost.allocator_reserved = allocator.1;
        cost.allocator_blocks = allocator.2;
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
        println!(
            "         allocated {} textures ({:.1} MB), {} of them cache rebuilds; \
             cache pool {}/{} recycled; offscreen misses {}; \
             allocator {:.1}/{:.1} MB in {} blocks",
            cost.textures_created,
            cost.texture_bytes_created as f64 / (1024.0 * 1024.0),
            cost.cache_textures_built,
            cost.cache_pool_hits,
            cost.cache_pool_takes,
            cost.offscreen_misses,
            cost.allocator_allocated as f64 / (1024.0 * 1024.0),
            cost.allocator_reserved as f64 / (1024.0 * 1024.0),
            cost.allocator_blocks,
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

    // --- phase 2: allocation traffic, and what the allocator holds to serve it
    let early_created: u64 = settled[..half].iter().map(|c| c.textures_created).sum();
    let late_created: u64 = settled[half..].iter().map(|c| c.textures_created).sum();
    let early_cache: u64 = settled[..half].iter().map(|c| c.cache_textures_built).sum();
    let late_cache: u64 = settled[half..].iter().map(|c| c.cache_textures_built).sum();
    println!(
        "\ntextures allocated: {early_created} over the first {half} settled cycles, \
         {late_created} over the last {}; cache rebuilds {early_cache} -> {late_cache}",
        settled.len() - half
    );
    println!(
        "allocator: {:.1}/{:.1} MB in {} blocks after the first settled cycle, \
         {:.1}/{:.1} MB in {} blocks at the end",
        first.allocator_allocated as f64 / (1024.0 * 1024.0),
        first.allocator_reserved as f64 / (1024.0 * 1024.0),
        first.allocator_blocks,
        last.allocator_allocated as f64 / (1024.0 * 1024.0),
        last.allocator_reserved as f64 / (1024.0 * 1024.0),
        last.allocator_blocks,
    );

    // A room that keeps coming back to scenes it has already drawn should stop
    // allocating. This is the phase 2 target: not zero - the pool gives sizes
    // back and a drifting scene meets new ones - but bounded, and no worse in
    // the second half of a long run than in the first.
    assert!(
        late_created * 4 <= early_created * 5,
        "{early_created} textures allocated over the first half of the run and \
         {late_created} over the second; the creation rate is not settling"
    );
    assert!(
        late_cache * 4 <= early_cache * 5 + 4,
        "{early_cache} cache rebuilds over the first half and {late_cache} over the \
         second; the capacity is not holding"
    );

    // The reserve is what the driver's allocator keeps in order to serve the
    // allocations. It is the part that shows up in the process's private bytes
    // without appearing in any count of live textures, and it is the shape of
    // the memory the client's Windows run could not account for.
    //
    // It cannot be tested by comparing its first reading with its last. A
    // suballocator takes and releases whole blocks, so a scene whose demand
    // rises and falls makes this oscillate between two values - here between
    // two blocks and three - and which of them a run happens to end on says
    // nothing. What matters is the ceiling: a reserve that is really ratcheting
    // reaches a higher one in the second half of a long run than it did in the
    // first, and one that has found its high-water mark does not.
    let reserve_max = |cycles: &[CycleCost]| {
        cycles
            .iter()
            .map(|c| c.allocator_reserved)
            .max()
            .unwrap_or(0)
    };
    let reserve_min = |cycles: &[CycleCost]| {
        cycles
            .iter()
            .map(|c| c.allocator_reserved)
            .min()
            .unwrap_or(0)
    };
    let early_reserve = reserve_max(&settled[..half]);
    let late_reserve = reserve_max(&settled[half..]);
    println!(
        "allocator reserve: {:.1}-{:.1} MB over the first {half} settled cycles, \
         {:.1}-{:.1} MB over the last {}",
        reserve_min(&settled[..half]) as f64 / (1024.0 * 1024.0),
        early_reserve as f64 / (1024.0 * 1024.0),
        reserve_min(&settled[half..]) as f64 / (1024.0 * 1024.0),
        late_reserve as f64 / (1024.0 * 1024.0),
        settled.len() - half,
    );
    if early_reserve > 0 {
        assert!(
            late_reserve <= early_reserve,
            "the allocator's reserve peaked at {:.1} MB over the first half of the run \
             and at {:.1} MB over the second; it is ratcheting",
            early_reserve as f64 / (1024.0 * 1024.0),
            late_reserve as f64 / (1024.0 * 1024.0),
        );
    }

    // And what is actually in use inside that reserve is the leak test: it is
    // flat if nothing is accumulating, whatever the reserve around it does.
    let early_allocated = settled[..half]
        .iter()
        .map(|c| c.allocator_allocated)
        .max()
        .unwrap_or(0);
    let late_allocated = settled[half..]
        .iter()
        .map(|c| c.allocator_allocated)
        .max()
        .unwrap_or(0);
    if early_allocated > 0 {
        assert!(
            late_allocated <= early_allocated * 5 / 4,
            "the allocator held {:.1} MB at once over the first half of the run and \
             {:.1} MB over the second; something is accumulating",
            early_allocated as f64 / (1024.0 * 1024.0),
            late_allocated as f64 / (1024.0 * 1024.0),
        );
    }
}
