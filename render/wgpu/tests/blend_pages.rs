//! That sharing a page changes nothing about the picture.
//!
//! A blended group used to be rendered through a render target of its own and
//! is now usually given a region of one shared with its siblings, and
//! consecutive complex blends that cannot see each other's work are composited
//! in one render pass rather than one each. Neither is allowed to move a pixel.
//!
//! Both are switches, so every test here renders the same scene twice - once
//! the new way, once the old - and compares the two frames pixel for pixel.
//! That is a stronger check than a stored reference image: it holds for every
//! blend mode, transform and arrangement the tests care to build, and it fails
//! on the actual difference rather than on a tolerance.
//!
//! ```text
//! cargo test --release -p ruffle_render_wgpu --test blend_pages -- --nocapture
//! ```

use ruffle_render::backend::{RenderBackend, ViewportDimensions};
use ruffle_render::bitmap::{Bitmap, BitmapFormat, BitmapHandle, PixelRegion, PixelSnapping};
use ruffle_render::commands::{CommandHandler, CommandList, RenderBlendMode};
use ruffle_render::matrix::Matrix;
use ruffle_render::quality::StageQuality;
use ruffle_render::transform::Transform;
use ruffle_render_wgpu::backend::{
    WgpuRenderBackend, create_wgpu_instance, request_adapter_and_device,
};
use ruffle_render_wgpu::descriptors::Descriptors;
use ruffle_render_wgpu::target::TextureTarget;
use ruffle_render_wgpu::tuning;
use ruffle_render_wgpu::wgpu;
use ruffle_render_wgpu::{PageFallback, render_stats};
use std::sync::{Arc, Mutex, MutexGuard};
use swf::{BlendMode, Color, ColorTransform, Fixed8, Twips};

const VIEWPORT: (u32, u32) = (640, 400);
const SPRITE: (u32, u32) = (40, 56);

/// One GPU, one set of process-wide switches.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn exclusive() -> MutexGuard<'static, ()> {
    let guard = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    tuning::set_blend_pages_enabled(true);
    tuning::set_blend_batching_enabled(true);
    guard
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

fn build_backend(
    descriptors: Arc<Descriptors>,
    quality: StageQuality,
) -> WgpuRenderBackend<TextureTarget> {
    let target = TextureTarget::new(&descriptors.device, VIEWPORT).expect("texture target");
    let mut backend = WgpuRenderBackend::new(descriptors, target).expect("render backend");
    backend.set_viewport_dimensions(ViewportDimensions {
        width: VIEWPORT.0,
        height: VIEWPORT.1,
        scale_factor: 1.0,
    });
    backend.set_quality(quality);
    backend
}

/// A sprite with a different colour in each quadrant and a transparent corner,
/// so that anything which flips, offsets or bleeds it is visible.
fn sprite(backend: &mut WgpuRenderBackend<TextureTarget>) -> BitmapHandle {
    let mut pixels = Vec::with_capacity((SPRITE.0 * SPRITE.1 * 4) as usize);
    for y in 0..SPRITE.1 {
        for x in 0..SPRITE.0 {
            let left = x < SPRITE.0 / 2;
            let top = y < SPRITE.1 / 2;
            let (r, g, b, a) = match (left, top) {
                (true, true) => (230, 40, 40, 255),
                (false, true) => (40, 220, 60, 255),
                (true, false) => (50, 60, 240, 255),
                // Transparent, and premultiplied, so a blend that reads the
                // destination has somewhere to show through.
                (false, false) => (0, 0, 0, 0),
            };
            pixels.extend_from_slice(&[r, g, b, a]);
        }
    }
    backend
        .register_bitmap(Bitmap::new(SPRITE.0, SPRITE.1, BitmapFormat::Rgba, pixels))
        .expect("bitmap registration")
}

/// Every blend mode the renderer supports.
const ALL_BLEND_MODES: &[BlendMode] = &[
    BlendMode::Normal,
    BlendMode::Layer,
    BlendMode::Multiply,
    BlendMode::Screen,
    BlendMode::Lighten,
    BlendMode::Darken,
    BlendMode::Difference,
    BlendMode::Add,
    BlendMode::Subtract,
    BlendMode::Invert,
    BlendMode::Alpha,
    BlendMode::Erase,
    BlendMode::Overlay,
    BlendMode::HardLight,
];

/// The placements a group can arrive under, each of which the region it is
/// given has to reproduce exactly.
fn placements() -> Vec<(&'static str, Matrix)> {
    let rotate = |angle: f32| Matrix {
        a: angle.cos(),
        b: angle.sin(),
        c: -angle.sin(),
        d: angle.cos(),
        tx: Twips::ZERO,
        ty: Twips::ZERO,
    };
    vec![
        ("whole pixels", Matrix::IDENTITY),
        (
            "fractional translation",
            Matrix::translate(Twips::from_pixels(0.37), Twips::from_pixels(0.62)),
        ),
        ("rotation", rotate(0.3)),
        (
            "skew",
            Matrix {
                a: 1.0,
                b: 0.0,
                c: 0.35,
                d: 1.0,
                tx: Twips::ZERO,
                ty: Twips::ZERO,
            },
        ),
        (
            "negative scale",
            Matrix {
                a: -1.0,
                b: 0.0,
                c: 0.0,
                d: 1.25,
                tx: Twips::from_pixels(60.0),
                ty: Twips::ZERO,
            },
        ),
        (
            "fractional scale",
            Matrix {
                a: 1.37,
                b: 0.0,
                c: 0.0,
                d: 0.83,
                tx: Twips::ZERO,
                ty: Twips::ZERO,
            },
        ),
    ]
}

/// A position on the twip grid that is not exactly half a pixel from a whole
/// one.
///
/// Everything Flash draws is on a twentieth of a pixel, and a bitmap whose
/// origin is at exactly half a pixel puts every texel boundary on a sample
/// point - a tie a nearest sampler breaks arbitrarily, and one of the few
/// things a page can decide differently. Scenes that are about anything else
/// step around it, and
/// `a_texel_boundary_picks_a_neighbouring_texel_at_worst` measures it head on.
fn off_the_half_pixel(pixels: f64) -> Twips {
    let mut twips = (pixels * 20.0).round() as i32;
    if twips.rem_euclid(20) == 10 {
        twips += 1;
    }
    Twips::new(twips)
}

fn draw_sprite(
    commands: &mut CommandList,
    bitmap: &BitmapHandle,
    matrix: Matrix,
    color_transform: ColorTransform,
) {
    commands.render_bitmap(
        bitmap.clone(),
        Transform {
            matrix,
            color_transform,
            perspective_projection: None,
        },
        false,
        PixelSnapping::Never,
        PixelRegion::for_whole_size(SPRITE.0, SPRITE.1),
    );
}

/// A row of blended groups, each of two or three overlapping children so that
/// none of them can take the direct single-drawable path.
///
/// `offstage` pushes the first and last groups over the edges of the viewport,
/// which is what makes a group's rectangle get clipped.
fn row_of_groups(
    bitmap: &BitmapHandle,
    mode: BlendMode,
    placement: Matrix,
    count: usize,
    offstage: bool,
) -> CommandList {
    let mut commands = CommandList::new();
    // Something underneath for the destination-reading modes to read.
    draw_sprite(
        &mut commands,
        bitmap,
        Matrix::create_box(
            VIEWPORT.0 as f32 / SPRITE.0 as f32,
            VIEWPORT.1 as f32 / SPRITE.1 as f32,
            Twips::ZERO,
            Twips::ZERO,
        ),
        ColorTransform::default(),
    );

    for i in 0..count {
        let x = if offstage && i == 0 {
            -22.0
        } else if offstage && i + 1 == count {
            VIEWPORT.0 as f64 - 18.0
        } else {
            18.0 + (i as f64) * 61.0
        };
        let y = 30.0 + ((i % 3) as f64) * 47.0;
        let place = Matrix::translate(Twips::from_pixels(x), Twips::from_pixels(y)) * placement;

        let mut group = CommandList::new();
        draw_sprite(&mut group, bitmap, place, ColorTransform::default());
        draw_sprite(
            &mut group,
            bitmap,
            place * Matrix::translate(Twips::from_pixels(11.0), Twips::from_pixels(9.0)),
            ColorTransform {
                r_multiply: Fixed8::from_f32(0.6),
                ..Default::default()
            },
        );
        if i % 2 == 0 {
            draw_sprite(
                &mut group,
                bitmap,
                place * Matrix::translate(Twips::from_pixels(-7.0), Twips::from_pixels(18.0)),
                ColorTransform::default(),
            );
        }
        commands.blend(group, RenderBlendMode::Builtin(mode));
    }
    commands
}

/// `Alpha` and `Erase` do nothing unless there is a layer above them, so every
/// scene is wrapped in one. It also means the outer group is one the pages
/// cannot take - it draws blended groups of its own - which is the nesting the
/// fallback has to keep working.
fn in_a_layer(commands: CommandList) -> CommandList {
    let mut outer = CommandList::new();
    outer.blend(commands, RenderBlendMode::Builtin(BlendMode::Layer));
    outer
}

fn render(
    backend: &mut WgpuRenderBackend<TextureTarget>,
    commands: CommandList,
) -> image::RgbaImage {
    backend.submit_frame(Color::from_rgb(0x30405a, 255), commands, vec![]);
    backend.capture_frame().expect("capture must succeed")
}

/// Renders `commands` with the batching on and off, and insists the two frames
/// are the same image.
fn assert_batching_is_invisible(
    backend: &mut WgpuRenderBackend<TextureTarget>,
    what: &str,
    build: impl Fn() -> CommandList,
) {
    tuning::set_blend_pages_enabled(true);
    tuning::set_blend_batching_enabled(true);
    let batched = render(backend, build());

    tuning::set_blend_pages_enabled(false);
    tuning::set_blend_batching_enabled(false);
    let separate = render(backend, build());

    tuning::set_blend_pages_enabled(true);
    tuning::set_blend_batching_enabled(true);

    assert_same(&batched, &separate, what);
}

fn assert_same(batched: &image::RgbaImage, separate: &image::RgbaImage, what: &str) {
    assert_eq!(batched.dimensions(), separate.dimensions(), "{what}");
    let mut differing = 0usize;
    let mut worst = (0u32, 0u32, [0u8; 4], [0u8; 4], 0i32);
    for (x, y, batched_pixel) in batched.enumerate_pixels() {
        let separate_pixel = separate.get_pixel(x, y);
        if batched_pixel != separate_pixel {
            differing += 1;
            let delta = batched_pixel
                .0
                .iter()
                .zip(separate_pixel.0.iter())
                .map(|(a, b)| (*a as i32 - *b as i32).abs())
                .max()
                .unwrap_or(0);
            if delta > worst.4 {
                worst = (x, y, batched_pixel.0, separate_pixel.0, delta);
            }
        }
    }
    assert_eq!(
        differing, 0,
        "{what}: {differing} pixels differ between the batched and the separate render; \
         worst at ({}, {}) - batched {:?}, separate {:?}",
        worst.0, worst.1, worst.2, worst.3,
    );
}

/// The whole point, for every blend mode and every way a group can be placed.
#[test]
fn every_blend_mode_survives_sharing_a_page() {
    let _exclusive = exclusive();
    let Some(descriptors) = descriptors() else {
        eprintln!("no GPU adapter available; skipping");
        return;
    };
    for quality in [StageQuality::Low, StageQuality::High] {
        let mut backend = build_backend(descriptors.clone(), quality);
        let bitmap = sprite(&mut backend);
        for mode in ALL_BLEND_MODES {
            for (name, placement) in placements() {
                assert_batching_is_invisible(
                    &mut backend,
                    &format!("{mode:?} under {name} at {quality:?} quality"),
                    || in_a_layer(row_of_groups(&bitmap, *mode, placement, 7, false)),
                );
            }
        }
    }
}

/// Groups that hang off the edges of the viewport, which is what makes a
/// region's rectangle get clipped to something smaller than the group.
#[test]
fn groups_over_the_edge_survive_sharing_a_page() {
    let _exclusive = exclusive();
    let Some(descriptors) = descriptors() else {
        eprintln!("no GPU adapter available; skipping");
        return;
    };
    let mut backend = build_backend(descriptors, StageQuality::High);
    let bitmap = sprite(&mut backend);
    for mode in [BlendMode::Layer, BlendMode::Multiply, BlendMode::Alpha] {
        assert_batching_is_invisible(&mut backend, &format!("{mode:?} partly offstage"), || {
            in_a_layer(row_of_groups(&bitmap, mode, Matrix::IDENTITY, 7, true))
        });
    }
}

/// Regions next to each other on a page, holding saturated colour and nothing
/// at all.
///
/// A group that draws nothing must composite to nothing, whatever its
/// neighbours on the page hold. If a region ever sampled past its own edge, the
/// red one's pixels would appear where the empty one is.
#[test]
fn a_neighbour_on_a_page_does_not_bleed_through() {
    let _exclusive = exclusive();
    let Some(descriptors) = descriptors() else {
        eprintln!("no GPU adapter available; skipping");
        return;
    };
    let mut backend = build_backend(descriptors, StageQuality::High);
    let bitmap = sprite(&mut backend);

    let background = Color::from_rgb(0x30405a, 255);
    // A wall of colour, an empty group, and more colour - as consecutive
    // groups, so their regions are next to each other on the shelf.
    let build = || {
        let mut commands = CommandList::new();
        let solid = |commands: &mut CommandList, x: f64, color: Color| {
            let mut group = CommandList::new();
            group.draw_rect(
                color,
                Matrix::create_box(52.0, 52.0, Twips::from_pixels(x), Twips::from_pixels(40.0)),
            );
            group.draw_rect(
                color,
                Matrix::create_box(52.0, 52.0, Twips::from_pixels(x), Twips::from_pixels(94.0)),
            );
            commands.blend(group, RenderBlendMode::Builtin(BlendMode::Layer));
        };
        solid(&mut commands, 20.0, Color::from_rgb(0xff0000, 255));
        // Nothing at all, in the middle of the row.
        let mut empty = CommandList::new();
        empty.draw_rect(
            Color::from_rgba(0),
            Matrix::create_box(
                52.0,
                52.0,
                Twips::from_pixels(100.0),
                Twips::from_pixels(40.0),
            ),
        );
        empty.draw_rect(
            Color::from_rgba(0),
            Matrix::create_box(
                52.0,
                52.0,
                Twips::from_pixels(100.0),
                Twips::from_pixels(94.0),
            ),
        );
        commands.blend(empty, RenderBlendMode::Builtin(BlendMode::Layer));
        solid(&mut commands, 180.0, Color::from_rgb(0x00ff00, 255));
        solid(&mut commands, 260.0, Color::from_rgb(0x0000ff, 255));
        // And something that is not axis aligned over the lot of them, so the
        // regions are not all the same size.
        let mut tilted = CommandList::new();
        draw_sprite(
            &mut tilted,
            &bitmap,
            Matrix {
                a: 0.94,
                b: 0.34,
                c: -0.34,
                d: 0.94,
                tx: Twips::from_pixels(120.5),
                ty: Twips::from_pixels(180.5),
            },
            ColorTransform::default(),
        );
        draw_sprite(
            &mut tilted,
            &bitmap,
            Matrix {
                a: 0.94,
                b: 0.34,
                c: -0.34,
                d: 0.94,
                tx: Twips::from_pixels(150.5),
                ty: Twips::from_pixels(190.5),
            },
            ColorTransform::default(),
        );
        commands.blend(tilted, RenderBlendMode::Builtin(BlendMode::Multiply));
        commands
    };

    assert_batching_is_invisible(&mut backend, "neighbouring regions", build);

    // And say the invariant outright: where the empty group is, the frame is
    // still the background.
    let frame = render(&mut backend, build());
    for y in 50..135 {
        for x in 110..142 {
            let pixel = frame.get_pixel(x, y).0;
            assert_eq!(
                [pixel[0], pixel[1], pixel[2]],
                [background.r, background.g, background.b],
                "a neighbouring region bled into the empty group's area at ({x}, {y})"
            );
        }
    }
}

/// A page comes out of a pool, so it holds whatever the last scene left in it.
///
/// Frame one fills a region with white; frame two draws a smaller, transparent
/// group where the white was. None of the white may survive.
#[test]
fn a_page_region_does_not_show_the_last_frame() {
    let _exclusive = exclusive();
    let Some(descriptors) = descriptors() else {
        eprintln!("no GPU adapter available; skipping");
        return;
    };
    let mut backend = build_backend(descriptors, StageQuality::High);

    let bright = || {
        let mut commands = CommandList::new();
        for i in 0..12 {
            let mut group = CommandList::new();
            for row in 0..2 {
                group.draw_rect(
                    Color::WHITE,
                    Matrix::create_box(
                        130.0,
                        130.0,
                        Twips::from_pixels(6.0 + i as f64 * 3.0),
                        Twips::from_pixels(6.0 + row as f64 * 140.0),
                    ),
                );
            }
            commands.blend(group, RenderBlendMode::Builtin(BlendMode::Layer));
        }
        commands
    };
    let faint = || {
        let mut commands = CommandList::new();
        for i in 0..12 {
            let mut group = CommandList::new();
            for row in 0..2 {
                group.draw_rect(
                    Color::from_rgba(0),
                    Matrix::create_box(
                        40.0,
                        40.0,
                        Twips::from_pixels(6.0 + i as f64 * 3.0),
                        Twips::from_pixels(6.0 + row as f64 * 140.0),
                    ),
                );
            }
            commands.blend(group, RenderBlendMode::Builtin(BlendMode::Layer));
        }
        commands
    };

    let background = Color::from_rgb(0x30405a, 255);
    render(&mut backend, bright());
    let after = render(&mut backend, faint());
    for (x, y, pixel) in after.enumerate_pixels() {
        assert_eq!(
            [pixel.0[0], pixel.0[1], pixel.0[2]],
            [background.r, background.g, background.b],
            "a page kept the last frame's pixels at ({x}, {y})"
        );
    }
}

/// Complex blends that overlap must not be composited together: each one reads
/// the destination the one before it wrote.
#[test]
fn overlapping_complex_blends_keep_their_order() {
    let _exclusive = exclusive();
    let Some(descriptors) = descriptors() else {
        eprintln!("no GPU adapter available; skipping");
        return;
    };
    let mut backend = build_backend(descriptors, StageQuality::High);
    let bitmap = sprite(&mut backend);

    // Three groups stacked on the same pixels, each of which changes what the
    // next one reads.
    let build = || {
        let mut commands = CommandList::new();
        draw_sprite(
            &mut commands,
            &bitmap,
            Matrix::create_box(
                VIEWPORT.0 as f32 / SPRITE.0 as f32,
                VIEWPORT.1 as f32 / SPRITE.1 as f32,
                Twips::ZERO,
                Twips::ZERO,
            ),
            ColorTransform::default(),
        );
        for (i, mode) in [
            BlendMode::Multiply,
            BlendMode::Difference,
            BlendMode::HardLight,
            BlendMode::Overlay,
        ]
        .into_iter()
        .enumerate()
        {
            let mut group = CommandList::new();
            let at = Matrix::translate(
                Twips::from_pixels(90.0 + i as f64 * 6.0),
                Twips::from_pixels(70.0 + i as f64 * 5.0),
            );
            draw_sprite(&mut group, &bitmap, at, ColorTransform::default());
            draw_sprite(
                &mut group,
                &bitmap,
                at * Matrix::translate(Twips::from_pixels(14.0), Twips::from_pixels(12.0)),
                ColorTransform::default(),
            );
            commands.blend(group, RenderBlendMode::Builtin(mode));
        }
        commands
    };

    assert_batching_is_invisible(&mut backend, "overlapping complex blends", build);

    let before = render_stats();
    render(&mut backend, build());
    let after = render_stats();
    let blends = after.complex_blends - before.complex_blends;
    let passes = after.complex_blend_passes - before.complex_blend_passes;
    assert_eq!(
        blends, passes,
        "{blends} overlapping complex blends were composited in {passes} passes; \
         each of them reads what the one before it wrote and must have its own"
    );
}

/// Complex blends that share no pixels are composited together, which is the
/// point of the exercise.
#[test]
fn separated_complex_blends_share_a_pass() {
    let _exclusive = exclusive();
    let Some(descriptors) = descriptors() else {
        eprintln!("no GPU adapter available; skipping");
        return;
    };
    let mut backend = build_backend(descriptors, StageQuality::High);
    let bitmap = sprite(&mut backend);

    // A grid with gaps, so no two groups can see each other.
    let build = || {
        let mut commands = CommandList::new();
        draw_sprite(
            &mut commands,
            &bitmap,
            Matrix::create_box(
                VIEWPORT.0 as f32 / SPRITE.0 as f32,
                VIEWPORT.1 as f32 / SPRITE.1 as f32,
                Twips::ZERO,
                Twips::ZERO,
            ),
            ColorTransform::default(),
        );
        for row in 0..3 {
            for column in 0..6 {
                let mut group = CommandList::new();
                let at = Matrix::translate(
                    Twips::from_pixels(6.0 + column as f64 * 78.0),
                    Twips::from_pixels(6.0 + row as f64 * 100.0),
                );
                draw_sprite(&mut group, &bitmap, at, ColorTransform::default());
                draw_sprite(
                    &mut group,
                    &bitmap,
                    at * Matrix::translate(Twips::from_pixels(9.0), Twips::from_pixels(7.0)),
                    ColorTransform::default(),
                );
                commands.blend(group, RenderBlendMode::Builtin(BlendMode::Multiply));
            }
        }
        commands
    };

    assert_batching_is_invisible(&mut backend, "separated complex blends", build);

    // And measure the batched render on its own.
    let before = render_stats();
    render(&mut backend, build());
    let after = render_stats();
    let blends = after.complex_blends - before.complex_blends;
    let passes = after.complex_blend_passes - before.complex_blend_passes;
    assert_eq!(blends, 18, "the scene should hold eighteen complex blends");
    assert!(
        passes < 4,
        "{blends} separated complex blends took {passes} passes; \
         they share no pixels and should have shared one"
    );
}

/// A room-shaped scene: hundreds of groups on a handful of pages, and the same
/// picture as hundreds of targets gave.
#[test]
fn a_crowd_on_pages_looks_like_a_crowd_on_targets() {
    let _exclusive = exclusive();
    let Some(descriptors) = descriptors() else {
        eprintln!("no GPU adapter available; skipping");
        return;
    };
    let mut backend = build_backend(descriptors, StageQuality::Low);
    let bitmap = sprite(&mut backend);

    let build = || {
        let mut commands = CommandList::new();
        for i in 0..220 {
            let x = ((i as f64 * 0.7548776662) % 1.0) * (VIEWPORT.0 - SPRITE.0) as f64;
            let y = ((i as f64 * 0.5698402909) % 1.0) * (VIEWPORT.1 - SPRITE.1) as f64;
            let at = Matrix::translate(off_the_half_pixel(x), off_the_half_pixel(y));
            let mut group = CommandList::new();
            draw_sprite(&mut group, &bitmap, at, ColorTransform::default());
            draw_sprite(
                &mut group,
                &bitmap,
                at * Matrix::translate(Twips::from_pixels(5.0), Twips::from_pixels(11.0)),
                ColorTransform::default(),
            );
            let mode = match i % 5 {
                0 => BlendMode::Layer,
                1 => BlendMode::Multiply,
                2 => BlendMode::Screen,
                3 => BlendMode::Darken,
                _ => BlendMode::Add,
            };
            commands.blend(group, RenderBlendMode::Builtin(mode));
        }
        commands
    };

    assert_batching_is_invisible(&mut backend, "a crowd", build);

    // And what the paged render cost, measured on its own.
    let before = render_stats();
    render(&mut backend, build());
    let after = render_stats();
    println!(
        "220 groups: {} pages, {} render passes, {} of {} groups took a region",
        after.pages_last_frame,
        after.render_passes_last_frame,
        after.batch_used - before.batch_used,
        after.batch_eligible - before.batch_eligible,
    );
    assert_eq!(
        after.batch_used - before.batch_used,
        220,
        "not every group took a region"
    );
    assert!(
        after.pages_last_frame > 0 && after.pages_last_frame < 8,
        "220 groups wanted {} pages",
        after.pages_last_frame
    );
    let paged_passes = after.render_passes_last_frame;

    // What the same frame took with a target per group.
    tuning::set_blend_pages_enabled(false);
    render(&mut backend, build());
    let separate_passes = render_stats().render_passes_last_frame;
    tuning::set_blend_pages_enabled(true);
    println!("  {paged_passes} render passes on pages, {separate_passes} on targets");
    assert!(
        paged_passes * 2 <= separate_passes,
        "220 groups took {paged_passes} render passes on pages against \
         {separate_passes} on targets of their own"
    );
}

/// Masks, alpha masks and nested blends keep the targets they had, and say so.
#[test]
fn groups_that_cannot_share_a_page_say_why() {
    let _exclusive = exclusive();
    let Some(descriptors) = descriptors() else {
        eprintln!("no GPU adapter available; skipping");
        return;
    };
    let mut backend = build_backend(descriptors, StageQuality::High);
    let bitmap = sprite(&mut backend);

    let at = |x: f64, y: f64| Matrix::translate(Twips::from_pixels(x), Twips::from_pixels(y));
    let build = || {
        let mut commands = CommandList::new();
        draw_sprite(
            &mut commands,
            &bitmap,
            Matrix::create_box(
                VIEWPORT.0 as f32 / SPRITE.0 as f32,
                VIEWPORT.1 as f32 / SPRITE.1 as f32,
                Twips::ZERO,
                Twips::ZERO,
            ),
            ColorTransform::default(),
        );

        // A stencil mask inside the group.
        let mut masked = CommandList::new();
        masked.push_mask();
        masked.draw_rect(
            Color::WHITE,
            Matrix::create_box(
                30.0,
                40.0,
                Twips::from_pixels(20.0),
                Twips::from_pixels(20.0),
            ),
        );
        masked.activate_mask();
        draw_sprite(
            &mut masked,
            &bitmap,
            at(14.0, 14.0),
            ColorTransform::default(),
        );
        draw_sprite(
            &mut masked,
            &bitmap,
            at(24.0, 24.0),
            ColorTransform::default(),
        );
        masked.deactivate_mask();
        masked.draw_rect(
            Color::WHITE,
            Matrix::create_box(
                30.0,
                40.0,
                Twips::from_pixels(20.0),
                Twips::from_pixels(20.0),
            ),
        );
        masked.pop_mask();
        commands.blend(masked, RenderBlendMode::Builtin(BlendMode::Multiply));

        // An alpha mask inside the group.
        let mut maskee = CommandList::new();
        draw_sprite(
            &mut maskee,
            &bitmap,
            at(110.0, 30.0),
            ColorTransform::default(),
        );
        draw_sprite(
            &mut maskee,
            &bitmap,
            at(120.0, 40.0),
            ColorTransform::default(),
        );
        let mut masker = CommandList::new();
        draw_sprite(
            &mut masker,
            &bitmap,
            at(116.0, 36.0),
            ColorTransform::default(),
        );
        let mut alpha_masked = CommandList::new();
        alpha_masked.render_alpha_mask(maskee, masker);
        draw_sprite(
            &mut alpha_masked,
            &bitmap,
            at(150.0, 60.0),
            ColorTransform::default(),
        );
        commands.blend(alpha_masked, RenderBlendMode::Builtin(BlendMode::Screen));

        // A blend inside a blend.
        let mut inner = CommandList::new();
        draw_sprite(
            &mut inner,
            &bitmap,
            at(240.0, 40.0),
            ColorTransform::default(),
        );
        draw_sprite(
            &mut inner,
            &bitmap,
            at(250.0, 50.0),
            ColorTransform::default(),
        );
        let mut outer = CommandList::new();
        outer.blend(inner, RenderBlendMode::Builtin(BlendMode::Darken));
        draw_sprite(
            &mut outer,
            &bitmap,
            at(270.0, 70.0),
            ColorTransform::default(),
        );
        commands.blend(outer, RenderBlendMode::Builtin(BlendMode::Layer));

        // And one that is simply too big for a region.
        let mut huge = CommandList::new();
        huge.draw_rect(
            Color::from_rgb(0x804020, 200),
            Matrix::create_box(
                620.0,
                380.0,
                Twips::from_pixels(10.0),
                Twips::from_pixels(10.0),
            ),
        );
        huge.draw_rect(
            Color::from_rgb(0x204080, 200),
            Matrix::create_box(
                620.0,
                380.0,
                Twips::from_pixels(12.0),
                Twips::from_pixels(12.0),
            ),
        );
        commands.blend(huge, RenderBlendMode::Builtin(BlendMode::Overlay));

        commands
    };

    let before = render_stats();
    assert_batching_is_invisible(&mut backend, "groups that keep their targets", build);
    let after = render_stats();

    let counted = |reason: PageFallback| {
        after.page_fallbacks[reason as usize] - before.page_fallbacks[reason as usize]
    };
    for reason in [
        PageFallback::Masked,
        PageFallback::AlphaMask,
        PageFallback::NestedBlend,
        PageFallback::Size,
    ] {
        assert!(
            counted(reason) > 0,
            "nothing was counted as {reason:?}, so that fallback is not being taken"
        );
    }
}

/// The one thing a page does change, measured.
///
/// A bitmap placed exactly half a pixel from a whole one puts every one of its
/// texel boundaries on a sample point. Which of the two texels a nearest
/// sampler picks there is arbitrary, and it is decided by the last bits of the
/// interpolated coordinate - which a page moves, because the group is drawn at
/// its region's place on the page rather than at the corner of a target of its
/// own.
///
/// So the tie can land the other way. This says how far that can go: a pixel
/// that differs holds what the pixel beside it holds, which is the neighbouring
/// texel. Nothing new appears, nothing moves further than one texel, and only
/// where the source itself changes colour from one texel to the next.
#[test]
fn a_texel_boundary_picks_a_neighbouring_texel_at_worst() {
    let _exclusive = exclusive();
    let Some(descriptors) = descriptors() else {
        eprintln!("no GPU adapter available; skipping");
        return;
    };
    let mut backend = build_backend(descriptors, StageQuality::Low);
    let bitmap = sprite(&mut backend);

    // Every group exactly half a pixel off the grid, which is the worst case.
    let build = || {
        let mut commands = CommandList::new();
        for i in 0..40 {
            let at = Matrix::translate(
                Twips::from_pixels((i % 8) as f64 * 74.0 + 6.5),
                Twips::from_pixels((i / 8) as f64 * 76.0 + 6.5),
            );
            let mut group = CommandList::new();
            draw_sprite(&mut group, &bitmap, at, ColorTransform::default());
            draw_sprite(
                &mut group,
                &bitmap,
                at * Matrix::translate(Twips::from_pixels(5.0), Twips::from_pixels(11.0)),
                ColorTransform::default(),
            );
            commands.blend(group, RenderBlendMode::Builtin(BlendMode::Layer));
        }
        commands
    };

    tuning::set_blend_pages_enabled(true);
    let paged = render(&mut backend, build());
    tuning::set_blend_pages_enabled(false);
    let separate = render(&mut backend, build());
    tuning::set_blend_pages_enabled(true);

    let (width, height) = paged.dimensions();
    let mut differing = 0usize;
    for (x, y, pixel) in paged.enumerate_pixels() {
        if pixel == separate.get_pixel(x, y) {
            continue;
        }
        differing += 1;
        let found_next_door = [
            (-1i32, 0i32),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (1, -1),
            (-1, 1),
            (1, 1),
        ]
        .into_iter()
        .filter_map(|(dx, dy)| {
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;
            (nx >= 0 && ny >= 0 && (nx as u32) < width && (ny as u32) < height)
                .then(|| separate.get_pixel(nx as u32, ny as u32))
        })
        .any(|neighbour| neighbour == pixel);
        assert!(
            found_next_door,
            "({x}, {y}) went from {:?} to {:?}, which is not what any neighbouring \
             texel holds - that is more than a boundary tie",
            separate.get_pixel(x, y).0,
            pixel.0
        );
    }
    println!(
        "40 groups on exact texel boundaries: {differing} of {} pixels took the other texel",
        width * height
    );
    // A seam along the boundaries inside the sprites, not a shifted image: the
    // groups cover about half the frame, and the lines inside them where a
    // texel meets a different one are a small fraction of that.
    assert!(
        differing * 20 < (width * height) as usize,
        "{differing} of {} pixels took the other texel, which is more than a seam",
        width * height
    );
}

/// Blended groups drawn while a stencil mask is up.
///
/// The group itself is a plain run of draws and takes a region, but the draw
/// that composites it back is inside the surface's masked chunk and has to be
/// stencil-tested like everything else there. A complex blend in the same place
/// composites in a pass of its own with the stencil attached, and its batch must
/// not reach across the mask's edges.
#[test]
fn groups_under_a_mask_survive_sharing_a_page() {
    let _exclusive = exclusive();
    let Some(descriptors) = descriptors() else {
        eprintln!("no GPU adapter available; skipping");
        return;
    };
    let mut backend = build_backend(descriptors, StageQuality::High);
    let bitmap = sprite(&mut backend);

    for mode in [
        BlendMode::Layer,
        BlendMode::Add,
        BlendMode::Multiply,
        BlendMode::Alpha,
    ] {
        assert_batching_is_invisible(&mut backend, &format!("{mode:?} under a mask"), || {
            let masker = || {
                Matrix::create_box(
                    260.0,
                    180.0,
                    Twips::from_pixels(60.0),
                    Twips::from_pixels(50.0),
                )
            };
            let mut inside = CommandList::new();
            draw_sprite(
                &mut inside,
                &bitmap,
                Matrix::create_box(
                    VIEWPORT.0 as f32 / SPRITE.0 as f32,
                    VIEWPORT.1 as f32 / SPRITE.1 as f32,
                    Twips::ZERO,
                    Twips::ZERO,
                ),
                ColorTransform::default(),
            );
            inside.push_mask();
            inside.draw_rect(Color::WHITE, masker());
            inside.activate_mask();
            // Groups that straddle the mask's edge, so what the stencil keeps
            // out is visible if it stops being kept out.
            for i in 0..6 {
                let at = Matrix::translate(
                    Twips::from_pixels(20.0 + i as f64 * 62.0),
                    Twips::from_pixels(30.0 + ((i % 2) as f64) * 130.0),
                );
                let mut group = CommandList::new();
                draw_sprite(&mut group, &bitmap, at, ColorTransform::default());
                draw_sprite(
                    &mut group,
                    &bitmap,
                    at * Matrix::translate(Twips::from_pixels(9.0), Twips::from_pixels(7.0)),
                    ColorTransform::default(),
                );
                group.render_bitmap(
                    bitmap.clone(),
                    Transform {
                        matrix: at
                            * Matrix::translate(Twips::from_pixels(-4.0), Twips::from_pixels(18.0)),
                        color_transform: ColorTransform::default(),
                        perspective_projection: None,
                    },
                    false,
                    PixelSnapping::Never,
                    PixelRegion::for_whole_size(SPRITE.0, SPRITE.1),
                );
                inside.blend(group, RenderBlendMode::Builtin(mode));
            }
            inside.deactivate_mask();
            inside.draw_rect(Color::WHITE, masker());
            inside.pop_mask();
            in_a_layer(inside)
        });
    }
}
