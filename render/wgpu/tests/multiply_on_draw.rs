//! That carrying a multiply on its own draw changes nothing about the picture.
//!
//! A group of one drawable blended with `Multiply` over a destination that is
//! known opaque used to be rendered through a target of its own and composited
//! by the multiply shader. It is now drawn straight onto the destination with a
//! blend state, which costs no target, no render pass and no composite. The
//! algebra is in [`TrivialBlend::MultiplyOpaque`]; these tests are what says it
//! is true of the renderer and not only of the algebra.
//!
//! Every test renders the same scene twice - once carried, once through the
//! shader - and compares the two frames. The tolerance is one level per
//! channel, and it is not slack: the shader quantises the group into an
//! eight-bit target before blending, so it rounds twice where the direct path
//! rounds once, and the two can differ by that last bit. Anything larger is a
//! real difference and fails.
//!
//! ```text
//! cargo test --release -p ruffle_render_wgpu --test multiply_on_draw -- --nocapture
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
use ruffle_render_wgpu::render_stats::render_stats;
use ruffle_render_wgpu::target::TextureTarget;
use ruffle_render_wgpu::tuning;
use ruffle_render_wgpu::wgpu;
use std::sync::{Arc, Mutex, MutexGuard};
use swf::{BlendMode, Color, ColorTransform, Fixed8, Twips};

const VIEWPORT: (u32, u32) = (640, 400);
const SPRITE: (u32, u32) = (40, 56);

/// The stage's own background, which is what makes the destination opaque.
const OPAQUE_STAGE: u32 = 0x30405a;

/// One GPU, one set of process-wide switches.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn exclusive() -> MutexGuard<'static, ()> {
    let guard = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    tuning::set_blend_pages_enabled(true);
    tuning::set_blend_batching_enabled(true);
    tuning::set_multiply_on_draw_enabled(true);
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

/// A sprite with a different colour in each quadrant, a transparent corner and
/// a half-transparent one, so that a blend which mishandles alpha shows.
fn sprite(backend: &mut WgpuRenderBackend<TextureTarget>) -> BitmapHandle {
    let mut pixels = Vec::with_capacity((SPRITE.0 * SPRITE.1 * 4) as usize);
    for y in 0..SPRITE.1 {
        for x in 0..SPRITE.0 {
            let left = x < SPRITE.0 / 2;
            let top = y < SPRITE.1 / 2;
            let (r, g, b, a) = match (left, top) {
                (true, true) => (230, 40, 40, 255),
                (false, true) => (40, 220, 60, 255),
                // Premultiplied at half alpha.
                (true, false) => (25, 30, 120, 128),
                (false, false) => (0, 0, 0, 0),
            };
            pixels.extend_from_slice(&[r, g, b, a]);
        }
    }
    backend
        .register_bitmap(Bitmap::new(SPRITE.0, SPRITE.1, BitmapFormat::Rgba, pixels))
        .expect("bitmap registration")
}

/// A position on the twip grid that is not exactly half a pixel from a whole
/// one. A bitmap whose origin sits on a texel boundary puts every boundary on a
/// sample point, and a nearest sampler breaks that tie arbitrarily - which the
/// three paths are entitled to do differently, since they place the group on
/// the grid in three different ways. Scenes that are about anything else step
/// around it.
fn off_the_half_pixel(pixels: f64) -> Twips {
    let mut twips = (pixels * 20.0).round() as i32;
    if twips.rem_euclid(20) == 10 {
        twips += 1;
    }
    Twips::new(twips)
}

fn draw_sprite(commands: &mut CommandList, bitmap: &BitmapHandle, matrix: Matrix) {
    draw_sprite_tinted(commands, bitmap, matrix, ColorTransform::default())
}

fn draw_sprite_tinted(
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

/// Something opaque underneath, then a row of groups of `draws_per_group`
/// each. One draw is what the direct path takes; more than one is what it must
/// refuse.
fn row_of_groups(
    bitmap: &BitmapHandle,
    mode: BlendMode,
    placement: Matrix,
    count: usize,
    draws_per_group: usize,
) -> CommandList {
    let mut commands = CommandList::new();
    draw_sprite(
        &mut commands,
        bitmap,
        Matrix::create_box(
            VIEWPORT.0 as f32 / SPRITE.0 as f32,
            VIEWPORT.1 as f32 / SPRITE.1 as f32,
            Twips::ZERO,
            Twips::ZERO,
        ),
    );

    for i in 0..count {
        let x = 18.0 + (i as f64) * 61.0;
        let y = 30.0 + ((i % 3) as f64) * 47.0;
        let place = Matrix::translate(Twips::from_pixels(x), Twips::from_pixels(y)) * placement;

        let mut group = CommandList::new();
        draw_sprite(&mut group, bitmap, place);
        for extra in 1..draws_per_group {
            draw_sprite_tinted(
                &mut group,
                bitmap,
                place
                    * Matrix::translate(
                        Twips::from_pixels(11.0 * extra as f64),
                        Twips::from_pixels(9.0 * extra as f64),
                    ),
                ColorTransform {
                    r_multiply: Fixed8::from_f32(0.6),
                    ..Default::default()
                },
            );
        }
        commands.blend(group, RenderBlendMode::Builtin(mode));
    }
    commands
}

fn in_a_layer(commands: CommandList) -> CommandList {
    let mut outer = CommandList::new();
    outer.blend(commands, RenderBlendMode::Builtin(BlendMode::Layer));
    outer
}

fn render(
    backend: &mut WgpuRenderBackend<TextureTarget>,
    commands: CommandList,
) -> image::RgbaImage {
    backend.submit_frame(Color::from_rgb(OPAQUE_STAGE, 255), commands, vec![]);
    backend.capture_frame().expect("capture must succeed")
}

/// Renders once carried and once through the shader, returning both frames and
/// how many multiplies the carried run actually carried.
fn both_ways(
    backend: &mut WgpuRenderBackend<TextureTarget>,
    build: impl Fn() -> CommandList,
) -> (image::RgbaImage, image::RgbaImage, u64) {
    tuning::set_multiply_on_draw_enabled(true);
    let before = render_stats();
    let carried = render(backend, build());
    let used = render_stats().multiply_on_draw_used - before.multiply_on_draw_used;

    tuning::set_multiply_on_draw_enabled(false);
    let through_shader = render(backend, build());
    tuning::set_multiply_on_draw_enabled(true);

    (carried, through_shader, used)
}

/// The largest per-channel difference between two frames, and how many pixels
/// differ at all.
fn difference(left: &image::RgbaImage, right: &image::RgbaImage) -> (i32, usize, String) {
    assert_eq!(left.dimensions(), right.dimensions());
    let mut worst = 0i32;
    let mut differing = 0usize;
    let mut where_worst = String::from("nowhere");
    for (x, y, pixel) in left.enumerate_pixels() {
        let other = right.get_pixel(x, y);
        if pixel != other {
            differing += 1;
            let delta = pixel
                .0
                .iter()
                .zip(other.0.iter())
                .map(|(a, b)| (*a as i32 - *b as i32).abs())
                .max()
                .unwrap_or(0);
            if delta > worst {
                worst = delta;
                where_worst = format!("({x},{y}) carried {:?} vs shader {:?}", pixel.0, other.0);
            }
        }
    }
    (worst, differing, where_worst)
}

/// One level per channel per multiply the pixel passes through, for the reason
/// in the module documentation. Anything more is a real difference.
fn assert_matches_the_shader(
    carried: &image::RgbaImage,
    through_shader: &image::RgbaImage,
    layers_deep: i32,
    what: &str,
) {
    let (worst, differing, where_worst) = difference(carried, through_shader);
    assert!(
        worst <= layers_deep,
        "{what}: carrying the multiply changed the picture by {worst} levels \
         across {differing} pixels, against a tolerance of {layers_deep}; \
         worst at {where_worst}"
    );
}

#[test]
fn a_multiply_over_an_opaque_stage_draws_what_the_shader_draws() {
    let _exclusive = exclusive();
    let Some(descriptors) = descriptors() else {
        eprintln!("no GPU adapter available; skipping");
        return;
    };
    for quality in [StageQuality::Low, StageQuality::High] {
        let mut backend = build_backend(descriptors.clone(), quality);
        let bitmap = sprite(&mut backend);
        for (name, placement) in placements() {
            let (carried, through_shader, used) = both_ways(&mut backend, || {
                row_of_groups(&bitmap, BlendMode::Multiply, placement, 7, 1)
            });
            assert_eq!(
                used, 7,
                "all seven multiplies should have been carried at {quality:?} quality under {name}"
            );
            let (worst, differing, where_worst) = difference(&carried, &through_shader);
            println!(
                "{quality:?} / {name}: {differing} pixels differ, worst {worst} level(s) at {where_worst}"
            );
            assert_matches_the_shader(
                &carried,
                &through_shader,
                1,
                &format!("{name} at {quality:?} quality"),
            );
        }
    }
}

#[test]
fn a_multiply_over_a_transparent_backdrop_keeps_its_shader() {
    let _exclusive = exclusive();
    let Some(descriptors) = descriptors() else {
        eprintln!("no GPU adapter available; skipping");
        return;
    };
    let mut backend = build_backend(descriptors, StageQuality::High);
    let bitmap = sprite(&mut backend);

    // A layer is a target of its own, cleared transparent, so the algebra does
    // not hold inside one and the shader has to run.
    let (carried, through_shader, used) = both_ways(&mut backend, || {
        in_a_layer(row_of_groups(
            &bitmap,
            BlendMode::Multiply,
            Matrix::IDENTITY,
            7,
            1,
        ))
    });
    assert_eq!(
        used, 0,
        "a multiply inside a layer is not over the opaque stage and must not be carried"
    );
    assert_matches_the_shader(&carried, &through_shader, 1, "multiply inside a layer");
}

#[test]
fn a_group_of_more_than_one_drawable_keeps_its_target() {
    let _exclusive = exclusive();
    let Some(descriptors) = descriptors() else {
        eprintln!("no GPU adapter available; skipping");
        return;
    };
    let mut backend = build_backend(descriptors, StageQuality::High);
    let bitmap = sprite(&mut backend);

    // Two overlapping children have to composite as a unit before the blend
    // applies, which is exactly what the target is for.
    let (carried, through_shader, used) = both_ways(&mut backend, || {
        row_of_groups(&bitmap, BlendMode::Multiply, Matrix::IDENTITY, 7, 2)
    });
    assert_eq!(
        used, 0,
        "a group of two drawables must composite before the blend applies"
    );
    assert_matches_the_shader(&carried, &through_shader, 1, "a group of two drawables");
}

#[test]
fn an_erase_stops_the_multiplies_that_follow_it_being_carried() {
    let _exclusive = exclusive();
    let Some(descriptors) = descriptors() else {
        eprintln!("no GPU adapter available; skipping");
        return;
    };
    let mut backend = build_backend(descriptors, StageQuality::High);
    let bitmap = sprite(&mut backend);

    // `Erase` writes alpha back out of the destination, so what was opaque is
    // not opaque afterwards and the multiplies after it must go back through
    // the shader.
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
        );

        let mut carried_first = CommandList::new();
        draw_sprite(
            &mut carried_first,
            &bitmap,
            Matrix::translate(Twips::from_pixels(20.0), Twips::from_pixels(20.0)),
        );
        commands.blend(carried_first, RenderBlendMode::Builtin(BlendMode::Multiply));

        let mut erase = CommandList::new();
        draw_sprite(
            &mut erase,
            &bitmap,
            Matrix::translate(Twips::from_pixels(120.0), Twips::from_pixels(20.0)),
        );
        commands.blend(erase, RenderBlendMode::Builtin(BlendMode::Erase));

        for i in 0..3 {
            let mut after = CommandList::new();
            draw_sprite(
                &mut after,
                &bitmap,
                Matrix::translate(
                    Twips::from_pixels(220.0 + i as f64 * 61.0),
                    Twips::from_pixels(20.0),
                ),
            );
            commands.blend(after, RenderBlendMode::Builtin(BlendMode::Multiply));
        }
        commands
    };

    let (carried, through_shader, used) = both_ways(&mut backend, build);
    assert_eq!(
        used, 1,
        "only the multiply before the erase is over a destination known opaque"
    );
    assert_matches_the_shader(&carried, &through_shader, 1, "multiplies around an erase");
}

/// What it is worth: a room-shaped scene of single-bitmap multiplies, which is
/// the shape AQW actually produces - over a 40-minute session of the game, only
/// 128 blended groups in 9.2 million held more than one drawable.
#[test]
fn a_crowd_of_multiplies_costs_no_passes_of_its_own() {
    let _exclusive = exclusive();
    let Some(descriptors) = descriptors() else {
        eprintln!("no GPU adapter available; skipping");
        return;
    };
    let mut backend = build_backend(descriptors, StageQuality::Low);
    let bitmap = sprite(&mut backend);

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
        );
        for i in 0..220 {
            let x = ((i as f64 * 0.7548776662) % 1.0) * (VIEWPORT.0 - SPRITE.0) as f64;
            let y = ((i as f64 * 0.5698402909) % 1.0) * (VIEWPORT.1 - SPRITE.1) as f64;
            let mut group = CommandList::new();
            draw_sprite(
                &mut group,
                &bitmap,
                Matrix::translate(off_the_half_pixel(x), off_the_half_pixel(y)),
            );
            commands.blend(group, RenderBlendMode::Builtin(BlendMode::Multiply));
        }
        commands
    };

    tuning::set_multiply_on_draw_enabled(true);
    let before = render_stats();
    let carried_frame = render(&mut backend, build());
    let after = render_stats();
    let carried_passes = after.render_passes_last_frame;
    let carried_copies = after.destination_copies - before.destination_copies;
    let used = after.multiply_on_draw_used - before.multiply_on_draw_used;

    tuning::set_multiply_on_draw_enabled(false);
    let before = render_stats();
    let shader_frame = render(&mut backend, build());
    let after = render_stats();
    let shader_passes = after.render_passes_last_frame;
    let shader_copies = after.destination_copies - before.destination_copies;
    tuning::set_multiply_on_draw_enabled(true);

    println!(
        "220 multiplies over an opaque stage: {carried_passes} render passes and \
         {carried_copies} destination copies carried, against {shader_passes} and \
         {shader_copies} through the shader ({used} carried)"
    );

    assert_eq!(
        used, 220,
        "every group is a single bitmap over the opaque stage"
    );
    assert_eq!(
        carried_copies, 0,
        "a carried multiply never reads the destination back"
    );
    assert!(
        carried_passes * 4 <= shader_passes,
        "carrying the multiplies took {carried_passes} render passes against \
         {shader_passes} through the shader"
    );
    // Two, not one: this scene is about twice overdrawn, so a pixel can pass
    // through two multiplies and carry the last bit of each.
    assert_matches_the_shader(&carried_frame, &shader_frame, 2, "a crowd of multiplies");
}
