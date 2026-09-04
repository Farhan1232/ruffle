//! That a cache texture bigger than the picture inside it changes nothing.
//!
//! A `cacheAsBitmap` texture is now rounded up to a capacity, so that an object
//! whose bounds breathe by a pixel or two between frames keeps the texture it
//! has instead of allocating a new one every frame. That means the texture is
//! usually a little bigger than the picture, and the whole safety argument is
//! that **nothing takes its extent from the texture**: the surface that redraws
//! the cache is built at the picture's size, its passes are held to the
//! picture's rectangle by a viewport and scissor, the filter chain is handed the
//! picture's rectangle rather than the whole texture, and the composite samples
//! the picture's rectangle through a `PixelRegion`.
//!
//! An earlier padding experiment did not do that, and `displacement_map` - which
//! samples by coordinate rather than by neighbourhood - read the padding. So
//! every test here renders the same content twice, once with the rounding on and
//! once with it off, and compares the two frames **pixel for pixel**. A padded
//! texture that is even one channel different from an exactly-sized one fails.
//!
//! ```text
//! cargo test --release -p ruffle_render_wgpu --test cache_capacity -- --nocapture
//! ```

use ruffle_render::backend::{BitmapCacheEntry, RenderBackend, ViewportDimensions};
use ruffle_render::bitmap::{Bitmap, BitmapFormat, BitmapHandle, PixelRegion, PixelSnapping};
use ruffle_render::cache_capacity::{
    capacity_fits, capacity_for, set_capacity_reuse_enabled, set_granularity,
};
use ruffle_render::commands::{CommandHandler, CommandList};
use ruffle_render::filters::{DisplacementMapFilter, DisplacementMapFilterMode, Filter};
use ruffle_render::matrix::Matrix;
use ruffle_render::quality::StageQuality;
use ruffle_render::transform::Transform;
use ruffle_render_wgpu::backend::{
    WgpuRenderBackend, create_wgpu_instance, request_adapter_and_device,
};
use ruffle_render_wgpu::descriptors::Descriptors;
use ruffle_render_wgpu::target::TextureTarget;
use ruffle_render_wgpu::wgpu;
use std::num::NonZeroU32;
use std::sync::{Arc, Mutex, MutexGuard};
use swf::{
    BevelFilter, BevelFilterFlags, BlurFilter, BlurFilterFlags, Color, ColorMatrixFilter,
    ConvolutionFilter, DropShadowFilter, DropShadowFilterFlags, Fixed8, Fixed16, GlowFilter,
    GlowFilterFlags, GradientFilter, GradientFilterFlags, GradientRecord, Twips,
};

const VIEWPORT: (u32, u32) = (420, 300);

static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn exclusive() -> MutexGuard<'static, ()> {
    let guard = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    set_capacity_reuse_enabled(true);
    set_granularity(ruffle_render::cache_capacity::DEFAULT_GRANULARITY);
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
/// so anything that flips, offsets or bleeds it shows up.
fn sprite(backend: &mut WgpuRenderBackend<TextureTarget>, size: (u32, u32)) -> BitmapHandle {
    let mut pixels = Vec::with_capacity((size.0 * size.1 * 4) as usize);
    for y in 0..size.1 {
        for x in 0..size.0 {
            let left = x < size.0 / 2;
            let top = y < size.1 / 2;
            let (r, g, b, a) = match (left, top) {
                (true, true) => (230, 40, 40, 255),
                (false, true) => (40, 220, 60, 255),
                (true, false) => (50, 60, 240, 255),
                (false, false) => (0, 0, 0, 0),
            };
            pixels.extend_from_slice(&[r, g, b, a]);
        }
    }
    backend
        .register_bitmap(Bitmap::new(size.0, size.1, BitmapFormat::Rgba, pixels))
        .expect("bitmap registration")
}

/// A displacement map: a smooth ramp, so every pixel of the source is asked for
/// from somewhere slightly different. This is the filter the old padding design
/// broke, because it samples by coordinate rather than by neighbourhood.
fn displacement_map(backend: &mut WgpuRenderBackend<TextureTarget>) -> BitmapHandle {
    let (w, h) = (64u32, 64u32);
    let mut pixels = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            pixels.extend_from_slice(&[
                (x * 255 / w.max(1)) as u8,
                (y * 255 / h.max(1)) as u8,
                0,
                255,
            ]);
        }
    }
    backend
        .register_bitmap(Bitmap::new(w, h, BitmapFormat::Rgba, pixels))
        .expect("displacement map")
}

fn gradient() -> Vec<GradientRecord> {
    vec![
        GradientRecord {
            ratio: 0,
            color: Color::from_rgba(0x80_00_00_ff),
        },
        GradientRecord {
            ratio: 255,
            color: Color::from_rgba(0xff_ff_ff_00),
        },
    ]
}

/// Every filter the brief names, each on the same content.
fn every_filter(map: &BitmapHandle) -> Vec<(&'static str, Vec<Filter>)> {
    vec![
        ("none", vec![]),
        (
            "BlurFilter",
            vec![Filter::BlurFilter(BlurFilter {
                blur_x: Fixed16::from_f32(6.0),
                blur_y: Fixed16::from_f32(6.0),
                flags: BlurFilterFlags::from_passes(2),
            })],
        ),
        (
            "GlowFilter",
            vec![Filter::GlowFilter(GlowFilter {
                color: Color::WHITE,
                blur_x: Fixed16::from_f32(5.0),
                blur_y: Fixed16::from_f32(5.0),
                strength: Fixed8::ONE,
                flags: GlowFilterFlags::from_passes(1),
            })],
        ),
        (
            "DropShadowFilter",
            vec![Filter::DropShadowFilter(DropShadowFilter {
                color: Color::BLACK,
                blur_x: Fixed16::from_f32(5.0),
                blur_y: Fixed16::from_f32(5.0),
                angle: Fixed16::from_f32(45.0),
                distance: Fixed16::from_f32(6.0),
                strength: Fixed8::ONE,
                flags: DropShadowFilterFlags::from_passes(1),
            })],
        ),
        (
            "BevelFilter",
            vec![Filter::BevelFilter(BevelFilter {
                shadow_color: Color::BLACK,
                highlight_color: Color::WHITE,
                blur_x: Fixed16::from_f32(4.0),
                blur_y: Fixed16::from_f32(4.0),
                angle: Fixed16::from_f32(45.0),
                distance: Fixed16::from_f32(4.0),
                strength: Fixed8::ONE,
                flags: BevelFilterFlags::from_passes(1),
            })],
        ),
        (
            "GradientGlowFilter",
            vec![Filter::GradientGlowFilter(GradientFilter {
                colors: gradient(),
                blur_x: Fixed16::from_f32(5.0),
                blur_y: Fixed16::from_f32(5.0),
                angle: Fixed16::from_f32(45.0),
                distance: Fixed16::from_f32(4.0),
                strength: Fixed8::ONE,
                flags: GradientFilterFlags::from_passes(1),
            })],
        ),
        (
            "GradientBevelFilter",
            vec![Filter::GradientBevelFilter(GradientFilter {
                colors: gradient(),
                blur_x: Fixed16::from_f32(5.0),
                blur_y: Fixed16::from_f32(5.0),
                angle: Fixed16::from_f32(45.0),
                distance: Fixed16::from_f32(4.0),
                strength: Fixed8::ONE,
                flags: GradientFilterFlags::from_passes(1),
            })],
        ),
        (
            "ColorMatrixFilter",
            // An alpha offset, so every pixel of the picture becomes opaque -
            // including the ones the object did not draw on. This is the filter
            // that makes the *size* of the cache visible, and the one that
            // proves the spare capacity is not part of the picture.
            vec![Filter::ColorMatrixFilter(ColorMatrixFilter {
                matrix: [
                    1.0, 0.0, 0.0, 0.0, 0.0, //
                    0.0, 1.0, 0.0, 0.0, 0.0, //
                    0.0, 0.0, 1.0, 0.0, 0.0, //
                    0.0, 0.0, 0.0, 0.5, 0.4,
                ],
            })],
        ),
        (
            "ConvolutionFilter",
            vec![Filter::ConvolutionFilter(ConvolutionFilter {
                num_matrix_cols: 3,
                num_matrix_rows: 3,
                matrix: vec![0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0],
                divisor: 6.0,
                bias: 0.0,
                default_color: Color::from_rgba(0),
                flags: swf::ConvolutionFilterFlags::empty(),
            })],
        ),
        (
            "DisplacementMapFilter",
            vec![Filter::DisplacementMapFilter(DisplacementMapFilter {
                color: Color::from_rgba(0),
                component_x: 1,
                component_y: 2,
                map_bitmap: Some(map.clone()),
                map_point: (0, 0),
                mode: DisplacementMapFilterMode::Clamp,
                scale_x: 12.0,
                scale_y: 12.0,
                viewscale_x: 1.0,
                viewscale_y: 1.0,
            })],
        ),
        (
            "displacement_map_through_filters",
            vec![
                Filter::BlurFilter(BlurFilter {
                    blur_x: Fixed16::from_f32(4.0),
                    blur_y: Fixed16::from_f32(4.0),
                    flags: BlurFilterFlags::from_passes(1),
                }),
                Filter::DisplacementMapFilter(DisplacementMapFilter {
                    color: Color::from_rgba(0),
                    component_x: 1,
                    component_y: 2,
                    map_bitmap: Some(map.clone()),
                    map_point: (0, 0),
                    mode: DisplacementMapFilterMode::Clamp,
                    scale_x: 10.0,
                    scale_y: 10.0,
                    viewscale_x: 1.0,
                    viewscale_y: 1.0,
                }),
                Filter::GlowFilter(GlowFilter {
                    color: Color::from_rgb(0x00aaff, 255),
                    blur_x: Fixed16::from_f32(4.0),
                    blur_y: Fixed16::from_f32(4.0),
                    strength: Fixed8::ONE,
                    flags: GlowFilterFlags::from_passes(1),
                }),
            ],
        ),
    ]
}

/// What a cached display object looks like to the renderer: a texture, the
/// picture inside it, and the commands that draw the picture.
struct Cached {
    handle: BitmapHandle,
    physical: (u32, u32),
}

impl Cached {
    /// Allocates a cache texture for a picture of `logical` size, under
    /// whatever the capacity policy currently is.
    fn new(backend: &mut WgpuRenderBackend<TextureTarget>, logical: (u32, u32)) -> Self {
        let physical = capacity_for(logical.0, logical.1);
        let handle = backend
            .create_empty_texture(
                NonZeroU32::new(physical.0).expect("non-zero"),
                NonZeroU32::new(physical.1).expect("non-zero"),
            )
            .expect("cache texture");
        Self { handle, physical }
    }
}

/// Draws `sprite` into a cache of `logical` size, applies `filters`, composites
/// the result onto the stage, and gives back the frame.
///
/// This is exactly the shape `render_base` gives the renderer: the cache entry
/// carries the picture's size, and the composite samples the picture's own
/// region of whatever texture is behind it.
fn render_cached(
    backend: &mut WgpuRenderBackend<TextureTarget>,
    cached: &Cached,
    sprite: &BitmapHandle,
    sprite_size: (u32, u32),
    logical: (u32, u32),
    filters: Vec<Filter>,
    at: (f64, f64),
    masked: bool,
) -> image::RgbaImage {
    let mut inner = CommandList::new();
    if masked {
        // A stencil mask inside the cache. This is the case that needs the
        // stencil attachment to be the size of the texture rather than of the
        // picture: every attachment in a render pass must be the same size.
        inner.push_mask();
        inner.render_bitmap(
            sprite.clone(),
            Transform {
                matrix: Matrix::translate(Twips::from_pixels(2.0), Twips::from_pixels(2.0))
                    * Matrix::scale(0.7, 0.7),
                color_transform: Default::default(),
                perspective_projection: None,
            },
            false,
            PixelSnapping::Never,
            PixelRegion::for_whole_size(sprite_size.0, sprite_size.1),
        );
        inner.activate_mask();
    }
    inner.render_bitmap(
        sprite.clone(),
        Transform {
            matrix: Matrix::translate(Twips::from_pixels(6.0), Twips::from_pixels(5.0)),
            color_transform: Default::default(),
            perspective_projection: None,
        },
        false,
        PixelSnapping::Never,
        PixelRegion::for_whole_size(sprite_size.0, sprite_size.1),
    );
    if masked {
        inner.deactivate_mask();
        inner.render_bitmap(
            sprite.clone(),
            Transform {
                matrix: Matrix::translate(Twips::from_pixels(2.0), Twips::from_pixels(2.0))
                    * Matrix::scale(0.7, 0.7),
                color_transform: Default::default(),
                perspective_projection: None,
            },
            false,
            PixelSnapping::Never,
            PixelRegion::for_whole_size(sprite_size.0, sprite_size.1),
        );
        inner.pop_mask();
    }

    let entry = BitmapCacheEntry {
        handle: cached.handle.clone(),
        commands: inner,
        clear: Color::from_rgba(0),
        filters,
        logical_width: logical.0,
        logical_height: logical.1,
    };

    let mut stage = CommandList::new();
    stage.render_bitmap(
        cached.handle.clone(),
        Transform {
            matrix: Matrix::translate(Twips::from_pixels(at.0), Twips::from_pixels(at.1)),
            color_transform: Default::default(),
            perspective_projection: None,
        },
        false,
        PixelSnapping::Always,
        // `BitmapInfo::full_region` of the picture, which is what core builds.
        PixelRegion::for_whole_size(logical.0, logical.1),
    );

    backend.submit_frame(Color::from_rgb(0x202030, 255), stage, vec![entry]);
    backend.capture_frame().expect("capture must succeed")
}

fn assert_same(padded: &image::RgbaImage, exact: &image::RgbaImage, what: &str) {
    assert_eq!(padded.dimensions(), exact.dimensions(), "{what}");
    let mut differing = 0usize;
    let mut worst = (0u32, 0u32, [0u8; 4], [0u8; 4], 0i32);
    for (x, y, padded_pixel) in padded.enumerate_pixels() {
        let exact_pixel = exact.get_pixel(x, y);
        if padded_pixel != exact_pixel {
            differing += 1;
            let delta = padded_pixel
                .0
                .iter()
                .zip(exact_pixel.0.iter())
                .map(|(a, b)| (*a as i32 - *b as i32).abs())
                .max()
                .unwrap_or(0);
            if delta > worst.4 {
                worst = (x, y, padded_pixel.0, exact_pixel.0, delta);
            }
        }
    }
    assert_eq!(
        differing, 0,
        "{what}: {differing} pixels differ between a padded cache texture and an \
         exactly sized one; worst at ({}, {}): padded {:?} vs exact {:?}",
        worst.0, worst.1, worst.2, worst.3
    );
}

/// The headline: for every filter, at several picture sizes, a cache texture
/// rounded up to a capacity draws exactly what an exactly-sized one draws.
#[test]
fn a_padded_cache_draws_the_same_picture() {
    let _exclusive = exclusive();
    let Some(descriptors) = descriptors() else {
        eprintln!("no GPU adapter available; skipping");
        return;
    };

    for quality in [StageQuality::Low, StageQuality::High] {
        let mut backend = build_backend(descriptors.clone(), quality);
        let sprite_size = (48u32, 40u32);
        let sprite = sprite(&mut backend, sprite_size);
        let map = displacement_map(&mut backend);

        // Sizes that round to very different capacities: one just over a
        // boundary, one just under, one exactly on it.
        for logical in [(61u32, 53u32), (64, 64), (65, 33), (147, 100)] {
            for (name, filters) in every_filter(&map) {
                let what = format!("{name} at {logical:?}, quality {quality:?}");

                set_capacity_reuse_enabled(true);
                let padded_cache = Cached::new(&mut backend, logical);
                assert!(
                    padded_cache.physical.0 >= logical.0 && padded_cache.physical.1 >= logical.1,
                    "{what}: capacity {:?} does not hold {logical:?}",
                    padded_cache.physical
                );
                let padded = render_cached(
                    &mut backend,
                    &padded_cache,
                    &sprite,
                    sprite_size,
                    logical,
                    filters.clone(),
                    (40.0, 30.0),
                    false,
                );

                set_capacity_reuse_enabled(false);
                let exact_cache = Cached::new(&mut backend, logical);
                assert_eq!(
                    exact_cache.physical, logical,
                    "the switch must be a true A/B"
                );
                let exact = render_cached(
                    &mut backend,
                    &exact_cache,
                    &sprite,
                    sprite_size,
                    logical,
                    filters,
                    (40.0, 30.0),
                    false,
                );

                set_capacity_reuse_enabled(true);
                assert_same(&padded, &exact, &what);
            }
        }
    }
}

/// The stale-pixel case. A texture that held a big bright picture is kept for a
/// smaller one; none of the old picture may survive, anywhere.
#[test]
fn a_kept_texture_does_not_show_the_picture_before_it() {
    let _exclusive = exclusive();
    let Some(descriptors) = descriptors() else {
        eprintln!("no GPU adapter available; skipping");
        return;
    };
    let mut backend = build_backend(descriptors, StageQuality::High);
    let big_sprite_size = (150u32, 150u32);
    let big = sprite(&mut backend, big_sprite_size);
    let map = displacement_map(&mut backend);

    let large = (160u32, 150u32);
    let small = (150u32, 141u32);

    for (masked, (name, filters)) in [false, true]
        .into_iter()
        .flat_map(|masked| every_filter(&map).into_iter().map(move |f| (masked, f)))
    {
        set_capacity_reuse_enabled(true);
        // One texture, used for the large picture and then kept for the small
        // one - which is the whole point of the capacity.
        let kept = Cached::new(&mut backend, large);
        assert!(
            capacity_fits(kept.physical, small),
            "the policy would not have kept this texture, so the test proves nothing"
        );
        render_cached(
            &mut backend,
            &kept,
            &big,
            big_sprite_size,
            large,
            filters.clone(),
            (30.0, 20.0),
            masked,
        );
        let reused = render_cached(
            &mut backend,
            &kept,
            &big,
            big_sprite_size,
            small,
            filters.clone(),
            (30.0, 20.0),
            masked,
        );

        // The same small picture drawn into a texture that has never held
        // anything else.
        let fresh = Cached::new(&mut backend, small);
        let clean = render_cached(
            &mut backend,
            &fresh,
            &big,
            big_sprite_size,
            small,
            filters,
            (30.0, 20.0),
            masked,
        );

        assert_same(
            &reused,
            &clean,
            &format!("{name} after a larger picture (masked: {masked})"),
        );
    }
}

/// And growing back: a texture kept across a shrink and then asked for the
/// larger size again must be rebuilt rather than silently clipping.
#[test]
fn growing_past_the_capacity_is_never_clipped() {
    let _exclusive = exclusive();
    let Some(descriptors) = descriptors() else {
        eprintln!("no GPU adapter available; skipping");
        return;
    };
    let mut backend = build_backend(descriptors, StageQuality::High);
    let sprite_size = (120u32, 96u32);
    let sprite = sprite(&mut backend, sprite_size);

    set_capacity_reuse_enabled(true);
    let masked = false;
    let small = (100u32, 80u32);
    let grown = (140u32, 130u32);

    let cache = Cached::new(&mut backend, small);
    assert!(
        !capacity_fits(cache.physical, grown),
        "a {:?} texture must not be kept for a {grown:?} picture",
        cache.physical
    );

    // What the policy makes core do: allocate again, at the new capacity.
    let regrown = Cached::new(&mut backend, grown);
    let padded = render_cached(
        &mut backend,
        &regrown,
        &sprite,
        sprite_size,
        grown,
        vec![],
        (30.0, 20.0),
        masked,
    );

    set_capacity_reuse_enabled(false);
    let exact = Cached::new(&mut backend, grown);
    let plain = render_cached(
        &mut backend,
        &exact,
        &sprite,
        sprite_size,
        grown,
        vec![],
        (30.0, 20.0),
        masked,
    );
    set_capacity_reuse_enabled(true);

    assert_same(&padded, &plain, "a cache regrown past its capacity");
}

/// Awkward placements: the composite is at a fractional offset, so the padding
/// would show up as a shifted or resampled image if it were part of the picture.
#[test]
fn a_padded_cache_survives_awkward_placement() {
    let _exclusive = exclusive();
    let Some(descriptors) = descriptors() else {
        eprintln!("no GPU adapter available; skipping");
        return;
    };
    let mut backend = build_backend(descriptors, StageQuality::High);
    let sprite_size = (48u32, 40u32);
    let sprite = sprite(&mut backend, sprite_size);

    let logical = (73u32, 59u32);
    let masked = false;
    for at in [
        (0.0, 0.0),
        (40.5, 30.5),
        (-12.0, -9.0),
        (0.37, 0.62),
        (VIEWPORT.0 as f64 - 40.0, VIEWPORT.1 as f64 - 30.0),
    ] {
        set_capacity_reuse_enabled(true);
        let padded_cache = Cached::new(&mut backend, logical);
        let padded = render_cached(
            &mut backend,
            &padded_cache,
            &sprite,
            sprite_size,
            logical,
            vec![Filter::GlowFilter(GlowFilter {
                color: Color::WHITE,
                blur_x: Fixed16::from_f32(4.0),
                blur_y: Fixed16::from_f32(4.0),
                strength: Fixed8::ONE,
                flags: GlowFilterFlags::from_passes(1),
            })],
            at,
            masked,
        );

        set_capacity_reuse_enabled(false);
        let exact_cache = Cached::new(&mut backend, logical);
        let exact = render_cached(
            &mut backend,
            &exact_cache,
            &sprite,
            sprite_size,
            logical,
            vec![Filter::GlowFilter(GlowFilter {
                color: Color::WHITE,
                blur_x: Fixed16::from_f32(4.0),
                blur_y: Fixed16::from_f32(4.0),
                strength: Fixed8::ONE,
                flags: GlowFilterFlags::from_passes(1),
            })],
            at,
            masked,
        );
        set_capacity_reuse_enabled(true);

        assert_same(&padded, &exact, &format!("placed at {at:?}"));
    }
}

/// A stencil mask inside a cached object, on a texture bigger than the picture.
///
/// This is the case that made the stencil attachment's size matter: a render
/// pass requires every attachment to be the same size, so the stencil buffer has
/// to follow the texture rather than the picture, while the drawing still has to
/// be held to the picture's rectangle.
#[test]
fn a_padded_cache_with_a_mask_inside_it() {
    let _exclusive = exclusive();
    let Some(descriptors) = descriptors() else {
        eprintln!("no GPU adapter available; skipping");
        return;
    };

    for quality in [StageQuality::Low, StageQuality::High] {
        let mut backend = build_backend(descriptors.clone(), quality);
        let sprite_size = (48u32, 40u32);
        let sprite = sprite(&mut backend, sprite_size);
        let map = displacement_map(&mut backend);

        for logical in [(61u32, 53u32), (65, 33), (147, 100)] {
            for (name, filters) in every_filter(&map) {
                let what = format!("masked {name} at {logical:?}, quality {quality:?}");

                set_capacity_reuse_enabled(true);
                let padded_cache = Cached::new(&mut backend, logical);
                let padded = render_cached(
                    &mut backend,
                    &padded_cache,
                    &sprite,
                    sprite_size,
                    logical,
                    filters.clone(),
                    (40.0, 30.0),
                    true,
                );

                set_capacity_reuse_enabled(false);
                let exact_cache = Cached::new(&mut backend, logical);
                let exact = render_cached(
                    &mut backend,
                    &exact_cache,
                    &sprite,
                    sprite_size,
                    logical,
                    filters,
                    (40.0, 30.0),
                    true,
                );

                set_capacity_reuse_enabled(true);
                assert_same(&padded, &exact, &what);
            }
        }
    }
}
