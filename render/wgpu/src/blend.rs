use enum_map::Enum;

use ruffle_render::{commands::RenderBlendMode, pixel_bender::PixelBenderShaderHandle};
use swf::BlendMode;

#[derive(Enum, Debug, Copy, Clone)]
pub enum ComplexBlend {
    Multiply,   // Can't be trivial, 0 alpha is special case
    Lighten,    // Might be trivial but I can't reproduce the right colors
    Darken,     // Might be trivial but I can't reproduce the right colors
    Difference, // Can't be trivial, relies on abs operation
    Invert,     // May be trivial using a constant? Hard because it's without premultiplied alpha
    Alpha,      // Can't be trivial, requires layer tracking
    Erase,      // Can't be trivial, requires layer tracking
    Overlay,    // Can't be trivial, big math expression
    HardLight,  // Can't be trivial, big math expression
}

#[derive(Debug, Clone)]
pub enum BlendType {
    /// Trivial blends can be expressed with just a "draw bitmap" with blend states
    Trivial(TrivialBlend),

    /// Complex blends require a shader to express, so they are separated out into their own render
    Complex(ComplexBlend),

    /// Invoke a custom `PixelBender` shader.
    Shader(PixelBenderShaderHandle),
}

impl BlendType {
    pub fn from(mode: RenderBlendMode) -> BlendType {
        match mode {
            RenderBlendMode::Builtin(BlendMode::Normal) => BlendType::Trivial(TrivialBlend::Normal),
            RenderBlendMode::Builtin(BlendMode::Layer) => BlendType::Trivial(TrivialBlend::Normal),
            RenderBlendMode::Builtin(BlendMode::Multiply) => {
                BlendType::Complex(ComplexBlend::Multiply)
            }
            RenderBlendMode::Builtin(BlendMode::Screen) => BlendType::Trivial(TrivialBlend::Screen),
            RenderBlendMode::Builtin(BlendMode::Lighten) => {
                BlendType::Complex(ComplexBlend::Lighten)
            }
            RenderBlendMode::Builtin(BlendMode::Darken) => BlendType::Complex(ComplexBlend::Darken),
            RenderBlendMode::Builtin(BlendMode::Difference) => {
                BlendType::Complex(ComplexBlend::Difference)
            }
            RenderBlendMode::Builtin(BlendMode::Add) => BlendType::Trivial(TrivialBlend::Add),
            RenderBlendMode::Builtin(BlendMode::Subtract) => {
                BlendType::Trivial(TrivialBlend::Subtract)
            }
            RenderBlendMode::Builtin(BlendMode::Invert) => BlendType::Complex(ComplexBlend::Invert),
            RenderBlendMode::Builtin(BlendMode::Alpha) => BlendType::Complex(ComplexBlend::Alpha),
            RenderBlendMode::Builtin(BlendMode::Erase) => BlendType::Complex(ComplexBlend::Erase),
            RenderBlendMode::Builtin(BlendMode::Overlay) => {
                BlendType::Complex(ComplexBlend::Overlay)
            }
            RenderBlendMode::Builtin(BlendMode::HardLight) => {
                BlendType::Complex(ComplexBlend::HardLight)
            }
            RenderBlendMode::Shader(shader) => BlendType::Shader(shader),
        }
    }

    pub fn default_color(&self) -> wgpu::Color {
        wgpu::Color::TRANSPARENT
    }
}

#[derive(Enum, Debug, Copy, Clone)]
pub enum TrivialBlend {
    Normal,
    Add,
    Subtract,
    Screen,
    /// Flash multiply, for the case where the destination is known to be fully
    /// opaque.
    ///
    /// Multiply is a complex blend because it reads the destination in a
    /// shader, and in general it cannot be written as a blend state: the
    /// premultiplied result is
    ///
    /// ```text
    ///   src*(1 - dst.a) + dst*(1 - src.a) + src*dst
    /// ```
    ///
    /// and no pair of blend factors is that sum. When `dst.a` is 1 the first
    /// term vanishes and what is left is exactly a blend state:
    ///
    /// ```text
    ///   src*dst + dst*(1 - src.a)  =  Dst * src  +  OneMinusSrcAlpha * dst
    /// ```
    ///
    /// The same factors on the alpha channel give `src.a*dst.a +
    /// dst.a*(1 - src.a)`, which is `dst.a`, so an opaque destination stays
    /// opaque and the precondition still holds for whatever is drawn next.
    ///
    /// The shader's two special cases agree. `dst.a == 0` cannot arise here.
    /// `src.a == 0` makes the shader discard, and this writes
    /// `Dst*0 + 1*dst`, which is the destination unchanged - the same thing,
    /// since a premultiplied source with `src.a == 0` has `src.rgb == 0`.
    ///
    /// Only ever produced by the direct fast path, which checks the
    /// destination's opacity. Never by [`BlendType::from`], because a multiply
    /// in general is still complex.
    MultiplyOpaque,
}

impl TrivialBlend {
    pub fn blend_state(self) -> wgpu::BlendState {
        // out = <src_factor> * src <operation> <dst_factor> * dst
        match self {
            TrivialBlend::Normal => wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING,
            TrivialBlend::Add => wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent::OVER,
            },
            TrivialBlend::Screen => wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::OneMinusSrc,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent::OVER,
            },
            TrivialBlend::Subtract => wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::ReverseSubtract,
                },
                alpha: wgpu::BlendComponent::OVER,
            },
            // Colour and alpha take the same factors; for alpha that is
            // `dst.a*src.a + (1 - src.a)*dst.a`, which is `dst.a`. See the
            // variant's own documentation for why this equals the shader.
            TrivialBlend::MultiplyOpaque => {
                let component = wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::Dst,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                };
                wgpu::BlendState {
                    color: component,
                    alpha: component,
                }
            }
        }
    }
}

#[cfg(test)]
mod multiply_opaque {
    //! That the blend state is the multiply shader, for an opaque destination.
    //!
    //! The renderer's own test (`tests/multiply_on_draw.rs`) says the two draw
    //! the same picture. This says why, which is the part that has to hold for
    //! every input rather than for the ones a test scene happens to contain.

    /// `shaders/blend/multiply.wgsl`, for the branch this path replaces:
    /// a source with alpha over a destination with alpha, both premultiplied.
    fn shader(src: [f64; 4], dst: [f64; 4]) -> [f64; 4] {
        let (sa, da) = (src[3], dst[3]);
        let mut out = [0.0; 4];
        for c in 0..3 {
            // The shader un-premultiplies both, multiplies, and premultiplies
            // the result back.
            let blended = (src[c] / sa) * (dst[c] / da);
            out[c] = src[c] * (1.0 - da) + dst[c] * (1.0 - sa) + sa * da * blended;
        }
        out[3] = sa + da * (1.0 - sa);
        out
    }

    /// `TrivialBlend::MultiplyOpaque`, which is
    /// `Dst * src + OneMinusSrcAlpha * dst` on both colour and alpha.
    fn blend_state(src: [f64; 4], dst: [f64; 4]) -> [f64; 4] {
        let mut out = [0.0; 4];
        for c in 0..4 {
            out[c] = dst[c] * src[c] + (1.0 - src[3]) * dst[c];
        }
        out
    }

    /// Premultiplied samples: colour never exceeds alpha.
    fn samples() -> Vec<[f64; 4]> {
        let mut samples = Vec::new();
        for a in [0.0, 0.25, 0.5, 0.75, 1.0] {
            for c in [0.0, 0.2, 0.6, 1.0] {
                samples.push([a * c, a * (1.0 - c), a * c * 0.5, a]);
            }
        }
        samples
    }

    #[test]
    fn against_an_opaque_destination_the_blend_state_equals_the_shader() {
        for src in samples() {
            if src[3] == 0.0 {
                // The shader discards; see the variant's documentation.
                continue;
            }
            for dst in samples() {
                let dst = [dst[0], dst[1], dst[2], 1.0];
                let (theirs, ours) = (shader(src, dst), blend_state(src, dst));
                for c in 0..4 {
                    assert!(
                        (theirs[c] - ours[c]).abs() < 1e-12,
                        "channel {c}: shader {theirs:?} vs blend state {ours:?} \
                         for src {src:?} over {dst:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn the_destination_stays_opaque() {
        for src in samples() {
            let dst = [0.3, 0.4, 0.5, 1.0];
            assert_eq!(
                blend_state(src, dst)[3],
                1.0,
                "an opaque destination has to stay opaque, or the next \
                 multiply's precondition is silently false"
            );
        }
    }

    #[test]
    fn a_transparent_source_leaves_the_destination_alone() {
        // The shader discards where `src.a == 0`. The blend state writes
        // `Dst*0 + 1*dst`, which is the same thing, because a premultiplied
        // source with no alpha has no colour either.
        let dst = [0.3, 0.4, 0.5, 1.0];
        assert_eq!(blend_state([0.0; 4], dst), dst);
    }

    #[test]
    fn a_transparent_destination_is_where_it_stops_being_true() {
        // The one case the whole condition exists for: with `dst.a` below one
        // the shader's first term is no longer zero, and no blend state can
        // produce it.
        let src = [0.5, 0.25, 0.1, 0.5];
        let dst = [0.2, 0.2, 0.2, 0.5];
        let (theirs, ours) = (shader(src, dst), blend_state(src, dst));
        assert!(
            (theirs[0] - ours[0]).abs() > 1e-6,
            "if these agreed, the opacity check would be pointless: \
             shader {theirs:?} vs blend state {ours:?}"
        );
    }
}
