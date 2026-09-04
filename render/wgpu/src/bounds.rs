//! Working out how much of the screen a group of draw commands can touch.
//!
//! Every display object with a non-normal blend mode, and every alpha mask, is
//! rendered through a temporary render target of its own before being
//! composited back. Sizing those targets at the whole viewport is what makes a
//! crowded room expensive: a screen-sized target is several megabytes to
//! allocate, clear, blend and sample, and a scene can want hundreds of them in
//! a single frame. Nearly all of that is empty space around one avatar.
//!
//! The commands themselves say where they draw, so this module walks them and
//! returns the rectangle that contains everything they can touch. What makes
//! the smaller target *equivalent* rather than merely cheaper is that every way
//! one of these targets is consumed leaves the destination untouched where the
//! target is transparent:
//!
//! * every complex-blend shader `discard`s where `src.a == 0`;
//! * every [`TrivialBlend`](crate::blend::TrivialBlend) blend state is the
//!   identity on the destination for a zero source (`Normal` and `Screen` scale
//!   the destination by `1 - src`, `Add` and `Subtract` add or subtract zero);
//! * the alpha-mask shader multiplies by `src.a`, giving transparent out.
//!
//! A target is cleared to transparent outside the area its commands draw, so
//! the pixels this rectangle excludes were contributing nothing.

use crate::as_texture;
use crate::mesh::as_mesh;
use ruffle_render::commands::{Command, CommandList};
use ruffle_render::matrix::Matrix;

/// An axis-aligned rectangle in a surface's pixel space.
///
/// Empty is represented by `x_min > x_max`, which is what [`PixelRect::EMPTY`]
/// starts from so that [`PixelRect::union`] of nothing is nothing.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct PixelRect {
    pub x_min: f32,
    pub y_min: f32,
    pub x_max: f32,
    pub y_max: f32,
}

impl PixelRect {
    pub const EMPTY: Self = Self {
        x_min: f32::INFINITY,
        y_min: f32::INFINITY,
        x_max: f32::NEG_INFINITY,
        y_max: f32::NEG_INFINITY,
    };

    pub fn new(x_min: f32, y_min: f32, x_max: f32, y_max: f32) -> Self {
        Self {
            x_min,
            y_min,
            x_max,
            y_max,
        }
    }

    /// The rectangle `(0, 0)` to `(width, height)`.
    pub fn from_size(width: u32, height: u32) -> Self {
        Self::new(0.0, 0.0, width as f32, height as f32)
    }

    /// Whether this rectangle stands for "nothing at all".
    ///
    /// A rectangle of zero width or height is not empty: a horizontal line has
    /// no height and still lights up a row of pixels once
    /// [`grow`](Self::grow) has given it the slack rasterisation needs.
    pub fn is_empty(&self) -> bool {
        !(self.x_max >= self.x_min && self.y_max >= self.y_min)
    }

    pub fn union(self, other: Self) -> Self {
        if self.is_empty() {
            return other;
        }
        if other.is_empty() {
            return self;
        }
        Self {
            x_min: self.x_min.min(other.x_min),
            y_min: self.y_min.min(other.y_min),
            x_max: self.x_max.max(other.x_max),
            y_max: self.y_max.max(other.y_max),
        }
    }

    pub fn intersect(self, other: Self) -> Self {
        let result = Self {
            x_min: self.x_min.max(other.x_min),
            y_min: self.y_min.max(other.y_min),
            x_max: self.x_max.min(other.x_max),
            y_max: self.y_max.min(other.y_max),
        };
        if result.is_empty() {
            Self::EMPTY
        } else {
            result
        }
    }

    /// Grows the rectangle by `amount` on every side.
    ///
    /// Used for the slack that rasterisation needs: a triangle edge or a line
    /// lights up samples in the pixels its geometry only partly covers.
    pub fn grow(self, amount: f32) -> Self {
        if self.is_empty() {
            return self;
        }
        Self {
            x_min: self.x_min - amount,
            y_min: self.y_min - amount,
            x_max: self.x_max + amount,
            y_max: self.y_max + amount,
        }
    }

    /// Rounds outwards to whole pixels.
    ///
    /// Targets have to start on a whole pixel: an object rendered into one
    /// keeps the sub-pixel phase it would have had on screen only if the
    /// target's origin is a whole number of pixels away from the screen's.
    pub fn snap_out(self) -> Self {
        if self.is_empty() {
            return self;
        }
        Self {
            x_min: self.x_min.floor(),
            y_min: self.y_min.floor(),
            x_max: self.x_max.ceil(),
            y_max: self.y_max.ceil(),
        }
    }

    pub fn width(&self) -> f32 {
        (self.x_max - self.x_min).max(0.0)
    }

    pub fn height(&self) -> f32 {
        (self.y_max - self.y_min).max(0.0)
    }

    /// The bounds of `self` after `matrix` maps it into another space.
    ///
    /// The renderer's world matrices take local pixels to surface pixels, with
    /// the translation carried in twips, which is how they are handed to the
    /// shaders.
    pub fn transform(self, matrix: &Matrix) -> Self {
        if self.is_empty() {
            return self;
        }
        let tx = matrix.tx.to_pixels() as f32;
        let ty = matrix.ty.to_pixels() as f32;
        let corners = [
            (self.x_min, self.y_min),
            (self.x_max, self.y_min),
            (self.x_min, self.y_max),
            (self.x_max, self.y_max),
        ];
        let mut out = Self::EMPTY;
        for (x, y) in corners {
            let px = matrix.a * x + matrix.c * y + tx;
            let py = matrix.b * x + matrix.d * y + ty;
            out.x_min = out.x_min.min(px);
            out.y_min = out.y_min.min(py);
            out.x_max = out.x_max.max(px);
            out.y_max = out.y_max.max(py);
        }
        out
    }
}

/// The unit square, which is the geometry of every quad-shaped draw command:
/// bitmaps, rectangles, lines and composited sub-targets all scale it by their
/// own matrix.
const UNIT_SQUARE: PixelRect = PixelRect {
    x_min: 0.0,
    y_min: 0.0,
    x_max: 1.0,
    y_max: 1.0,
};

/// How far past its geometry a draw can light up pixels.
///
/// Triangles are clipped to their own edges, but multisample resolve and the
/// one-pixel-wide line topology both reach into the neighbouring pixel, and a
/// bitmap sampled with smoothing takes half a texel of slack at its edges.
const RASTER_SLACK: f32 = 1.0;

/// Where a group of commands is in the middle of drawing a masker rather than
/// the content it masks.
#[derive(Copy, Clone, PartialEq)]
enum MaskPhase {
    /// Between `PushMask` and `ActivateMask`: the masker's own geometry, which
    /// writes the stencil and no colour.
    Masker,
    /// Between `ActivateMask` and `DeactivateMask`: the masked content.
    Content,
    /// Between `DeactivateMask` and `PopMask`: the masker's geometry again, to
    /// take the stencil back down. No colour either.
    ClearMasker,
}

struct MaskLevel {
    phase: MaskPhase,
    /// What the masker drew. Content under this level can only show through
    /// here, so it is intersected with this.
    bounds: PixelRect,
}

/// Accumulates the area a [`CommandList`] can draw to, in its surface's pixels.
struct BoundsWalker {
    masks: Vec<MaskLevel>,
    result: PixelRect,
}

impl BoundsWalker {
    fn new() -> Self {
        Self {
            masks: Vec::new(),
            result: PixelRect::EMPTY,
        }
    }

    /// Records a rectangle a draw command can touch.
    ///
    /// While a masker is being drawn the rectangle belongs to that masker
    /// rather than to the result; the masker writes no colour of its own, and
    /// what it does write bounds the content underneath it.
    fn add(&mut self, rect: PixelRect) {
        match self.masks.last_mut() {
            Some(level) if level.phase == MaskPhase::Masker => {
                level.bounds = level.bounds.union(rect);
            }
            // The clearing pass redraws geometry already recorded when the
            // masker was first drawn.
            Some(level) if level.phase == MaskPhase::ClearMasker => {}
            _ => {
                let mut rect = rect;
                for level in &self.masks {
                    rect = rect.intersect(level.bounds);
                }
                self.result = self.result.union(rect);
            }
        }
    }

    fn walk(&mut self, commands: &CommandList) {
        for command in &commands.commands {
            match command {
                Command::RenderBitmap {
                    transform,
                    pixel_snapping,
                    region,
                    ..
                } => {
                    let mut matrix = transform.matrix;
                    pixel_snapping.apply(&mut matrix);
                    matrix *= Matrix::scale(region.width() as f32, region.height() as f32);
                    self.add(UNIT_SQUARE.transform(&matrix).grow(RASTER_SLACK));
                }
                Command::RenderStage3D { bitmap, transform } => {
                    let texture = as_texture(bitmap);
                    let matrix = transform.matrix
                        * Matrix::scale(
                            texture.texture.width() as f32,
                            texture.texture.height() as f32,
                        );
                    self.add(UNIT_SQUARE.transform(&matrix).grow(RASTER_SLACK));
                }
                Command::RenderShape { shape, transform } => {
                    let mesh = as_mesh(shape);
                    self.add(mesh.bounds.transform(&transform.matrix).grow(RASTER_SLACK));
                }
                Command::DrawRect { matrix, .. }
                | Command::DrawLine { matrix, .. }
                | Command::DrawLineRect { matrix, .. } => {
                    self.add(UNIT_SQUARE.transform(matrix).grow(RASTER_SLACK));
                }
                Command::RenderAlphaMask {
                    maskee_commands,
                    mask_commands,
                } => {
                    // The alpha-mask shader multiplies the maskee by the mask's
                    // alpha, so the result is inside both.
                    let maskee = content_bounds(maskee_commands);
                    let mask = content_bounds(mask_commands);
                    self.add(maskee.intersect(mask));
                }
                Command::Blend(commands, _) => {
                    self.add(content_bounds(commands));
                }
                Command::PushMask => self.masks.push(MaskLevel {
                    phase: MaskPhase::Masker,
                    bounds: PixelRect::EMPTY,
                }),
                Command::ActivateMask => {
                    if let Some(level) = self.masks.last_mut() {
                        level.phase = MaskPhase::Content;
                    }
                }
                Command::DeactivateMask => {
                    if let Some(level) = self.masks.last_mut() {
                        level.phase = MaskPhase::ClearMasker;
                    }
                }
                Command::PopMask => {
                    self.masks.pop();
                }
            }
        }
    }
}

/// The area `commands` can draw to, in the pixels of the surface they are drawn
/// into.
///
/// The result is conservative: it may be larger than what is drawn (a masker
/// that covers more than its content, geometry that does not fill its own
/// bounding box), never smaller.
pub fn content_bounds(commands: &CommandList) -> PixelRect {
    let mut walker = BoundsWalker::new();
    walker.walk(commands);
    walker.result
}

/// A whole-pixel rectangle: where a render target sits in the surface that
/// composites it, and how big it is.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct TargetRect {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

impl TargetRect {
    pub fn from_size(width: u32, height: u32) -> Self {
        Self {
            x: 0,
            y: 0,
            width,
            height,
        }
    }

    pub fn right(&self) -> i32 {
        self.x + self.width as i32
    }

    pub fn bottom(&self) -> i32 {
        self.y + self.height as i32
    }

    pub fn extent(&self) -> wgpu::Extent3d {
        wgpu::Extent3d {
            width: self.width,
            height: self.height,
            depth_or_array_layers: 1,
        }
    }

    pub fn as_pixel_rect(&self) -> PixelRect {
        PixelRect::new(
            self.x as f32,
            self.y as f32,
            self.right() as f32,
            self.bottom() as f32,
        )
    }
}

/// Rounds a target dimension up to one of a small set of sizes.
///
/// Targets come from a pool keyed on their exact dimensions, so a renderer that
/// asked for the precise size of every object would never re-use anything and
/// would hold a pool entry per size it had ever seen. Rounding to a size class
/// trades a little slack inside each target - at most a class step per side -
/// for a pool with tens of keys instead of thousands. The steps grow with the
/// size so that the slack stays proportional.
pub fn size_class(size: u32) -> u32 {
    fn round_up(size: u32, step: u32) -> u32 {
        size.div_ceil(step) * step
    }
    match size {
        0..=64 => 64,
        65..=512 => round_up(size, 64),
        513..=1024 => round_up(size, 128),
        1025..=2048 => round_up(size, 256),
        _ => round_up(size, 512),
    }
}

/// Chooses the render target for a group of commands that will be composited
/// back into `parent`.
///
/// The target has to contain everything `content` can draw, sit on whole pixels
/// so that what is drawn into it keeps the sub-pixel phase it would have had on
/// screen, and stay inside `parent`, which is all that will ever be seen of it.
pub fn target_rect_for(content: PixelRect, parent: TargetRect) -> TargetRect {
    let content = content.snap_out().intersect(parent.as_pixel_rect());
    let (content_x, content_y, content_width, content_height) = if content.is_empty() {
        // Nothing to draw - a group that is entirely off-screen, or masked away.
        // It still needs a target to be composited from, but the smallest one
        // will do: every consumer leaves the destination alone where the target
        // is transparent.
        (parent.x as f32, parent.y as f32, 0.0, 0.0)
    } else {
        (
            content.x_min,
            content.y_min,
            content.width(),
            content.height(),
        )
    };

    let width = size_class(content_width as u32).clamp(1, parent.width.max(1));
    let height = size_class(content_height as u32).clamp(1, parent.height.max(1));

    // Slide the target back inside the parent if rounding pushed it over the
    // edge. `width` is never less than the content's width and never more than
    // the parent's, so this keeps the content covered either way.
    let x = (content_x as i32).clamp(parent.x, (parent.right() - width as i32).max(parent.x));
    let y = (content_y as i32).clamp(parent.y, (parent.bottom() - height as i32).max(parent.y));

    TargetRect {
        x,
        y,
        width,
        height,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const VIEWPORT: TargetRect = TargetRect {
        x: 0,
        y: 0,
        width: 1920,
        height: 985,
    };

    /// The whole point: an avatar-sized object does not get a screen-sized
    /// target.
    #[test]
    fn a_small_object_gets_a_small_target() {
        let rect = target_rect_for(PixelRect::new(800.0, 400.0, 950.0, 600.0), VIEWPORT);
        assert!(
            rect.width <= 256 && rect.height <= 256,
            "a 150x200 object was given a {}x{} target",
            rect.width,
            rect.height
        );
    }

    /// Whatever else it does, the target has to contain everything the commands
    /// can draw, or the object would be cropped on screen.
    #[test]
    fn the_target_covers_the_content() {
        let cases = [
            PixelRect::new(0.0, 0.0, 1920.0, 985.0),
            PixelRect::new(-500.0, -500.0, 40.0, 40.0),
            PixelRect::new(1900.0, 960.0, 2400.0, 1200.0),
            PixelRect::new(1919.5, 984.5, 1920.5, 985.5),
            PixelRect::new(0.25, 0.25, 0.75, 0.75),
            PixelRect::new(600.0, 300.0, 600.0, 700.0),
        ];
        for content in cases {
            let rect = target_rect_for(content, VIEWPORT);
            let visible = content.snap_out().intersect(VIEWPORT.as_pixel_rect());
            assert!(
                rect.x >= VIEWPORT.x
                    && rect.y >= VIEWPORT.y
                    && rect.right() <= VIEWPORT.right()
                    && rect.bottom() <= VIEWPORT.bottom(),
                "{rect:?} leaves the viewport for {content:?}"
            );
            if !visible.is_empty() {
                assert!(
                    rect.x as f32 <= visible.x_min
                        && rect.y as f32 <= visible.y_min
                        && rect.right() as f32 >= visible.x_max
                        && rect.bottom() as f32 >= visible.y_max,
                    "{rect:?} does not cover {visible:?}"
                );
            }
        }
    }

    /// Content that fills the screen still gets exactly the screen, so the
    /// full-size pool key that everything used to share is still used.
    #[test]
    fn full_screen_content_keeps_the_full_size_target() {
        let rect = target_rect_for(PixelRect::new(0.0, 0.0, 1920.0, 985.0), VIEWPORT);
        assert_eq!(rect, VIEWPORT);
    }

    /// Sizes have to collapse onto a handful of classes, or the pool would hold
    /// a key per object.
    #[test]
    fn sizes_collapse_onto_few_classes() {
        let classes: std::collections::BTreeSet<u32> = (1..=1920).map(size_class).collect();
        assert!(
            classes.len() < 24,
            "{} distinct size classes below 1920",
            classes.len()
        );
        for size in 1..=1920u32 {
            assert!(size_class(size) >= size, "{size} rounded down");
        }
    }

    /// Targets start on a whole pixel, so an object drawn into one lands on the
    /// same sub-pixel positions it would have on screen.
    #[test]
    fn targets_are_whole_pixels() {
        let rect = target_rect_for(PixelRect::new(100.37, 200.62, 260.37, 400.62), VIEWPORT);
        assert_eq!(rect.x, 100);
        assert_eq!(rect.y, 200);
    }
}
