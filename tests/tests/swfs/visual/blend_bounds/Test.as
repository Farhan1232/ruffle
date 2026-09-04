package {
    import flash.display.Sprite;
    import flash.display.Shape;
    import flash.display.BlendMode;
    import flash.display.MovieClip;
    import flash.filters.BlurFilter;
    import flash.filters.GlowFilter;
    import flash.filters.DropShadowFilter;
    import flash.filters.BevelFilter;
    import flash.filters.GradientGlowFilter;
    import flash.filters.ColorMatrixFilter;
    import flash.filters.DisplacementMapFilter;
    import flash.filters.DisplacementMapFilterMode;
    import flash.display.BitmapData;
    import flash.geom.Matrix;
    import flash.geom.Point;

    /// Every shape of blended object whose render target is now sized on its
    /// own contents rather than on the whole stage.
    ///
    /// A blended display object is drawn through a temporary render target and
    /// composited back. That target used to be the size of the viewport
    /// whatever the object's size; it is now the smallest rectangle the
    /// object's own drawing commands can reach, rounded out to whole pixels and
    /// up to a size class. Every case below is one where getting the rectangle
    /// or its placement wrong would move, crop or rescale what is drawn:
    ///
    ///   * plain translation, and translation onto a fractional pixel;
    ///   * scale, including a negative scale, which flips the bounds;
    ///   * rotation and skew, where the bounds are of the transformed shape;
    ///   * objects partly and wholly outside the stage;
    ///   * blends inside blends, and blends inside masks;
    ///   * a mask larger than what it masks, and one smaller;
    ///   * filters, whose growth has to be inside the target;
    ///   * an object that really does cover the whole stage.
    ///
    /// Each row is drawn twice, once with a trivial blend (Layer, composited
    /// with a blend state) and once with a complex one (Multiply, composited
    /// through a shader that reads the destination), because the two take
    /// different paths through the renderer.
    [SWF(width="480", height="384", frameRate="24", backgroundColor="#101010")]
    public class Test extends Sprite {
        public function Test() {
            drawBackdrop();

            // Column 1: trivial blend. Column 2: complex blend.
            buildColumn(0, BlendMode.LAYER);
            buildColumn(240, BlendMode.MULTIPLY);

            // A blended object that really is the whole stage, which must still
            // get a stage-sized target.
            var full:Sprite = boxSprite(0x2244aa, 480, 380);
            full.alpha = 0.25;
            full.blendMode = BlendMode.SCREEN;
            addChild(full);
        }

        /// Something for the blends to blend with, so a wrongly positioned
        /// target shows up as a wrongly coloured overlap rather than as
        /// nothing.
        private function drawBackdrop():void {
            var backdrop:Shape = new Shape();
            for (var i:int = 0; i < 12; i++) {
                backdrop.graphics.beginFill(i % 2 == 0 ? 0x606060 : 0xb0b0b0);
                backdrop.graphics.drawRect(0, i * 32, 480, 32);
                backdrop.graphics.endFill();
            }
            addChild(backdrop);
        }

        private function boxSprite(color:uint, width:Number, height:Number):Sprite {
            var sprite:Sprite = new Sprite();
            var shape:Shape = new Shape();
            shape.graphics.beginFill(color);
            shape.graphics.drawRect(0, 0, width, height);
            shape.graphics.endFill();
            // A stroke, so that the tessellated bounds are wider than the fill
            // and a target sized on the fill alone would clip it.
            shape.graphics.lineStyle(4, 0xffffff);
            shape.graphics.drawRect(0, 0, width, height);
            sprite.addChild(shape);
            return sprite;
        }

        private function place(child:Sprite, x:Number, y:Number, blend:String):void {
            child.x = x;
            child.y = y;
            child.blendMode = blend;
            addChild(child);
        }

        private function buildColumn(left:Number, blend:String):void {
            // Plain translation.
            place(boxSprite(0xcc3333, 40, 30), left + 10, 10, blend);

            // A fractional position, which must keep its sub-pixel phase.
            place(boxSprite(0x33cc33, 40, 30), left + 60.37, 10.62, blend);

            // Scaled up, and scaled negatively so the bounds flip.
            var scaled:Sprite = boxSprite(0x3333cc, 20, 15);
            scaled.scaleX = 2.5;
            scaled.scaleY = 2.0;
            place(scaled, left + 115, 10, blend);

            var flipped:Sprite = boxSprite(0xcccc33, 40, 30);
            flipped.scaleX = -1;
            flipped.scaleY = -1;
            place(flipped, left + 210, 45, blend);

            // Rotated and skewed, where the bounds are of the transformed shape.
            var rotated:Sprite = boxSprite(0xcc33cc, 40, 30);
            rotated.rotation = 33;
            place(rotated, left + 15, 60, blend);

            var skewed:Sprite = boxSprite(0x33cccc, 40, 30);
            var skew:Matrix = skewed.transform.matrix;
            skew.c = 0.6;
            skew.b = 0.25;
            skewed.transform.matrix = skew;
            place(skewed, left + 80, 60, blend);

            // Partly off the left edge, and off the top.
            place(boxSprite(0xff8800, 60, 40), left == 0 ? -25 : left + 195, 65, blend);

            // A nested transform: a blended child inside a scaled, rotated,
            // blended parent.
            var outer:Sprite = new Sprite();
            var inner:Sprite = boxSprite(0x88ff88, 30, 24);
            inner.blendMode = blend;
            inner.x = 12;
            inner.y = 8;
            outer.addChild(boxSprite(0x004400, 60, 44));
            outer.addChild(inner);
            outer.scaleX = 1.4;
            outer.scaleY = 1.2;
            outer.rotation = -12;
            place(outer, left + 25, 130, blend);

            // A mask bigger than its content, and one smaller, each around a
            // blended object.
            addChild(masked(left + 120, 120, blend, 80, 60));
            addChild(masked(left + 120, 190, blend, 24, 18));

            // Filters, whose growth is baked into the bitmap the renderer sees.
            var blurred:Sprite = boxSprite(0xffffff, 34, 26);
            blurred.filters = [new BlurFilter(8, 8, 2)];
            place(blurred, left + 15, 250, blend);

            var glowing:Sprite = boxSprite(0x442200, 34, 26);
            glowing.filters = [new GlowFilter(0xffaa00, 1.0, 12, 12, 2, 2)];
            place(glowing, left + 80, 250, blend);

            var shadowed:Sprite = boxSprite(0x224422, 34, 26);
            shadowed.filters = [new DropShadowFilter(6, 45, 0, 1.0, 8, 8, 1, 2)];
            place(shadowed, left + 145, 250, blend);

            // Bevel and gradient glow, which grow the bounds asymmetrically.
            var bevelled:Sprite = boxSprite(0x666688, 30, 24);
            bevelled.filters = [new BevelFilter(6, 45, 0xffffff, 1.0, 0x000000, 1.0, 6, 6, 1, 2)];
            place(bevelled, left + 200, 250, blend);

            var gradientGlow:Sprite = boxSprite(0x222222, 26, 20);
            gradientGlow.filters = [new GradientGlowFilter(
                0, 45, [0xff0000, 0x00ff00], [1.0, 1.0], [0, 255], 10, 10, 2, 2)];
            place(gradientGlow, left + 160, 310, blend);

            // A displacement map, which is where a previous attempt at bounding
            // these targets went wrong: its output is not inside its input.
            var displaced:Sprite = boxSprite(0x8800ff, 34, 26);
            displaced.filters = [
                new ColorMatrixFilter([
                    1, 0, 0, 0, 0,
                    0, 1, 0, 0, 0,
                    0, 0, 1, 0, 0,
                    0, 0, 0, 1, 0
                ]),
                new DisplacementMapFilter(
                    displacementMap(), new Point(0, 0), 1, 2, 12, 12,
                    DisplacementMapFilterMode.CLAMP)
            ];
            place(displaced, left + 215, 310, blend);

            // A cached object under a blend.
            var cached:Sprite = boxSprite(0xaa00aa, 40, 30);
            cached.cacheAsBitmap = true;
            place(cached, left + 15, 310, blend);

            // Wholly off-stage: must draw nothing and disturb nothing.
            place(boxSprite(0xff0000, 40, 30), left - 400, -400, blend);

            // A blend whose child is itself a differently blended group, so one
            // cropped target composites into another.
            var group:Sprite = new Sprite();
            var a:Sprite = boxSprite(0x0088ff, 36, 28);
            a.blendMode = BlendMode.ADD;
            var b:Sprite = boxSprite(0xff8800, 36, 28);
            b.blendMode = BlendMode.DIFFERENCE;
            b.x = 18;
            b.y = 10;
            group.addChild(a);
            group.addChild(b);
            place(group, left + 80, 310, blend);
        }

        /// A ramp for the displacement filter to push pixels around with.
        private function displacementMap():BitmapData {
            var map:BitmapData = new BitmapData(34, 26, false, 0x808080);
            for (var x:int = 0; x < 34; x++) {
                for (var y:int = 0; y < 26; y++) {
                    map.setPixel(x, y, (x * 7) << 16 | (y * 9) << 8);
                }
            }
            return map;
        }

        /// A blended box behind a mask of the given size, so the target has to
        /// be right whether the mask is larger or smaller than the content.
        private function masked(x:Number, y:Number, blend:String,
                                maskWidth:Number, maskHeight:Number):Sprite {
            var holder:Sprite = new Sprite();
            var content:Sprite = boxSprite(0xff44aa, 46, 34);
            content.blendMode = blend;
            var mask:Shape = new Shape();
            mask.graphics.beginFill(0x000000);
            mask.graphics.drawRect(0, 0, maskWidth, maskHeight);
            mask.graphics.endFill();
            content.mask = mask;
            holder.addChild(content);
            holder.addChild(mask);
            holder.x = x;
            holder.y = y;
            return holder;
        }
    }
}
