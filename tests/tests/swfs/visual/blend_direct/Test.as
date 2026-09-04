package {
    import flash.display.Sprite;
    import flash.display.Shape;
    import flash.display.Bitmap;
    import flash.display.BitmapData;
    import flash.display.BlendMode;
    import flash.display.DisplayObject;
    import flash.filters.BlurFilter;
    import flash.geom.ColorTransform;
    import flash.geom.Matrix;

    /// A blended group of one bitmap is drawn straight onto its destination
    /// instead of through a render target of its own. This is the matrix of
    /// shapes that can reach - or fail to reach - that path, drawn so that
    /// getting either the direct draw or the decision wrong is visible.
    ///
    /// The rows are, in order: what the direct path accepts (a lone bitmap
    /// under each trivial blend mode, moved, rotated, scaled, flipped, made
    /// transparent, colour-transformed, pushed off the stage edge, overlapped),
    /// and what it must refuse (a shape, a container of several children, a
    /// masked bitmap, a nested blend, a complex blend). Both halves matter: a
    /// refusal that should have been accepted only costs speed, but an
    /// acceptance that should have been refused changes the picture.
    [SWF(width="520", height="400", frameRate="24", backgroundColor="#101010")]
    public class Test extends Sprite {
        private var source:BitmapData;

        public function Test() {
            drawBackdrop();
            source = makeSource();

            // Row 1 - every trivial blend mode on a lone bitmap. These are the
            // modes the direct path is allowed to take.
            var modes:Array = [BlendMode.NORMAL, BlendMode.LAYER, BlendMode.ADD,
                               BlendMode.SUBTRACT, BlendMode.SCREEN];
            for (var i:int = 0; i < modes.length; i++) {
                place(bitmap(), 10 + i * 100, 10, modes[i]);
            }

            // Row 2 - the same lone bitmap moved and transformed.
            place(bitmap(), 10.37, 90.62, BlendMode.LAYER);          // fractional
            var rotated:Bitmap = bitmap();
            rotated.rotation = 27;
            place(rotated, 110, 90, BlendMode.ADD);
            var scaled:Bitmap = bitmap();
            scaled.scaleX = 1.6; scaled.scaleY = 0.7;
            place(scaled, 210, 90, BlendMode.SCREEN);
            var flipped:Bitmap = bitmap();
            flipped.scaleX = -1; flipped.scaleY = -1;
            place(flipped, 390, 160, BlendMode.LAYER);
            var faded:Bitmap = bitmap();
            faded.alpha = 0.45;
            place(faded, 410, 90, BlendMode.LAYER);

            // Row 3 - colour transform, partial offscreen, overlap.
            var tinted:Bitmap = bitmap();
            tinted.transform.colorTransform =
                new ColorTransform(1.0, 0.35, 0.35, 0.8, 40, 0, 0, 0);
            place(tinted, 10, 170, BlendMode.LAYER);
            place(bitmap(), -30, 170, BlendMode.ADD);                // off the left
            place(bitmap(), 110, 170, BlendMode.SCREEN);
            place(bitmap(), 140, 185, BlendMode.ADD);                // overlapping it
            place(bitmap(), 230, 250, BlendMode.SUBTRACT);
            place(bitmap(), 250, 265, BlendMode.SCREEN);             // overlapping it

            // Row 4 - the cases the direct path must refuse.
            //
            // A shape: its pipelines have no blend-state variants.
            var shape:Sprite = new Sprite();
            var box:Shape = new Shape();
            box.graphics.beginFill(0x44ddaa);
            box.graphics.drawRect(0, 0, 60, 46);
            box.graphics.endFill();
            shape.addChild(box);
            place(shape, 10, 250, BlendMode.SCREEN);

            // A container of several children: it must composite as a unit.
            var container:Sprite = new Sprite();
            container.addChild(bitmap());
            var second:Bitmap = bitmap();
            second.x = 22; second.y = 16;
            container.addChild(second);
            place(container, 90, 250, BlendMode.ADD);

            // A masked bitmap.
            var maskedHolder:Sprite = new Sprite();
            var masked:Bitmap = bitmap();
            var cover:Shape = new Shape();
            cover.graphics.beginFill(0x000000);
            cover.graphics.drawRect(0, 0, 44, 30);
            cover.graphics.endFill();
            masked.mask = cover;
            maskedHolder.addChild(masked);
            maskedHolder.addChild(cover);
            place(maskedHolder, 170, 250, BlendMode.LAYER);

            // A cached object, and a cached object under a blend - the case
            // the direct path exists for.
            var cached:Bitmap = bitmap();
            cached.cacheAsBitmap = true;
            place(cached, 320, 250, BlendMode.LAYER);

            var filtered:Sprite = new Sprite();
            filtered.addChild(bitmap());
            filtered.filters = [new BlurFilter(6, 6, 2)];
            place(filtered, 400, 250, BlendMode.SCREEN);

            // A trivial blend nested inside another trivial blend.
            var outer:Sprite = new Sprite();
            var inner:Bitmap = bitmap();
            inner.blendMode = BlendMode.ADD;
            inner.x = 14; inner.y = 10;
            outer.addChild(bitmap());
            outer.addChild(inner);
            place(outer, 10, 320, BlendMode.LAYER);

            // A trivial blend beside a complex one, so the two paths meet.
            place(bitmap(), 130, 320, BlendMode.SCREEN);
            place(bitmap(), 160, 330, BlendMode.MULTIPLY);
            place(bitmap(), 250, 320, BlendMode.DIFFERENCE);
            place(bitmap(), 280, 330, BlendMode.ADD);
            place(bitmap(), 370, 320, BlendMode.HARDLIGHT);
        }

        /// Stripes, so a blend that lands in the wrong place or samples the
        /// wrong destination shows as a broken stripe rather than as nothing.
        private function drawBackdrop():void {
            var backdrop:Shape = new Shape();
            for (var i:int = 0; i < 14; i++) {
                backdrop.graphics.beginFill(i % 2 == 0 ? 0x585858 : 0xa8a8a8);
                backdrop.graphics.drawRect(0, i * 30, 520, 30);
                backdrop.graphics.endFill();
            }
            addChild(backdrop);
        }

        /// A bitmap with structure in it, so resampling or a one-pixel drift
        /// would be visible.
        private function makeSource():BitmapData {
            var data:BitmapData = new BitmapData(60, 46, true, 0x00000000);
            for (var x:int = 0; x < 60; x++) {
                for (var y:int = 0; y < 46; y++) {
                    var edge:Boolean = x < 3 || y < 3 || x > 56 || y > 42;
                    var alpha:uint = edge ? 0xff : 0xc0;
                    data.setPixel32(x, y,
                        (alpha << 24) | ((x * 4) << 16) | ((y * 5) << 8) | 0x80);
                }
            }
            return data;
        }

        private function bitmap():Bitmap {
            return new Bitmap(source);
        }

        private function place(child:DisplayObject, x:Number, y:Number, blend:String):void {
            child.x = x;
            child.y = y;
            child.blendMode = blend;
            addChild(child);
        }
    }
}
