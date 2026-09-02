package {
    import flash.display.Sprite;
    import flash.events.Event;
    import flash.utils.Dictionary;

    // A weak-keyed Dictionary must not keep its keys alive, and an entry must
    // not keep itself alive through its own value. The test harness forces a
    // full collection at frame 20; frame 30 reports what survived.
    public class Test extends Sprite {
        private var weak:Dictionary = new Dictionary(true);
        private var strong:Dictionary = new Dictionary();
        private var held:Object;
        private var frame:int = 0;

        public function Test() {
            for (var i:int = 0; i < 20; i++) {
                // The value reaches its own key: `owner` holds `part`, and
                // the entry keyed on `part` stores an object holding `owner`.
                // Nothing outside the dictionaries references either.
                var owner:Object = { part: null };
                var part:Object = { owner: owner };
                owner.part = part;
                weak[part] = { owner: owner };
                // The control gets keys of its own, so that it does not keep
                // the weak dictionary's keys alive.
                var strongOwner:Object = { part: null };
                var strongPart:Object = { owner: strongOwner };
                strongOwner.part = strongPart;
                strong[strongPart] = { owner: strongOwner };
            }
            held = { name: "held" };
            weak[held] = "value of held";
            weak["stringKey"] = 1;
            weak[7] = 2;
            trace("weak entries before collection: " + count(weak));
            trace("strong entries before collection: " + count(strong));
            trace("held in weak: " + (held in weak));
            addEventListener(Event.ENTER_FRAME, onFrame);
        }

        private function count(d:Dictionary):int {
            var n:int = 0;
            for (var k:* in d) {
                n++;
            }
            return n;
        }

        private function onFrame(e:Event):void {
            frame++;
            if (frame == 30) {
                trace("weak entries after collection: " + count(weak));
                trace("strong entries after collection: " + count(strong));
                trace("held value: " + weak[held]);
                trace("string key: " + weak["stringKey"]);
                trace("int key: " + weak[7]);
                var seen:int = 0;
                for (var k:* in weak) {
                    if (k === held) {
                        seen++;
                    }
                }
                trace("held enumerated: " + seen);
                delete weak[held];
                trace("held in weak after delete: " + (held in weak));
                trace("weak entries after delete: " + count(weak));
                removeEventListener(Event.ENTER_FRAME, onFrame);
            }
        }
    }
}
