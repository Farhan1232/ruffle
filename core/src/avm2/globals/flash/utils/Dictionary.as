package flash.utils {
    [Ruffle(InstanceAllocator)]
    public dynamic class Dictionary {
        prototype.toJSON = function(r:String):* {
            return "Dictionary";
        };
        prototype.setPropertyIsEnumerable("toJSON", false);

        public function Dictionary(weakKeys:Boolean = false) {
            this.init(weakKeys);
        }

        private native function init(weakKeys:Boolean):void;
    }
}
