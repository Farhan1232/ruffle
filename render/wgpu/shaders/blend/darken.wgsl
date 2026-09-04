// NOTE: The `common.wgsl` source is prepended to this before compilation.

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    /// Where this corner lands in the destination being blended onto, which is
    /// not always the target being drawn into: an Alpha or Erase reads the
    /// nearest layer above it.
    @location(0) parent_uv: vec2<f32>,
    /// Where this corner is in the blended group's own target. The quad covers
    /// that target exactly, so this is just the quad's own coordinates.
    @location(1) current_uv: vec2<f32>,
};

@group(1) @binding(0) var<uniform> transforms: common__Transforms;
@group(2) @binding(0) var parent_texture: texture_2d<f32>;
@group(2) @binding(1) var current_texture: texture_2d<f32>;
@group(2) @binding(2) var texture_sampler: sampler;

@vertex
fn main_vertex(in: common__VertexInputUv) -> VertexOutput {
    let pos = common__globals.view_matrix * transforms.world_matrix * vec4<f32>(in.position.x, in.position.y, 1.0, 1.0);
    // The group's pixels may be a region of a page shared with its siblings
    // rather than a texture of its own, so where to sample them is passed in
    // rather than assumed to be the whole texture. `transforms.mult_color` is
    // `[u0, v0, du, dv]`; a whole texture is `[0, 0, 1, 1]`, which gives back
    // the quad's own coordinates.
    let current_uv = transforms.mult_color.xy + in.position * transforms.mult_color.zw;
    return VertexOutput(pos, in.uv.xy / in.uv.z, current_uv);
}

fn blend_func(src: vec3<f32>, dst: vec3<f32>) -> vec3<f32> {
    return min(src, dst);
}

@fragment
fn main_fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    // dst is the parent pixel we're blending onto
    var dst: vec4<f32> = textureSample(parent_texture, texture_sampler, in.parent_uv);
    // src is the pixel that we want to apply
    var src: vec4<f32> = textureSample(current_texture, texture_sampler, in.current_uv);

    if (src.a > 0.0) {
        return vec4<f32>(src.rgb * (1.0 - dst.a) + dst.rgb * (1.0 - src.a) + src.a * dst.a * blend_func(src.rgb / src.a, dst.rgb / dst.a), src.a + dst.a * (1.0 - src.a));
    } else {
        if (true) {
            // This needs to be in a branch because... reasons. Bug in naga.
            // https://github.com/gfx-rs/naga/issues/2168
            discard;
        }
        return dst;
    }
}
