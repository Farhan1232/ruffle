# Where `output.expected.png` came from

Like `visual/blend_bounds`, this is not a capture from Flash Player. It is what
Ruffle drew immediately before blended groups of a single bitmap began to be
drawn without a render target of their own - `fix/aqw-blend-render-performance`
at `87797ecc1` - because that is the only image that can show the change is
invisible.

Taken at `quality = "high"`, so the multisample resolve is part of what is
compared. The two renderers agree on all 208,000 pixels with a maximum channel
difference of zero.

The first version of the direct path also accepted `Add`, `Subtract` and
`Screen`, and this test is why it does not any more: a rotated bitmap under
`Add` differed by up to 43 levels along its edges, because a saturating blend
clamps once per multisample when drawn directly and once for the whole group
when composited through a target. Rows three and four exist to keep that
covered.

If a future change to Ruffle's rendering is *meant* to change these pixels,
this image is not authority for what Flash does - check the case against Flash
Player before updating it.
