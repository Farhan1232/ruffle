# Where `output.expected.png` came from

Unlike most image comparisons here, this one is not a capture from Flash
Player. It is what Ruffle's own renderer drew before blend render targets were
sized on their contents (`fix/aqw-render-surface-pool`, commit `140979e2`),
which is exactly what this test is for: the optimization has to leave every
pixel where it was, and the only image that can prove that is the one the old
renderer produced.

It was taken at `quality = "high"`, so multisampling is part of what is being
compared, and the two renderers agreed on all 184,320 pixels with a maximum
channel difference of zero.

If a future change to Ruffle's rendering is *meant* to change these pixels,
this image is not authority for what Flash does - check the case against Flash
Player before updating it.
