# WebGPU Block Compressed Marching Cubes

This is the implementation of the GPU-parallel marching cubes algorithm for
block-compressed data sets described in "Interactive Visualization of Terascale
Data in the Browser: Fact or Fiction?" by Will Usher and Valerio Pascucci appearing
at LDAV 2020. Please see [the paper](https://www.willusher.io/publications/teraweb)
and [the supplemental video](https://youtu.be/O7Tboj2dDVA) for more details.

## Usage

This code got blocked by the missing `GPULimits` API for WebGPU, Chrome Canary recently started
enforcing the minimum spec limits on storage buffers that this code exceeds, but didn't add
the API to request higher limits (which are available on desktop and laptop hardware).
Please see https://bugs.chromium.org/p/dawn/issues/detail?id=519 for some additional
information.

Note: I think this is now resolved, however the application also makes use of dynamic
storage buffers which are also now temporarily disabled in Chrome. See [issue 429](https://bugs.chromium.org/p/dawn/issues/detail?id=429)

A live demo will be put online along with final code polish and updates done once this
issue is resolved.

## Images

The data sets shown are available on the [Open SciVis Data Sets page](https://klacansky.com/open-scivis-datasets/).

The Chameleon data set
![Chameleon](https://i.imgur.com/l5goAsc.png)

The Plasma data set
![Plasma](https://i.imgur.com/HWJ4DHn.png)
