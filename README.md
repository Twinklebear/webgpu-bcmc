# WebGPU Block Compressed Marching Cubes

This is the implementation of the GPU-parallel marching cubes algorithm for
block-compressed data sets described in "Interactive Visualization of Terascale
Data in the Browser: Fact or Fiction?" by Will Usher and Valerio Pascucci appearing
at LDAV 2020. Please see [the paper](https://www.willusher.io/publications/teraweb)
and [the supplemental video](https://youtu.be/O7Tboj2dDVA) for more details.

## Usage

This is now running to workarounds and updates implemented to get around temporary
missing APIs and disabled features or limitations. The application runs in
Chrome Canary 92.0.4512.0 (and later versions as long as the WebGPU API is not changed),
though some data decompression corruption has been seen on macOS for some data sets. On Windows it runs
well, so try it out online!

- [Skull](https://www.willusher.io/webgpu-bcmc/webgpu_bcmc.html) (256^3)
- [Magnetic Reconnection](https://www.willusher.io/webgpu-bcmc/webgpu_bcmc.html#magnetic) (512^3)

## Images

The data sets shown are available on the [Open SciVis Data Sets page](https://klacansky.com/open-scivis-datasets/).

The Chameleon data set
![Chameleon](https://i.imgur.com/l5goAsc.png)

The Plasma data set
![Plasma](https://i.imgur.com/HWJ4DHn.png)
