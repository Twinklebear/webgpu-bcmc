var VolumeRaycaster = function (device) {
    this.device = device;
    this.scanPipeline = new ExclusiveScanPipeline(device);
    this.streamCompact = new StreamCompact(device);
    this.numActiveBlocks = 0;
    this.numVertices = 0;
    this.numBlocksWithVertices = 0;

    var canvas = document.getElementById("webgpu-canvas");
    var context = canvas.getContext("gpupresent");

    // Max dispatch size for more computationally heavy kernels
    // which might hit TDR on lower power devices
    this.maxDispatchSize = 512000;

    this.numActiveBlocksStorage = 0;
    this.numBlocksWithVerticesStorage = 0;
    this.numVerticesStorage = 0;

    var triTableBuf = device.createBuffer({
        size: triTable.byteLength,
        usage: GPUBufferUsage.UNIFORM,
        mappedAtCreation: true,
    });
    new Int32Array(triTableBuf.getMappedRange()).set(triTable);
    triTableBuf.unmap();
    this.triTable = triTableBuf;

    this.computeBlockRangeBGLayout = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "storage",
                }
            },
            {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "uniform",
                }
            },
            {
                binding: 2,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "storage",
                }
            },
        ],
    });

    this.computeBlockRangePipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({
            bindGroupLayouts: [this.computeBlockRangeBGLayout],
        }),
        compute: {
            module: device.createShaderModule({
                code: zfp_compute_block_range_comp_spv,
            }),
            entryPoint: "main",
        },
    });

    this.decompressBlocksBGLayout = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "storage",
                }
            },
            {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "uniform",
                }
            },
            {
                binding: 2,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "storage",
                }
            },
            {
                binding: 3,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "storage",
                }
            },
            {
                binding: 4,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "storage",
                }
            },
        ],
    });
    this.ub1binding0BGLayout = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "uniform",
                }
            },
        ],
    });

    this.decompressBlocksPipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({
            bindGroupLayouts: [
                this.decompressBlocksBGLayout,
                this.ub1binding0BGLayout,
            ],
        }),
        compute: {
            module: device.createShaderModule({
                code: zfp_decompress_block_comp_spv,
            }),
            entryPoint: "main",
        },
    });

    // Set up compute initial rays pipeline
    this.viewParamBuf = device.createBuffer({
        size: 20 * 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // We'll need a max of canvas.width * canvas.height RayInfo structs in the buffer,
    // so just allocate it once up front
    this.rayInformationBuffer = device.createBuffer({
        size: canvas.width * canvas.size * 20,
        usage: GPUBufferUsage.STORAGE,
    });

    this.volumeInfoBuffer = this.device.createBuffer({
        size: 16 * 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
    });

    var computeInitialRaysBGLayout = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.VERTEX,
                buffer: {
                    type: "uniform",
                }
            },
            {
                binding: 1,
                visibility: GPUShaderStage.FRAGMENT,
                buffer: {
                    type: "uniform",
                }
            },
            {
                binding: 2,
                visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                buffer: {
                    type: "storage",
                }
            },
        ],
    });

    this.initialRaysBindGroup = device.createBindGroup({
        layout: computeInitialRaysBGLayout,
        entries: [
            {
                binding: 0,
                resource: {
                    buffer: this.viewParamBuf,
                },
            },
            {
                binding: 1,
                resource: {
                    buffer: this.rayInformationBuffer,
                },
            },
            {
                binding: 2,
                resource: {
                    buffer: this.volumeInfoBuffer,
                },
            },
        ],
    });

    // Specify vertex data for compute initial rays
    this.dataBuf = device.createBuffer({
        size: 12 * 3 * 3 * 4,
        usage: GPUBufferUsage.VERTEX,
        mappedAtCreation: true,
    });
    new Float32Array(dataBuf.getMappedRange()).set([
        1, 0, 0, 0, 0, 0, 1, 1, 0,

        0, 1, 0, 1, 1, 0, 0, 0, 0,

        1, 0, 1, 1, 0, 0, 1, 1, 1,

        1, 1, 0, 1, 1, 1, 1, 0, 0,

        0, 0, 1, 1, 0, 1, 0, 1, 1,

        1, 1, 1, 0, 1, 1, 1, 0, 1,

        0, 0, 0, 0, 0, 1, 0, 1, 0,

        0, 1, 1, 0, 1, 0, 0, 0, 1,

        1, 1, 0, 0, 1, 0, 1, 1, 1,

        0, 1, 1, 1, 1, 1, 0, 1, 0,

        0, 0, 1, 0, 0, 0, 1, 0, 1,

        1, 0, 0, 1, 0, 1, 0, 0, 0,
    ]);
    dataBuf.unmap();

    // Setup render outputs
    var swapChainFormat = "bgra8unorm";
    this.swapChain = context.configureSwapChain({
        device: device,
        format: swapChainFormat,
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });

    var depthFormat = "depth24plus-stencil8";
    var depthTexture = device.createTexture({
        size: {
            width: canvas.width,
            height: canvas.height,
            depth: 1,
        },
        format: depthFormat,
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });

    this.initialRaysPipeline = device.createRenderPipeline({
        layout: device.createPipelineLayout({
            bindGroupLayouts: [computeInitialRaysBGLayout],
        }),
        vertex: {
            module: device.createShaderModule({ code: compute_initial_rays_vert_spv }),
            entryPoint: "main",
            buffers: [
                {
                    arrayStride: 3 * 4,
                    attributes: [
                        {
                            format: "float32x3",
                            offset: 0,
                            shaderLocation: 0,
                        },
                    ],
                },
            ],
        },
        fragment: {
            module: device.createShaderModule({ code: compute_initial_rays_frag_spv }),
            entryPoint: "main",
            targets: [
                {
                    format: swapChainFormat,
                    blend: {
                        color: {
                            srcFactor: "one",
                            dstFactor: "one-minus-src-alpha",
                        },
                        alpha: {
                            srcFactor: "one",
                            dstFactor: "one-minus-src-alpha",
                        },
                    }
                },
            ],
        },
        primitive: {
            topology: 'triangle-list',
            cullMode: "front",
        },
        depthStencil: {
            format: depthFormat,
            depthWriteEnabled: true,
            depthCompare: "less",
        },
    });

    this.initialRaysPassDesc = {
        colorAttachments: [
            {
                view: undefined,
                loadValue: [0.3, 0.3, 0.3, 1],
            },
        ],
        depthStencilAttachment: {
            view: depthTexture.createView(),
            depthLoadValue: 1.0,
            depthStoreOp: "store",
            stencilLoadValue: 0,
            stencilStoreOp: "store",
        },
    };

};

VolumeRaycaster.prototype.setCompressedVolume =
    async function (volume, compressionRate, volumeDims, volumeScale) {
        // Upload the volume
        this.volumeDims = volumeDims;
        this.paddedDims = [
            alignTo(volumeDims[0], 4),
            alignTo(volumeDims[1], 4),
            alignTo(volumeDims[2], 4),
        ];
        this.totalBlocks = (this.paddedDims[0] * this.paddedDims[1] * this.paddedDims[2]) / 64;
        console.log(`total blocks ${this.totalBlocks}`);
        const groupThreadCount = 32;
        this.numWorkGroups = Math.ceil(this.totalBlocks / groupThreadCount);
        console.log(`num work groups ${this.numWorkGroups}`);
        console.log(`Cache initial size: ${Math.ceil(this.totalBlocks * 0.05)}`);

        this.lruCache = new LRUCache(this.device,
            this.scanPipeline,
            this.streamCompact,
            Math.ceil(this.totalBlocks * 0.05),
            64 * 4,
            this.totalBlocks);

        {
            var mapping = volumeInfoBuffer.getMappedRange();
            var maxBits = (1 << (2 * 3)) * compressionRate;
            var buf = new Uint32Array(mapping);
            buf.set(volumeDims);
            buf.set(this.paddedDims, 4);
            buf.set([maxBits], 12);
            buf.set([canvas.width], 14);

            var buf = new Float32Array(mapping);
            buf.set(volumeScale, 8);
        }
        volumeInfoBuffer.unmap();
        this.volumeInfoBuffer = volumeInfoBuffer;

        var compressedBuffer = this.device.createBuffer({
            size: volume.byteLength,
            usage: GPUBufferUsage.STORAGE,
            mappedAtCreation: true,
        });
        new Uint8Array(compressedBuffer.getMappedRange()).set(volume);
        compressedBuffer.unmap();
        this.compressedBuffer = compressedBuffer;
        this.compressedDataSize = volume.byteLength;

        this.uploadIsovalueBuf = this.device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC,
        });

        await this.computeBlockRanges();
    };

VolumeRaycaster.prototype.computeBlockRanges = async function () {
    // Note: this could be done by the server for us, but for this prototype
    // it's a bit easier to just do it here
    // Decompress each block and compute its value range, output to the blockRangesBuffer
    this.blockRangesBuffer = this.device.createBuffer({
        size: this.totalBlocks * 2 * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    var bindGroup = this.device.createBindGroup({
        layout: this.computeBlockRangeBGLayout,
        entries: [
            {
                binding: 0,
                resource: {
                    buffer: this.compressedBuffer,
                },
            },
            {
                binding: 1,
                resource: {
                    buffer: this.volumeInfoBuffer,
                },
            },
            {
                binding: 2,
                resource: {
                    buffer: this.blockRangesBuffer,
                },
            },
        ],
    });

    var commandEncoder = this.device.createCommandEncoder();
    var pass = commandEncoder.beginComputePass();

    // Decompress each block and compute its range
    pass.setPipeline(this.computeBlockRangePipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatch(this.numWorkGroups, 1, 1);

    pass.endPass();
    this.device.queue.submit([commandEncoder.finish()]);
};

VolumeRaycaster.prototype.computeInitialRays = async function (viewParamUpload) {
    this.initialRaysPassDesc.colorAttachments[0].view = this.swapChain
        .getCurrentTexture()
        .createView();

    var commandEncoder = device.createCommandEncoder();

    commandEncoder.copyBufferToBuffer(viewParamUpload, 0, this.viewParamBuf, 0, 20 * 4);

    var initialRaysPass = commandEncoder.beginRenderPass(this.initialRaysPassDesc);

    initialRaysPass.setPipeline(this.initialRaysPipeline);
    initialRaysPass.setVertexBuffer(0, this.dataBuf);
    initialRaysPass.setBindGroup(0, this.initialRaysBindGroup);
    initialRaysPass.draw(12 * 3, 1, 0, 0);

    initialRaysPass.endPass();
    device.queue.submit([commandEncoder.finish()]);
}

VolumeRaycaster.prototype.computeSurface = async function (isovalue, perfTracker) {
    console.log(`=====\nIsovalue = ${isovalue}`);
    // TODO: Conditionally free if memory use of VBO is very high to make
    // sure we don't OOM with some of our temp allocations?
    // This isn't quite enough to get miranda running on the 4GB VRAM surface
    // Adds about 100ms cost on RTX2070 on miranda
    /*
      if (this.vertexBuffer) {
          this.vertexBuffer.destroy();
          this.numVerticesStorage = 0;
      }
      */

    if (perfTracker.computeActiveBlocks === undefined) {
        perfTracker.computeActiveBlocks = [];
        perfTracker.numActiveBlocks = [];

        perfTracker.cacheUpdate = [];
        perfTracker.numBlocksDecompressed = [];
        perfTracker.decompression = [];

        perfTracker.compactActiveIDs = [];
        perfTracker.computeBlockHasVertices = [];

        perfTracker.numBlocksWithVertices = [];
        perfTracker.compactBlocksWithVertices = [];

        perfTracker.numVertices = [];
        perfTracker.computeVertices = [];
    }

    // Upload the isovalue
    await this.uploadIsovalueBuf.mapAsync(GPUMapMode.WRITE);
    new Float32Array(this.uploadIsovalueBuf.getMappedRange()).set([isovalue]);
    this.uploadIsovalueBuf.unmap();

    var commandEncoder = this.device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(this.uploadIsovalueBuf, 0, this.volumeInfoBuffer, 52, 4);
    this.device.queue.submit([commandEncoder.finish()]);

    // Compute list of active blocks
    var start = performance.now();
    var numActiveBlocks = await this.computeActiveBlocks();
    var end = performance.now();
    console.log(`Compute active took ${end - start}ms`);
    console.log(`# of active blocks = ${numActiveBlocks}`);

    perfTracker.computeActiveBlocks.push(end - start);
    perfTracker.numActiveBlocks.push(numActiveBlocks);

    if (numActiveBlocks == 0) {
        perfTracker.cacheUpdate.push(0);
        perfTracker.numBlocksDecompressed.push(0);
        perfTracker.decompression.push(0);
        perfTracker.compactActiveIDs.push(0);
        perfTracker.computeBlockHasVertices.push(0);
        perfTracker.numBlocksWithVertices.push(0);
        perfTracker.compactBlocksWithVertices.push(0);
        perfTracker.numVertices.push(0);
        perfTracker.computeVertices.push(0);
        return 0;
    }

    // Update cache to get offsets within it where we can decompress the new blocks we need to
    // cache
    var start = performance.now();
    // TODO: want a way to explicitly clear the cache so we can time an empty cache update
    // without having the first launch overhead also timed
    var [nBlocksToDecompress, decompressBlockIDs] =
        await this.lruCache.update(this.blockActiveBuffer, perfTracker);
    var end = performance.now();
    this.newDecompressed = nBlocksToDecompress;
    console.log(`# Blocks to decompress ${nBlocksToDecompress}`);
    console.log(`Cache update took ${end - start}ms`);
    perfTracker.cacheUpdate.push(end - start);
    perfTracker.numBlocksDecompressed.push(nBlocksToDecompress);

    if (numActiveBlocks > this.numActiveBlocksStorage) {
        this.numActiveBlocksStorage = Math.ceil(Math.min(
            this.totalBlocks, Math.max(numActiveBlocks, this.numActiveBlocksStorage * 1.5)));
        var scanAlignedSize = this.scanPipeline.getAlignedSize(this.numActiveBlocksStorage);

        // Explicitly release the old buffers first, so we don't have to wait for the GC
        // to free up the GPU memory
        if (this.blockHasVertices) {
            this.activeBlockIDs.destroy();
            this.blockHasVertices.destroy();
            this.blockHasVertsOffsets.destroy();
        }

        this.activeBlockIDs = this.device.createBuffer({
            size: this.numActiveBlocksStorage * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        this.blockHasVertices = this.device.createBuffer({
            size: this.numActiveBlocksStorage * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        this.blockHasVertsOffsets = this.device.createBuffer({
            size: scanAlignedSize * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });

        // Recreate the smaller scanners
        this.blockWithVerticesScanner =
            this.scanPipeline.prepareGPUInput(this.blockHasVertsOffsets, scanAlignedSize);
    }
    this.numActiveBlocks = numActiveBlocks;

    // Decompress the new blocks we need into the cache
    var start = performance.now();
    if (nBlocksToDecompress > 0) {
        await this.decompressBlocks(nBlocksToDecompress, decompressBlockIDs);
        decompressBlockIDs.destroy();
    }
    var end = performance.now();
    console.log(`Block decompression took ${end - start}`);
    perfTracker.decompression.push(end - start);

    // NOTE: we need to keep active block IDs and offsets as well, but as a separate thing
    // from the cache. The active block IDs & offsets are different from the cached ones,
    // which may not longer contain the surface
    // Compact the list of active block IDs
    var start = performance.now();
    await this.streamCompact.compactActiveIDs(this.totalBlocks,
        this.blockActiveBuffer,
        this.activeBlockOffsets,
        this.activeBlockIDs);
    var end = performance.now();
    console.log(`Compact active block IDs took ${end - start}ms`);
    perfTracker.compactActiveIDs.push(end - start);

    var start = performance.now();
    await this.computeBlockHasVertices();
    var end = performance.now();
    perfTracker.computeBlockHasVertices.push(end - start);

    // Pass over the blocks and filter out the ones which won't output vertices
    // Our compacted output will be the "active block index", which is the location of the
    // actual block id within the block_ids/block_offsets/vertex_offsets buffers
    var start = performance.now();
    var commandEncoder = this.device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(
        this.blockHasVertices, 0, this.blockHasVertsOffsets, 0, this.numActiveBlocks * 4);
    this.device.queue.submit([commandEncoder.finish()]);

    var numBlocksWithVertices = await this.blockWithVerticesScanner.scan(this.numActiveBlocks);
    console.log(
        `Of ${numActiveBlocks} active, only ${numBlocksWithVertices} will output vertices`);
    if (numBlocksWithVertices > this.numBlocksWithVerticesStorage) {
        this.numBlocksWithVerticesStorage = Math.floor(Math.min(
            this.totalBlocks,
            Math.max(numBlocksWithVertices, this.numBlocksWithVerticesStorage * 1.5)));
        var scanAlignedSize =
            this.scanPipeline.getAlignedSize(this.numBlocksWithVerticesStorage);

        if (this.blocksWithVertices) {
            this.blocksWithVertices.destroy();
            this.blockVertexOffsets.destroy();
        }
        this.blocksWithVertices = this.device.createBuffer({
            size: this.numBlocksWithVerticesStorage * 4,
            usage: GPUBufferUsage.STORAGE,
        });
        this.blockVertexOffsets = this.device.createBuffer({
            size: scanAlignedSize * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });

        this.blockVertexOffsetsScanner =
            this.scanPipeline.prepareGPUInput(this.blockVertexOffsets, scanAlignedSize);
    }
    this.numBlocksWithVertices = numBlocksWithVertices;

    await this.streamCompact.compactActiveIDs(this.numActiveBlocks,
        this.blockHasVertices,
        this.blockHasVertsOffsets,
        this.blocksWithVertices);
    var end = performance.now();
    console.log(`Active blocks w/ verts reduction took ${end - start}ms`);

    perfTracker.numBlocksWithVertices.push(numBlocksWithVertices);
    perfTracker.compactBlocksWithVertices.push(end - start);

    // For each block which will output vertices, compute the number of vertices its voxels
    // and the total for the block
    var start = performance.now();
    var numVertices = await this.computeBlockVertexCounts();
    var end = performance.now();
    console.log(`Vertex count computation took ${end - start}ms`);
    console.log(`# Vertices: ${numVertices}`);
    perfTracker.numVertices.push(numVertices);
    if (numVertices == 0) {
        perfTracker.computeVertices.push(0);
        return 0;
    }

    if (numVertices > this.numVerticesStorage) {
        this.numVerticesStorage =
            Math.floor(Math.max(numVertices, this.numVerticesStorage * 1.5));

        if (this.vertexBuffer) {
            this.vertexBuffer.destroy();
        }
        this.vertexBuffer = this.device.createBuffer({
            size: this.numVerticesStorage * 2 * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_SRC,
        });
    }
    this.numVertices = numVertices;

    // For each block:
    // Compute the active voxels within the block. We know there should be at least one,
    // since the block + neighboring halo voxels contain the isovalue
    // The active voxel IDs should be written to a single global buffer, so all blocks
    // basically appear as one giant volume When we do the scan here to compute the total
    // number of active voxels we need to keep each blocks offset in the compacted buffer and
    // its output number of voxels

    // Compute the number of vertices which will be output by each voxel in the blocks
    // This also needs to write to a global buffer which is shared across the blocks,
    // so that our output can be a single compact vertex buffer
    // So we need to do the same thing for the scan and tracking the offsets for each block
    // to start at when we compute it.
    var start = performance.now();
    await this.computeVertices();
    var end = performance.now();
    console.log(`Vertex computation took ${end - start}ms`);
    perfTracker.computeVertices.push(end - start);

    // Compute the vertices and output them to the single compacted buffer
    return numVertices;
};

VolumeRaycaster.prototype.decompressBlocks =
    async function (nBlocksToDecompress, decompressBlockIDs) {
        var decompressBlocksBG = this.device.createBindGroup({
            layout: this.decompressBlocksBGLayout,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: this.compressedBuffer,
                    },
                },
                {
                    binding: 1,
                    resource: {
                        buffer: this.volumeInfoBuffer,
                    },
                },
                {
                    binding: 2,
                    resource: {
                        buffer: this.lruCache.cache,
                    },
                },
                {
                    binding: 3,
                    resource: {
                        buffer: decompressBlockIDs,
                    },
                },
                {
                    binding: 4,
                    resource: {
                        buffer: this.lruCache.cachedItemSlots,
                    },
                },
            ],
        });

        var numChunks = Math.ceil(nBlocksToDecompress / this.maxDispatchSize);
        var dispatchChunkOffsetsBuf = this.device.createBuffer({
            size: numChunks * 256,
            usage: GPUBufferUsage.UNIFORM,
            mappedAtCreation: true,
        });
        var map = new Uint32Array(dispatchChunkOffsetsBuf.getMappedRange());
        for (var i = 0; i < numChunks; ++i) {
            map[i * 64] = i * this.maxDispatchSize;
        }
        dispatchChunkOffsetsBuf.unmap();

        // We execute these chunks in separate submissions to avoid having them
        // execute all at once and trigger a TDR if we're decompressing a large amount of data
        for (var i = 0; i < numChunks; ++i) {
            var numWorkGroups =
                Math.min(nBlocksToDecompress - i * this.maxDispatchSize, this.maxDispatchSize);
            var commandEncoder = this.device.createCommandEncoder();
            var pass = commandEncoder.beginComputePass();
            pass.setPipeline(this.decompressBlocksPipeline);
            pass.setBindGroup(0, decompressBlocksBG);
            // Have to create bind group here because dynamic offsets are not allowed
            var decompressBlocksStartOffsetBG = this.device.createBindGroup({
                layout: this.ub1binding0BGLayout,
                entries: [
                    {
                        binding: 0,
                        resource: {
                            buffer: dispatchChunkOffsetsBuf,
                            size: 4,
                            offset: i * 256,
                        },
                    },
                ],
            });
            pass.setBindGroup(1, decompressBlocksStartOffsetBG);
            pass.dispatch(numWorkGroups, 1, 1);
            pass.endPass();
            this.device.queue.submit([commandEncoder.finish()]);
        }
        await this.device.queue.onSubmittedWorkDone();
        dispatchChunkOffsetsBuf.destroy();
    };

VolumeRaycaster.prototype.computeBlockHasVertices = async function () {
    this.computeBlockVertexInfoBG = this.device.createBindGroup({
        layout: this.computeBlockVertsInfoBGLayout,
        entries: [
            {
                binding: 0,
                resource: {
                    buffer: this.volumeInfoBuffer,
                },
            },
            {
                binding: 1,
                resource: {
                    buffer: this.triTable,
                },
            },
            {
                binding: 2,
                resource: {
                    buffer: this.lruCache.cache,
                },
            },
            {
                binding: 3,
                resource: {
                    buffer: this.blockActiveBuffer,
                },
            },
            {
                binding: 4,
                resource: {
                    buffer: this.lruCache.cachedItemSlots,
                },
            },
            {
                binding: 5,
                resource: {
                    buffer: this.activeBlockIDs,
                },
            },
        ],
    });
    var blockHasVerticesBG = this.device.createBindGroup({
        layout: this.sb1Binding0BGLayout,
        entries: [
            {
                binding: 0,
                resource: {
                    buffer: this.blockHasVertices,
                },
            },
        ],
    });

    var commandEncoder = this.device.createCommandEncoder();
    var pass = commandEncoder.beginComputePass();
    pass.setPipeline(this.computeBlockHasVerticesPipeline);
    pass.setBindGroup(0, this.computeBlockVertexInfoBG);
    pass.setBindGroup(1, blockHasVerticesBG);
    pass.dispatch(this.numActiveBlocks, 1, 1);
    pass.endPass();
    this.device.queue.submit([commandEncoder.finish()]);

    await this.device.queue.onSubmittedWorkDone();
};

VolumeRaycaster.prototype.computeBlockVertexCounts = async function () {
    this.blockVertexOffsetsBG = this.device.createBindGroup({
        layout: this.sb2Binding01BGLayout,
        entries: [
            {
                binding: 0,
                resource: {
                    buffer: this.blockVertexOffsets,
                },
            },
            {
                binding: 1,
                resource: {
                    buffer: this.blocksWithVertices,
                },
            },
        ],
    });

    var commandEncoder = this.device.createCommandEncoder();
    var pass = commandEncoder.beginComputePass();
    pass.setPipeline(this.computeBlockVertexCountsPipeline);
    pass.setBindGroup(0, this.computeBlockVertexInfoBG);
    pass.setBindGroup(1, this.blockVertexOffsetsBG);
    pass.dispatch(this.numBlocksWithVertices, 1, 1);
    pass.endPass();
    this.device.queue.submit([commandEncoder.finish()]);

    var total = await this.blockVertexOffsetsScanner.scan(this.numBlocksWithVertices);
    return total;
};
