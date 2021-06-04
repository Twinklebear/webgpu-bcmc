var VolumeRaycaster = function(device, canvas_dims) {
    this.device = device;
    this.scanPipeline = new ExclusiveScanPipeline(device);
    this.streamCompact = new StreamCompact(device);
    this.numActiveBlocks = 0;
    this.numVertices = 0;
    this.numBlocksWithVertices = 0;

    this.canvas = document.getElementById("webgpu-canvas");
    var context = this.canvas.getContext("gpupresent");

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
        size: this.canvas.width * this.canvas.height * 20,
        usage: GPUBufferUsage.STORAGE,
    });

    this.computeInitialRaysBGLayout = device.createBindGroupLayout({
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
                    type: "storage",
                }
            },
            {
                binding: 2,
                visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                buffer: {
                    type: "uniform",
                }
            },
        ],
    });

    // Specify vertex data for compute initial rays
    this.dataBuf = device.createBuffer({
        size: 12 * 3 * 3 * 4,
        usage: GPUBufferUsage.VERTEX,
        mappedAtCreation: true,
    });
    new Float32Array(this.dataBuf.getMappedRange()).set([
        1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1,
        1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1,
        0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1,
        0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0,
    ]);
    this.dataBuf.unmap();

    // Setup render outputs
    var renderTargetFormat = "rgba8unorm";
    this.renderTarget = this.device.createTexture({
        size: [canvas_dims[0], canvas_dims[1], 1],
        format: renderTargetFormat,
        usage: GPUTextureUsage.STORAGE | GPUTextureUsage.RENDER_ATTACHMENT
    });

    this.initialRaysPipeline = device.createRenderPipeline({
        layout: device.createPipelineLayout({
            bindGroupLayouts: [this.computeInitialRaysBGLayout],
        }),
        vertex: {
            module: device.createShaderModule({code: compute_initial_rays_vert_spv}),
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
            module: device.createShaderModule({code: compute_initial_rays_frag_spv}),
            entryPoint: "main",
            targets: [
                {
                    format: renderTargetFormat,
                    // NOTE: allow writes for debugging
                    //writeMask: 0,
                },
            ],
        },
        primitive: {
            topology: 'triangle-list',
            cullMode: "front",
        }
    });

    this.initialRaysPassDesc = {
        colorAttachments: [
            {
                view: this.renderTarget.createView(),
                loadValue: [0.3, 0.3, 0.3, 1],
            },
        ]
    };

    this.macroTraverseBGLayout = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "uniform",
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
            }
        ],
    });

    this.macroTraversePipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({
            bindGroupLayouts: [
                this.macroTraverseBGLayout,
            ],
        }),
        compute: {
            module: device.createShaderModule({
                code: macro_traverse_comp_spv,
            }),
            entryPoint: "main",
        },
    });
};

VolumeRaycaster.prototype.setCompressedVolume =
    async function(volume, compressionRate, volumeDims, volumeScale) {
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

    this.volumeInfoBuffer = this.device.createBuffer({
        size: 16 * 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
    });
    {
        var mapping = this.volumeInfoBuffer.getMappedRange();
        var maxBits = (1 << (2 * 3)) * compressionRate;
        var buf = new Uint32Array(mapping);
        buf.set(volumeDims);
        buf.set(this.paddedDims, 4);
        buf.set([maxBits], 12);
        buf.set([this.canvas.width], 14);

        var buf = new Float32Array(mapping);
        buf.set(volumeScale, 8);
    }
    this.volumeInfoBuffer.unmap();

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

    this.initialRaysBindGroup = this.device.createBindGroup({
        layout: this.computeInitialRaysBGLayout,
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

    this.macroTraverseBindGroup = this.device.createBindGroup({
        layout: this.macroTraverseBGLayout,
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
                    buffer: this.viewParamBuf,
                },
            },
            {
                binding: 2,
                resource: {
                    buffer: this.blockRangesBuffer,
                },
            },
            {
                binding: 3,
                resource: {
                    buffer: this.rayInformationBuffer,
                },
            },
        ],
    });
};

VolumeRaycaster.prototype.computeBlockRanges = async function() {
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

VolumeRaycaster.prototype.computeInitialRays = async function(viewParamUpload) {
    console.log("Computing initial rays");
    var commandEncoder = this.device.createCommandEncoder();

    commandEncoder.copyBufferToBuffer(viewParamUpload, 0, this.viewParamBuf, 0, 20 * 4);

    var initialRaysPass = commandEncoder.beginRenderPass(this.initialRaysPassDesc);

    initialRaysPass.setPipeline(this.initialRaysPipeline);
    initialRaysPass.setVertexBuffer(0, this.dataBuf);
    initialRaysPass.setBindGroup(0, this.initialRaysBindGroup);
    initialRaysPass.draw(12 * 3, 1, 0, 0);

    initialRaysPass.endPass();
    this.device.queue.submit([commandEncoder.finish()]);
};

VolumeRaycaster.prototype.macroTraverse = async function(isovalue) {
    // Upload the isovalue
    await this.uploadIsovalueBuf.mapAsync(GPUMapMode.WRITE);
    new Float32Array(this.uploadIsovalueBuf.getMappedRange()).set([isovalue]);
    this.uploadIsovalueBuf.unmap();

    var commandEncoder = this.device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(this.uploadIsovalueBuf, 0, this.volumeInfoBuffer, 52, 4);

    var pass = commandEncoder.beginComputePass();

    // Decompress each block and compute its range
    pass.setPipeline(this.macroTraversePipeline);
    pass.setBindGroup(0, this.macroTraverseBindGroup);
    pass.dispatch(this.canvas.width, this.canvas.height, 1);

    pass.endPass();
    this.device.queue.submit([commandEncoder.finish()]);
};

VolumeRaycaster.prototype.decompressBlocks =
    async function(nBlocksToDecompress, decompressBlockIDs) {
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
