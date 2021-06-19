var VolumeRaycaster = function (device, canvas) {
    this.device = device;
    this.scanPipeline = new ExclusiveScanPipeline(device);
    this.streamCompact = new StreamCompact(device);
    this.numActiveBlocks = 0;
    this.renderComplete = false;

    this.canvas = canvas;
    console.log(`canvas size ${canvas.width}x${canvas.height}`);

    // Max dispatch size for more computationally heavy kernels
    // which might hit TDR on lower power devices
    this.maxDispatchSize = 512000;

    this.numActiveBlocksStorage = 0;

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
        size: this.canvas.width * this.canvas.height * 32,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    this.resetRaysBGLayout = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } }
        ]
    });

    this.resetRaysPipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({
            bindGroupLayouts: [
                this.resetRaysBGLayout,
            ],
        }),
        compute: {
            module: device.createShaderModule({
                code: reset_rays_comp_spv,
            }),
            entryPoint: "main",
        },
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
        size: [this.canvas.width, this.canvas.height, 1],
        format: renderTargetFormat,
        usage: GPUTextureUsage.STORAGE | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC
    });

    this.initialRaysPipeline = device.createRenderPipeline({
        layout: device.createPipelineLayout({
            bindGroupLayouts: [this.computeInitialRaysBGLayout],
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
                    format: renderTargetFormat,
                    // NOTE: allow writes for debugging
                    // writeMask: 0,
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
                loadValue: [1.0, 1.0, 1.0, 1],
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
            },
            {
                // Also pass the render target for debugging
                binding: 4,
                visibility: GPUShaderStage.COMPUTE,
                storageTexture: { access: "write-only", format: renderTargetFormat }
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

    this.resetBlockActiveBGLayout = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }
        ]
    });
    this.resetBlockActivePipeline = device.createComputePipeline({
        layout:
            device.createPipelineLayout({ bindGroupLayouts: [this.resetBlockActiveBGLayout] }),
        compute: {
            module: device.createShaderModule({ code: reset_block_active_comp_spv }),
            entryPoint: "main"
        }
    });

    this.resetBlockNumRaysBGLayout = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }
        ]
    });
    this.resetBlockNumRaysPipeline = device.createComputePipeline({
        layout:
            device.createPipelineLayout({ bindGroupLayouts: [this.resetBlockNumRaysBGLayout] }),
        compute: {
            module: device.createShaderModule({ code: reset_block_num_rays_comp_spv }),
            entryPoint: "main"
        }
    });

    this.LODThresholdBuf = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.markBlockActiveBGLayout = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        ]
    });
    this.markBlockActivePipeline = device.createComputePipeline({
        layout:
            device.createPipelineLayout({ bindGroupLayouts: [this.markBlockActiveBGLayout] }),
        compute: {
            module: device.createShaderModule({ code: mark_block_active_comp_spv }),
            entryPoint: "main"
        }
    });

    this.debugViewBlockRayCountsBGLayout = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            {
                binding: 3,
                visibility: GPUShaderStage.COMPUTE,
                storageTexture: { access: "write-only", format: renderTargetFormat }
            }
        ]
    });
    this.debugViewBlockRayCountsPipeline = device.createComputePipeline({
        layout: device.createPipelineLayout(
            { bindGroupLayouts: [this.debugViewBlockRayCountsBGLayout] }),
        compute: {
            module: device.createShaderModule({ code: debug_view_rays_per_block_comp_spv }),
            entryPoint: "main"
        }
    });

    // Intermediate buffers for sorting ray IDs using their block ID as the key
    this.radixSorter = new RadixSorter(device);
    this.rayIDBuffer = device.createBuffer({
        size: this.radixSorter.getAlignedSize(this.canvas.width * this.canvas.height) * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this.rayBlockIDBuffer = device.createBuffer({
        size: this.radixSorter.getAlignedSize(this.canvas.width * this.canvas.height) * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this.compactRayBlockIDBuffer = device.createBuffer({
        size: this.radixSorter.getAlignedSize(this.canvas.width * this.canvas.height) * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    this.rayActiveBuffer = device.createBuffer({
        size: this.canvas.width * this.canvas.height * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    this.rayActiveCompactOffsetBuffer = device.createBuffer({
        size: this.scanPipeline.getAlignedSize(this.canvas.width * this.canvas.height) * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this.scanRayActive = this.scanPipeline.prepareGPUInput(
        this.rayActiveCompactOffsetBuffer,
        this.scanPipeline.getAlignedSize(this.canvas.width * this.canvas.height));

    this.writeRayAndBlockIDBGLayout = device.createBindGroupLayout({
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
                    type: "storage",
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
    this.writeRayAndBlockIDPipeline = device.createComputePipeline({
        layout:
            device.createPipelineLayout({ bindGroupLayouts: [this.writeRayAndBlockIDBGLayout] }),
        compute: {
            module: device.createShaderModule({ code: write_ray_and_block_id_comp_spv }),
            entryPoint: "main",
        }
    });

    this.combineBlockInformationBGLayout = device.createBindGroupLayout({
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
                    type: "storage",
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
        ]
    });
    this.combineBlockInformationPipeline = device.createComputePipeline({
        layout: device.createPipelineLayout(
            { bindGroupLayouts: [this.combineBlockInformationBGLayout] }),
        compute: {
            module: device.createShaderModule({ code: combine_block_information_comp_spv }),
            entryPoint: "main"
        }
    });

    this.rtBlocksPipelineBG0Layout = device.createBindGroupLayout({
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
                    type: "storage",
                }
            },
            {
                binding: 2,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "storage",
                }
            },
        ]
    });
    this.rtBlocksPipelineBG1Layout = device.createBindGroupLayout({
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
                    type: "storage",
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
                storageTexture: { access: "write-only", format: renderTargetFormat }
            },
            {
                binding: 5,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "storage",
                }
            },
            {
                binding: 6,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "uniform",
                }
            },
        ]
    });

    this.raytraceBlocksPipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({
            bindGroupLayouts: [this.rtBlocksPipelineBG0Layout, this.rtBlocksPipelineBG1Layout]
        }),
        compute: {
            module: device.createShaderModule({ code: raytrace_active_block_comp_spv }),
            entryPoint: "main"
        }
    });
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
        this.blockGridDims =
            [this.paddedDims[0] / 4, this.paddedDims[1] / 4, this.paddedDims[2] / 4];
        this.totalBlocks = (this.paddedDims[0] * this.paddedDims[1] * this.paddedDims[2]) / 64;

        console.log(`total blocks ${this.totalBlocks}`);
        const groupThreadCount = 32;
        this.numWorkGroups = Math.ceil(this.totalBlocks / groupThreadCount);
        console.log(`num work groups ${this.numWorkGroups}`);
        var cacheInitialSize = Math.ceil(this.totalBlocks * 0.05);
        console.log(`Cache initial size: ${cacheInitialSize}`);

        this.lruCache = new LRUCache(this.device,
            this.scanPipeline,
            this.streamCompact,
            cacheInitialSize,
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

        this.uploadLODBuf = this.device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC,
        });

        await this.computeBlockRanges();

        this.blockActiveBuffer = this.device.createBuffer(
            { size: 4 * this.totalBlocks, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });

        this.blockVisibleBuffer = this.device.createBuffer(
            { size: 4 * this.totalBlocks, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });

        this.blockNumRaysBuffer = this.device.createBuffer(
            { size: 4 * this.totalBlocks, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });

        // Scan result buffer for the block ray offsets (computed by scanning the result in
        // blockNumRaysBuffer)
        this.blockRayOffsetBuffer = this.device.createBuffer({
            size: 4 * this.scanPipeline.getAlignedSize(this.totalBlocks),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        });
        this.scanBlockRayOffsets = this.scanPipeline.prepareGPUInput(
            this.blockRayOffsetBuffer, this.scanPipeline.getAlignedSize(this.totalBlocks));

        // Buffers for use when compacting down IDs of the active blocks
        // TODO: could do a bit better and filter neighbor blocks out too by just marking
        // visible blocks in a separate buffer (i.e., ones a ray is immediately in).
        // This set will include neighbors for now
        this.blockActiveCompactOffsetBuffer = this.device.createBuffer({
            size: 4 * this.scanPipeline.getAlignedSize(this.totalBlocks),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        });
        this.activeBlockIDBuffer = this.device.createBuffer({
            size: 4 * this.totalBlocks,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        });

        this.scanBlockActiveOffsets =
            this.scanPipeline.prepareGPUInput(this.blockActiveCompactOffsetBuffer,
                this.scanPipeline.getAlignedSize(this.totalBlocks));

        this.resetRaysBG = this.device.createBindGroup({
            layout: this.resetRaysBGLayout,
            entries: [
                { binding: 0, resource: { buffer: this.rayInformationBuffer } },
                { binding: 1, resource: { buffer: this.volumeInfoBuffer } },
            ]
        });

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
                { binding: 4, resource: this.renderTarget.createView() }
            ],
        });

        this.resetBlockActiveBG = this.device.createBindGroup({
            layout: this.resetBlockActiveBGLayout,
            entries: [
                { binding: 0, resource: { buffer: this.volumeInfoBuffer } },
                { binding: 1, resource: { buffer: this.blockActiveBuffer } },
                { binding: 2, resource: { buffer: this.blockVisibleBuffer } },
            ]
        });
        this.resetBlockNumRaysBG = this.device.createBindGroup({
            layout: this.resetBlockNumRaysBGLayout,
            entries: [
                { binding: 0, resource: { buffer: this.volumeInfoBuffer } },
                { binding: 1, resource: { buffer: this.blockNumRaysBuffer } }
            ]
        });

        this.markBlockActiveBG = this.device.createBindGroup({
            layout: this.markBlockActiveBGLayout,
            entries: [
                { binding: 0, resource: { buffer: this.volumeInfoBuffer } },
                { binding: 1, resource: { buffer: this.blockActiveBuffer } },
                { binding: 2, resource: { buffer: this.blockNumRaysBuffer } },
                { binding: 3, resource: { buffer: this.rayInformationBuffer } },
                { binding: 4, resource: { buffer: this.blockVisibleBuffer } },
                { binding: 5, resource: { buffer: this.LODThresholdBuf } },
            ]
        });

        this.writeRayAndBlockIDBG = this.device.createBindGroup({
            layout: this.writeRayAndBlockIDBGLayout,
            entries: [
                { binding: 0, resource: { buffer: this.volumeInfoBuffer } },
                { binding: 1, resource: { buffer: this.rayInformationBuffer } },
                { binding: 2, resource: { buffer: this.rayBlockIDBuffer } },
                { binding: 3, resource: { buffer: this.rayActiveBuffer } },
            ]
        });

        this.combinedBlockInformationBuffer = this.device.createBuffer(
            { size: this.totalBlocks * 4 * 4, usage: GPUBufferUsage.STORAGE });

        this.combineBlockInformationBG = this.device.createBindGroup({
            layout: this.combineBlockInformationBGLayout,
            entries: [
                { binding: 0, resource: { buffer: this.combinedBlockInformationBuffer } },
                { binding: 1, resource: { buffer: this.activeBlockIDBuffer } },
                { binding: 2, resource: { buffer: this.blockRayOffsetBuffer } },
                { binding: 3, resource: { buffer: this.blockNumRaysBuffer } },
            ]
        });

        this.rtBlocksPipelineBG1 = this.device.createBindGroup({
            layout: this.rtBlocksPipelineBG1Layout,
            entries: [
                { binding: 0, resource: { buffer: this.viewParamBuf } },
                { binding: 1, resource: { buffer: this.rayInformationBuffer } },
                { binding: 2, resource: { buffer: this.rayIDBuffer } },
                { binding: 3, resource: { buffer: this.combinedBlockInformationBuffer } },
                { binding: 4, resource: this.renderTarget.createView() },
                { binding: 5, resource: { buffer: this.blockRangesBuffer } },
                { binding: 6, resource: { buffer: this.LODThresholdBuf } },
            ]
        });
    };

VolumeRaycaster.prototype.computeBlockRanges = async function () {
    // Note: this could be done by the server for us, but for this prototype
    // it's a bit easier to just do it here
    // Decompress each block and compute its value range, output to the blockRangesBuffer
    this.blockRangesBuffer = this.device.createBuffer({
        size: this.totalBlocks * 10 * 4,
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

// Progressively compute the surface, returns true when rendering is complete
VolumeRaycaster.prototype.renderSurface =
    async function (isovalue, LODThreshold, viewParamUpload, perfTracker, renderParamsChanged) {
        if (this.renderComplete && !renderParamsChanged) {
            return this.renderComplete;
        }
        console.log("===== Rendering Surface =======");

        if (renderParamsChanged) {
            console.log(`Render params changed, LOD: ${LODThreshold}`);
            // Upload the isovalue
            await this.uploadIsovalueBuf.mapAsync(GPUMapMode.WRITE);
            new Float32Array(this.uploadIsovalueBuf.getMappedRange()).set([isovalue]);
            this.uploadIsovalueBuf.unmap();

            // Upload new LOD threshold
            var commandEncoder = this.device.createCommandEncoder();
            await this.uploadLODBuf.mapAsync(GPUMapMode.WRITE);
            new Uint32Array(this.uploadLODBuf.getMappedRange()).set([LODThreshold]);
            this.uploadLODBuf.unmap();
            commandEncoder.copyBufferToBuffer(this.uploadLODBuf, 0, this.LODThresholdBuf, 0, 4);

            // Reset active blocks for the new viewpoint/isovalue to allow eviction of old blocks
            var pass = commandEncoder.beginComputePass();
            pass.setPipeline(this.resetBlockActivePipeline);
            pass.setBindGroup(0, this.resetBlockActiveBG);
            pass.dispatch(this.blockGridDims[0], this.blockGridDims[1], this.blockGridDims[2]);
            pass.endPass();
            this.device.queue.submit([commandEncoder.finish()]);

            await this.computeInitialRays(viewParamUpload);

            this.totalPassTime = 0;
            this.numPasses = 0;
        }
        // for (var i = 0; i < 50; ++i) {
        console.log(`++++ Surface pass ${this.numPasses} ++++`);
        var startPass = performance.now();

        var start = performance.now();
        await this.macroTraverse();
        var end = performance.now();
        console.log(`Macro Traverse: ${end - start}ms`);

        start = performance.now();
        await this.markActiveBlocks();
        end = performance.now();
        console.log(`Mark Active Blocks: ${end - start}ms`);

        // Decompress any new blocks needed for the pass
        start = performance.now();
        var [nBlocksToDecompress, decompressBlockIDs] =
            await this.lruCache.update(this.blockActiveBuffer, perfTracker);
        end = performance.now();
        console.log(`LRU: ${end - start}ms`);
        if (nBlocksToDecompress != 0) {
            console.log(`Will decompress ${nBlocksToDecompress} blocks`);
            start = performance.now();
            await this.decompressBlocks(nBlocksToDecompress, decompressBlockIDs);
            end = performance.now();
            console.log(`Decompress: ${end - start}ms`);
        }

        start = performance.now();
        var numRaysActive = await this.computeBlockRayOffsets();
        end = performance.now();
        console.log(`Ray active and offsets: ${end - start}ms`);
        console.log(`numRaysActive = ${numRaysActive}`);
        if (numRaysActive > 0) {
            start = performance.now();
            var numActiveBlocks = await this.sortActiveRaysByBlock(numRaysActive);
            end = performance.now();
            console.log(`Sort active rays by block: ${end - start}ms`);

            start = performance.now();
            await this.raytraceVisibleBlocks(numActiveBlocks);
            end = performance.now();
            console.log(`Raytrace blocks: ${end - start}ms`);
            console.log(`PASS TOOK: ${end - startPass}ms`);
            console.log(`++++++++++`);
        }
        this.totalPassTime += end - startPass;
        this.numPasses += 1;
        //}
        this.renderComplete = numRaysActive == 0;
        if (this.renderComplete) {
            console.log(`Avg time per pass ${this.totalPassTime / this.numPasses}ms`);
        }
        return this.renderComplete;
    };

// Reset the rays and compute the initial set of rays that intersect the volume
// Rays that miss the volume will have:
// dir = vec3(0)
// block_id = UINT_MAX
// t = FLT_MAX
//
// Rays that hit the volume will have valid directions and t values
VolumeRaycaster.prototype.computeInitialRays = async function (viewParamUpload) {
    var commandEncoder = this.device.createCommandEncoder();

    commandEncoder.copyBufferToBuffer(viewParamUpload, 0, this.viewParamBuf, 0, 20 * 4);

    var resetRaysPass = commandEncoder.beginComputePass(this.resetRaysPipeline);
    resetRaysPass.setBindGroup(0, this.resetRaysBG);
    resetRaysPass.setPipeline(this.resetRaysPipeline);
    resetRaysPass.dispatch(this.canvas.width, this.canvas.height, 1);
    resetRaysPass.endPass();

    var initialRaysPass = commandEncoder.beginRenderPass(this.initialRaysPassDesc);

    initialRaysPass.setPipeline(this.initialRaysPipeline);
    initialRaysPass.setVertexBuffer(0, this.dataBuf);
    initialRaysPass.setBindGroup(0, this.initialRaysBindGroup);
    initialRaysPass.draw(12 * 3, 1, 0, 0);

    initialRaysPass.endPass();
    this.device.queue.submit([commandEncoder.finish()]);
};

// Step the active rays forward in the macrocell grid, updating block_id and t.
// Rays that exit the volume will have block_id = UINT_MAX and t = FLT_MAX
VolumeRaycaster.prototype.macroTraverse = async function () {
    // TODO: Would it be worth doing a scan and compact to find just the IDs of the currently
    // active rays? then only advancing them? We'll do that anyways so it would tell us when
    // we're done too (no rays active)

    var commandEncoder = this.device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(this.uploadIsovalueBuf, 0, this.volumeInfoBuffer, 52, 4);

    var pass = commandEncoder.beginComputePass();

    pass.setPipeline(this.macroTraversePipeline);
    pass.setBindGroup(0, this.macroTraverseBindGroup);
    pass.dispatch(this.canvas.width, this.canvas.height, 1);

    pass.endPass();
    this.device.queue.submit([commandEncoder.finish()]);
};

// Mark the active blocks for the current viewpoint/isovalue and count the # of rays
// that we need to process for each block
VolumeRaycaster.prototype.markActiveBlocks = async function () {
    var commandEncoder = this.device.createCommandEncoder();
    var pass = commandEncoder.beginComputePass();

    // Reset the # of rays for each block
    pass.setPipeline(this.resetBlockNumRaysPipeline);
    pass.setBindGroup(0, this.resetBlockNumRaysBG);
    pass.dispatch(this.blockGridDims[0], this.blockGridDims[1], this.blockGridDims[2]);

    // Compute which blocks are active and how many rays each has
    pass.setPipeline(this.markBlockActivePipeline);
    pass.setBindGroup(0, this.markBlockActiveBG);
    pass.dispatch(this.canvas.width, this.canvas.height, 1);

    // Debugging: view # rays per block
    /*
    {
        var debugViewBlockRayCountsBG = this.device.createBindGroup({
            layout: this.debugViewBlockRayCountsBGLayout,
            entries: [
                {binding: 0, resource: {buffer: this.volumeInfoBuffer}},
                {binding: 1, resource: {buffer: this.blockNumRaysBuffer}},
                {binding: 2, resource: {buffer: this.rayInformationBuffer}},
                {binding: 3, resource: this.renderTarget.createView()}
            ]
        });
        pass.setPipeline(this.debugViewBlockRayCountsPipeline);
        pass.setBindGroup(0, debugViewBlockRayCountsBG);
        pass.dispatch(this.canvas.width, this.canvas.height, 1);
    }
    */
    pass.endPass();

    this.device.queue.submit([commandEncoder.finish()]);
};

// Scan the blockNumRaysBuffer storing the output in blockRayOffsetBuffer and
// return the number of active rays.
VolumeRaycaster.prototype.computeBlockRayOffsets = async function () {
    var commandEncoder = this.device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(
        this.blockNumRaysBuffer, 0, this.blockRayOffsetBuffer, 0, 4 * this.totalBlocks);
    this.device.queue.submit([commandEncoder.finish()]);

    return await this.scanBlockRayOffsets.scan(this.totalBlocks);
};

// Sort the active ray IDs by their block ID in ascending order (inactive rays will be at the
// end).
VolumeRaycaster.prototype.sortActiveRaysByBlock = async function (numRaysActive) {
    // Populate the ray ID, ray block ID and ray active buffers
    var commandEncoder = this.device.createCommandEncoder();
    var pass = commandEncoder.beginComputePass()
    pass.setPipeline(this.writeRayAndBlockIDPipeline);
    pass.setBindGroup(0, this.writeRayAndBlockIDBG);
    pass.dispatch(this.canvas.width, this.canvas.height, 1);
    pass.endPass();

    // We scan the rayActiveCompactOffsetBuffer, so copy the ray active information over
    commandEncoder.copyBufferToBuffer(this.rayActiveBuffer,
        0,
        this.rayActiveCompactOffsetBuffer,
        0,
        this.canvas.width * this.canvas.height * 4);

    // We also scan the active block buffer to produce offsets for compacting active block IDs
    // down This will let us reduce the dispatch size of the ray tracing step to just active
    // blocks
    commandEncoder.copyBufferToBuffer(this.blockVisibleBuffer,
        0,
        this.blockActiveCompactOffsetBuffer,
        0,
        this.totalBlocks * 4);
    this.device.queue.submit([commandEncoder.finish()]);

    // Scan the active ray buffer and compact the active ray IDs before we sort
    // so that the sort doesn't have to process such a large number of items
    var nactive = await this.scanRayActive.scan(this.canvas.width * this.canvas.height);
    // Should match numRaysActive, sanity check here
    if (numRaysActive != nactive) {
        console.log(`nactive ${nactive} doesn't match numRaysActive ${numRaysActive}!?`);
    }
    var startCompacts = performance.now();
    // Compact the active ray IDs and their block IDs down
    await this.streamCompact.compactActiveIDs(this.canvas.width * this.canvas.height,
        this.rayActiveBuffer,
        this.rayActiveCompactOffsetBuffer,
        this.rayIDBuffer);

    await this.streamCompact.compactActive(this.canvas.width * this.canvas.height,
        this.rayActiveBuffer,
        this.rayActiveCompactOffsetBuffer,
        this.rayBlockIDBuffer,
        this.compactRayBlockIDBuffer);

    // Compact the active block IDs down as well
    var numActiveBlocks = await this.scanBlockActiveOffsets.scan(this.totalBlocks);
    await this.streamCompact.compactActiveIDs(this.totalBlocks,
        this.blockVisibleBuffer,
        this.blockActiveCompactOffsetBuffer,
        this.activeBlockIDBuffer);
    var endCompacts = performance.now();
    console.log(`sortActiveRaysByBlock: Compacts ${endCompacts - startCompacts}ms`);

    var start = performance.now();
    // Sort active ray IDs by their block ID
    await this.radixSorter.sort(
        this.compactRayBlockIDBuffer, this.rayIDBuffer, numRaysActive, false);
    var end = performance.now();
    console.log(`sortActiveRaysByBlock: Sort rays by blocks: ${end - start}ms`);

    /*
    {
        var debugReadbackBlock = this.device.createBuffer({
            size: numRaysActive * 4,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
        var debugReadbackRay = this.device.createBuffer({
            size: numRaysActive * 4,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
        var debugReadbackRayInformation = this.device.createBuffer({
            size: this.canvas.width * this.canvas.height * 32,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });

        var commandEncoder = this.device.createCommandEncoder();
        commandEncoder.copyBufferToBuffer(
            this.compactRayBlockIDBuffer, 0, debugReadbackBlock, 0, numRaysActive * 4);
        commandEncoder.copyBufferToBuffer(
            this.rayIDBuffer, 0, debugReadbackRay, 0, numRaysActive * 4);
        commandEncoder.copyBufferToBuffer(this.rayInformationBuffer,
                                          0,
                                          debugReadbackRayInformation,
                                          0,
                                          this.canvas.width * this.canvas.height * 32);
        await this.device.queue.submit([commandEncoder.finish()]);

        await debugReadbackBlock.mapAsync(GPUMapMode.READ);
        await debugReadbackRay.mapAsync(GPUMapMode.READ);
        await debugReadbackRayInformation.mapAsync(GPUMapMode.READ);

        var blocks = new Uint32Array(debugReadbackBlock.getMappedRange());
        var rays = new Uint32Array(debugReadbackRay.getMappedRange());
        var rayInfoMapped = debugReadbackRayInformation.getMappedRange();
        var rayInformationFloat = new Float32Array(rayInfoMapped);
        var rayInformationInt = new Float32Array(rayInfoMapped);

        var blockRayCounts = {};
        for (var i = 0; i < numRaysActive; ++i) {
            if (!(blocks[i] in blockRayCounts)) {
                blockRayCounts[blocks[i]] = [rays[i]];
            } else {
                blockRayCounts[blocks[i]].push(rays[i]);
            }
        }
        // size of a ray in floats/u32's
        var sizeofRay = 32 / 4;
        console.log(blockRayCounts);
        for (var ids in blockRayCounts) {
            for (var i = 0; i < ids.length; ++i) {
                var rstart = rays[i] * sizeofRay;
                var dir = [
                    rayInformationFloat[rstart],
                    rayInformationFloat[rstart + 1],
                    rayInformationFloat[rstart + 2]
                ];
                var block_id = rayInformationInt[rstart + 3];
                var t = rayInformationFloat[rstart + 4];
                var t_next = rayInformationFloat[rstart + 5];
                console.log(`Ray ${rays[i]}: dir=${dir}, block_id=${block_id}, t=${
                    t}, t_next=${t_next}`);
            }
        }

        debugReadbackBlock.unmap();
        debugReadbackRay.unmap();
        debugReadbackRayInformation.unmap();

        debugReadbackBlock.destroy();
        debugReadbackRay.destroy();
        debugReadbackRayInformation.destroy();
    }
    */

    return numActiveBlocks;
};

VolumeRaycaster.prototype.raytraceVisibleBlocks = async function (numActiveBlocks) {
    console.log(`Raytracing ${numActiveBlocks} blocks`);

    // Must recreate each time b/c cache buffer will grow
    var rtBlocksPipelineBG0 = this.device.createBindGroup({
        layout: this.rtBlocksPipelineBG0Layout,
        entries: [
            { binding: 0, resource: { buffer: this.volumeInfoBuffer } },
            { binding: 1, resource: { buffer: this.lruCache.cache } },
            { binding: 2, resource: { buffer: this.lruCache.cachedItemSlots } },
        ]
    });

    var commandEncoder = this.device.createCommandEncoder();
    var pass = commandEncoder.beginComputePass();

    // First make the combined block information buffer (to fit in 6 storage buffers)
    // The limit will be bumped up to 8 so we could remove this piece in a bit once
    // the change lands in Chromium
    pass.setPipeline(this.combineBlockInformationPipeline);
    pass.setBindGroup(0, this.combineBlockInformationBG);
    pass.dispatch(numActiveBlocks, 1, 1);

    // TODO: Might be worth for data sets where many blocks
    // project to a lot of pixels to split up the dispatches,
    // and do multiple dispatch indirect for the blocks touching
    // many pixels so that we don't serialize so badly on them, while
    // still doing a single dispatch for all blocks touching <= 64 pixels
    // since that will be most of the blocks, especially for large data.
    pass.setPipeline(this.raytraceBlocksPipeline);
    pass.setBindGroup(0, rtBlocksPipelineBG0);
    pass.setBindGroup(1, this.rtBlocksPipelineBG1);
    pass.dispatch(numActiveBlocks, 1, 1);
    pass.endPass();

    await this.device.queue.submit([commandEncoder.finish()]);
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
