var VolumeRaycaster = function(
    device, width, height, recordVisibleBlocksUI, enableSpeculationUI) {
    this.device = device;
    this.scanPipeline = new ExclusiveScanPipeline(device);
    this.streamCompact = new StreamCompact(device);
    // Number of blocks visible for the current pass
    this.numVisibleBlocks = 0;
    // Number of blocks decompressed for the current pass
    this.newDecompressed = 0;

    this.renderComplete = false;

    this.width = width;
    this.height = height;

    // Record visible blocks will optionally track the total % of blocks that were active or
    // visible while rendering the surface. This is just needed to provide this statistic for
    // the paper and is computed on the host by just reading back the blockVisible buffer and
    // or'ing it with the previous pass's one to accumulate the total block visible list
    // without double-counting
    this.recordVisibleBlocksUI = recordVisibleBlocksUI;

    // For testing/demo/benchmarking of enable/disable speculation
    this.enableSpeculationUI = enableSpeculationUI;

    // Each pass appends its performance stats to the perfStats array,
    // this is reset for each new isovalue
    this.perfStats = [];

    // Max dispatch size for more computationally heavy kernels
    // which might hit TDR on lower power devices
    this.maxDispatchSize = device.limits.maxComputeWorkgroupsPerDimension;

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
    this.pushConstantS1B0DynamicLayout = device.createBindGroupLayout({
        entries: [{
            binding: 0,
            visibility: GPUShaderStage.COMPUTE,
            buffer: {type: "uniform", hasDynamicOffset: true}
        }]
    });
    this.computeBlockRangePipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({
            bindGroupLayouts:
                [this.computeBlockRangeBGLayout, this.pushConstantS1B0DynamicLayout],
        }),
        compute: {
            module: device.createShaderModule({
                code: zfp_compute_block_range_comp_spv,
            }),
            entryPoint: "main",
        },
    });

    this.computeVoxelRangeBGLayout = device.createBindGroupLayout({
        entries: [{
            binding: 0,
            visibility: GPUShaderStage.COMPUTE,
            buffer: {
                type: "storage",
            }
        }]
    });
    this.computeVoxelRangePipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({
            bindGroupLayouts: [
                this.computeBlockRangeBGLayout,
                this.pushConstantS1B0DynamicLayout,
                this.computeVoxelRangeBGLayout
            ]
        }),
        compute: {
            module: device.createShaderModule({
                code: compute_voxel_range_comp_spv,
            }),
            entryPoint: "main",
        }
    });

    this.computeCoarseCellRangeBGLayout = device.createBindGroupLayout({
        entries: [
            {binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: {type: "uniform"}},
            {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {type: "uniform", hasDynamicOffset: true}
            },
            {binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: {type: "storage"}},
            {binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: {type: "storage"}}
        ]
    });
    this.computeCoarseCellRangePipeline = device.createComputePipeline({
        layout: device.createPipelineLayout(
            {bindGroupLayouts: [this.computeCoarseCellRangeBGLayout]}),
        compute: {
            module: device.createShaderModule({code: compute_coarse_cell_range_comp_spv}),
            entryPoint: "main"
        }
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
        // mat4, 2 vec4's, a float, 2 uint + some extra to align
        size: 32 * 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // We'll need a max of width * height RayInfo structs in the buffer,
    // so just allocate it once up front
    this.rayInformationBuffer = device.createBuffer({
        size: this.width * this.height * 16,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // We need width * height RayIDs for speculation,
    // with ray indices repeated as speculation occurs
    this.speculativeRayIDBuffer = device.createBuffer({
        size: this.width * this.height * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    this.rayRGBZBuffer = device.createBuffer({
        size: this.width * this.height * 2 * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // Each ray stores 2 iterator states, the coarse one followed by the fine one.
    // Each state is 32b
    this.gridIteratorBuffer = device.createBuffer({
        size: this.width * this.height * 8 * 4,
        usage: GPUBufferUsage.STORAGE,
    });

    this.resetRaysBGLayout = device.createBindGroupLayout({
        entries: [
            {binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: {type: "storage"}},
            {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: {type: "uniform"}},
            {binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: {type: "storage"}},
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

    // this.initSpeculativeIDsBGLayout = device.createBindGroupLayout({
    //     entries: [
    //         {binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: {type: "uniform"}},
    //         {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: {type: "uniform"}},
    //         {binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: {type: "storage"}},
    //         {binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: {type: "storage"}}
    //     ]
    // });

    // this.initSpeculativeIDsPipeline = device.createComputePipeline({
    //     layout: device.createPipelineLayout({
    //         bindGroupLayouts: [
    //             this.initSpeculativeIDsBGLayout,
    //         ],
    //     }),
    //     compute: {
    //         module: device.createShaderModule({
    //             code: speculative_ids_init_comp_spv,
    //         }),
    //         entryPoint: "main",
    //     },
    // });

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
            {
                binding: 3,
                visibility: GPUShaderStage.FRAGMENT,
                buffer: {
                    type: "storage",
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
        size: [this.width, this.height, 1],
        format: renderTargetFormat,
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING |
                   GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC
    });

    this.renderTargetDebugBGLayout = this.device.createBindGroupLayout({
        entries: [{
            binding: 0,
            visibility: GPUShaderStage.COMPUTE,
            storageTexture: {access: "write-only", format: renderTargetFormat}
        }]
    });
    this.renderTargetDebugBG = this.device.createBindGroup({
        layout: this.renderTargetDebugBGLayout,
        entries: [{binding: 0, resource: this.renderTarget.createView()}]
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
                    writeMask: 0,
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
                loadOp: "clear",
                clearValue: [1.0, 1.0, 1.0, 1],
                storeOp: "store"
            },
        ]
    };

    this.depthCompositeBGLayout = device.createBindGroupLayout({
        entries: [
            {binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: {type: "uniform"}},
            {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: {type: "storage"}},
            {binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: {type: "storage"}},
            {
                binding: 3,
                visibility: GPUShaderStage.COMPUTE,
                storageTexture: {access: "write-only", format: renderTargetFormat}
            },
            {binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: {type: "uniform"}},
        ]
    });
    this.depthCompositeBG1Layout = device.createBindGroupLayout({
        entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: {type: "storage"}}]
    });

    this.depthCompositePipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({
            bindGroupLayouts: [this.depthCompositeBGLayout, this.depthCompositeBG1Layout],
        }),
        compute: {
            module: device.createShaderModule({
                code: depth_composite_comp_spv,
            }),
            entryPoint: "main",
        },
    });

    this.resetSpeculativeIDsBGLayout = device.createBindGroupLayout({
        entries: [
            {binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: {type: "uniform"}},
            {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: {type: "storage"}},
            {binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: {type: "storage"}},
            {binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: {type: "storage"}},
        ]
    });

    this.resetSpeculativeIDsPipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({
            bindGroupLayouts: [
                this.resetSpeculativeIDsBGLayout,
            ],
        }),
        compute: {
            module: device.createShaderModule({
                code: reset_speculative_ids_comp_spv,
            }),
            entryPoint: "main",
        },
    });

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
                // Also pass the render target for debugging
                binding: 3,
                visibility: GPUShaderStage.COMPUTE,
                storageTexture: {access: "write-only", format: renderTargetFormat}
            },
            {
                binding: 4,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "storage",
                }
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
                    type: "storage",
                }
            },
            {
                binding: 7,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "storage",
                }
            }
        ],
    });
    this.macroTraverseRangesBGLayout = device.createBindGroupLayout({
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
            }
        ]
    });

    this.macroTraversePipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({
            bindGroupLayouts: [this.macroTraverseBGLayout, this.macroTraverseRangesBGLayout],
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
            {binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: {type: "uniform"}},
            {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: {type: "storage"}},
            {binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: {type: "storage"}}
        ]
    });
    this.resetBlockActivePipeline = device.createComputePipeline({
        layout:
            device.createPipelineLayout({bindGroupLayouts: [this.resetBlockActiveBGLayout]}),
        compute: {
            module: device.createShaderModule({code: reset_block_active_comp_spv}),
            entryPoint: "main"
        }
    });

    this.resetBlockNumRaysBGLayout = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {type: "uniform", hasDynamicOffset: true}
            },
            {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: {type: "storage"}}
        ]
    });
    this.resetBlockNumRaysPipeline = device.createComputePipeline({
        layout:
            device.createPipelineLayout({bindGroupLayouts: [this.resetBlockNumRaysBGLayout]}),
        compute: {
            module: device.createShaderModule({code: reset_block_num_rays_comp_spv}),
            entryPoint: "main"
        }
    });

    this.LODThresholdBuf = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.markBlockActiveBGLayout = device.createBindGroupLayout({
        entries: [
            {binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: {type: "uniform"}},
            {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: {type: "uniform"}},
            {binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: {type: "uniform"}},
            {binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: {type: "storage"}},
            {binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: {type: "storage"}},
            {binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: {type: "storage"}},
            {binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: {type: "storage"}},
        ]
    });
    this.markBlockActivePipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({
            bindGroupLayouts: [this.markBlockActiveBGLayout, this.renderTargetDebugBGLayout]
        }),
        compute: {
            module: device.createShaderModule({code: mark_block_active_wgsl_spv}),
            entryPoint: "main"
        }
    });

    this.countBlockRaysBGLayout = device.createBindGroupLayout({
        entries: [
            {binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: {type: "uniform"}},
            {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: {type: "storage"}},
            {binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: {type: "storage"}},
            {binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: {type: "storage"}},
        ]
    });
    this.countBlockRaysPipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({
            bindGroupLayouts: [this.countBlockRaysBGLayout],
        }),
        compute: {
            module: device.createShaderModule({code: count_block_rays_wgsl_spv}),
            entryPoint: "main"
        }
    });

    this.debugViewBlockRayCountsBGLayout = device.createBindGroupLayout({
        entries: [
            {binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: {type: "uniform"}},
            {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: {type: "storage"}},
            {binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: {type: "storage"}},
            {binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: {type: "storage"}},
            {
                binding: 3,
                visibility: GPUShaderStage.COMPUTE,
                storageTexture: {access: "write-only", format: renderTargetFormat}
            }
        ]
    });
    this.debugViewBlockRayCountsPipeline = device.createComputePipeline({
        layout: device.createPipelineLayout(
            {bindGroupLayouts: [this.debugViewBlockRayCountsBGLayout]}),
        compute: {
            module: device.createShaderModule({code: debug_view_rays_per_block_comp_spv}),
            entryPoint: "main"
        }
    });

    // Intermediate buffers for sorting ray IDs using their block ID as the key
    this.radixSorter = new RadixSorter(device);
    this.rayIDBuffer = device.createBuffer({
        size: this.radixSorter.getAlignedSize(this.width * this.height) * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this.compactSpeculativeIDBuffer = device.createBuffer({
        size: this.radixSorter.getAlignedSize(this.width * this.height) * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this.rayBlockIDBuffer = device.createBuffer({
        size: this.radixSorter.getAlignedSize(this.width * this.height) * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this.compactRayBlockIDBuffer = device.createBuffer({
        size: this.radixSorter.getAlignedSize(this.width * this.height) * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    this.rayActiveBuffer = device.createBuffer({
        size: this.width * this.height * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    this.rayActiveCompactOffsetBuffer = device.createBuffer({
        size: this.scanPipeline.getAlignedSize(this.width * this.height) * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this.speculativeRayOffsetBuffer = device.createBuffer({
        size: this.scanPipeline.getAlignedSize(this.width * this.height) * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this.scanRayActive = this.scanPipeline.prepareGPUInput(
        this.rayActiveCompactOffsetBuffer,
        this.scanPipeline.getAlignedSize(this.width * this.height));
    this.scanRayAfterActive = this.scanPipeline.prepareGPUInput(
        this.speculativeRayOffsetBuffer,
        this.scanPipeline.getAlignedSize(this.width * this.height));

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
        ],
    });
    this.writeRayAndBlockIDPipeline = device.createComputePipeline({
        layout:
            device.createPipelineLayout({bindGroupLayouts: [this.writeRayAndBlockIDBGLayout]}),
        compute: {
            module: device.createShaderModule({code: write_ray_and_block_id_comp_spv}),
            entryPoint: "main",
        }
    });

    this.markRayActiveBGLayout = device.createBindGroupLayout({
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
        ],
    });
    this.markRayActivePipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({bindGroupLayouts: [this.markRayActiveBGLayout]}),
        compute: {
            module: device.createShaderModule({code: mark_ray_active_comp_spv}),
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
            },
            {
                binding: 4,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "storage",
                }
            }
        ]
    });
    this.combineBlockInformationPipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({
            bindGroupLayouts:
                [this.combineBlockInformationBGLayout, this.pushConstantS1B0DynamicLayout]
        }),
        compute: {
            module: device.createShaderModule({code: combine_block_information_comp_spv}),
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
                storageTexture: {access: "write-only", format: renderTargetFormat}
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
                    type: "storage",
                }
            },
        ]
    });

    this.raytraceBlocksPipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({
            bindGroupLayouts: [
                this.rtBlocksPipelineBG0Layout,
                this.rtBlocksPipelineBG1Layout,
                this.pushConstantS1B0DynamicLayout
            ]
        }),
        compute: {
            module: device.createShaderModule({code: raytrace_active_block_comp_spv}),
            entryPoint: "main"
        }
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
    this.blockGridDims =
        [this.paddedDims[0] / 4, this.paddedDims[1] / 4, this.paddedDims[2] / 4];
    this.totalBlocks = (this.paddedDims[0] * this.paddedDims[1] * this.paddedDims[2]) / 64;

    this.coarseGridDims = [
        alignTo(this.blockGridDims[0], 4) / 4,
        alignTo(this.blockGridDims[1], 4) / 4,
        alignTo(this.blockGridDims[2], 4) / 4,
    ];
    this.totalCoarseCells =
        (this.coarseGridDims[0] * this.coarseGridDims[1] * this.coarseGridDims[2]);

    const groupThreadCount = 32;
    this.numWorkGroups = Math.ceil(this.totalBlocks / groupThreadCount);
    var cacheInitialSize = Math.ceil(this.totalBlocks * 0.01);

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
        buf.set([this.width], 14);

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

    this.uploadIsovalueBuf = this.device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC,
    });

    this.uploadLODBuf = this.device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC,
    });

    await this.computeBlockRanges();

    this.blockVisibleBuffer = this.device.createBuffer(
        {size: 4 * this.totalBlocks, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC});

    // Buffers for use when compacting down IDs of the active blocks
    // TODO: could do a bit better and filter neighbor blocks out too by just marking
    // visible blocks in a separate buffer (i.e., ones a ray is immediately in).
    // This set will include neighbors for now
    this.blockVisibleCompactOffsetBuffer = this.device.createBuffer({
        size: 4 * this.scanPipeline.getAlignedSize(this.totalBlocks),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    });

    // blockActive and blockVisibleCompactOffsetBuffer can alias each other as they are used at
    // different times in the pipeline
    this.blockActiveBuffer = this.blockVisibleCompactOffsetBuffer;

    this.scanBlockVisibleOffsets =
        this.scanPipeline.prepareGPUInput(this.blockVisibleCompactOffsetBuffer,
                                          this.scanPipeline.getAlignedSize(this.totalBlocks));

    this.resetRaysBG = this.device.createBindGroup({
        layout: this.resetRaysBGLayout,
        entries: [
            {binding: 0, resource: {buffer: this.rayInformationBuffer}},
            {binding: 1, resource: {buffer: this.volumeInfoBuffer}},
            {binding: 2, resource: {buffer: this.rayBlockIDBuffer}},
        ]
    });

    this.resetSpeculativeIDsBG = this.device.createBindGroup({
        layout: this.resetSpeculativeIDsBGLayout,
        entries: [
            {binding: 0, resource: {buffer: this.volumeInfoBuffer}},
            {binding: 1, resource: {buffer: this.speculativeRayIDBuffer}},
            {binding: 2, resource: {buffer: this.rayRGBZBuffer}},
            {binding: 3, resource: {buffer: this.rayBlockIDBuffer}},
        ]
    });

    this.depthCompositeBG = this.device.createBindGroup({
        layout: this.depthCompositeBGLayout,
        entries: [
            {binding: 0, resource: {buffer: this.viewParamBuf}},
            {binding: 1, resource: {buffer: this.speculativeRayIDBuffer}},
            {binding: 2, resource: {buffer: this.rayRGBZBuffer}},
            {binding: 3, resource: this.renderTarget.createView()},
            {binding: 4, resource: {buffer: this.volumeInfoBuffer}},
        ]
    });

    this.depthCompositeBG1 = this.device.createBindGroup({
        layout: this.depthCompositeBG1Layout,
        entries: [
            {binding: 0, resource: {buffer: this.rayInformationBuffer}},
        ]
    });

    // this.initSpeculativeIDsBG = this.device.createBindGroup({
    //     layout: this.initSpeculativeIDsBGLayout,
    //     entries: [
    //         {binding: 0, resource: {buffer: this.volumeInfoBuffer}},
    //         {binding: 1, resource: {buffer: this.viewParamBuf}},
    //         {binding: 2, resource: {buffer: this.speculativeIDBuffer}},
    //         {binding: 3, resource: {buffer: this.speculativeRayOffsetBuffer}},
    //     ]
    // });

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
            {
                binding: 3,
                resource: {
                    buffer: this.rayBlockIDBuffer,
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
                    buffer: this.rayInformationBuffer,
                },
            },
            {binding: 3, resource: this.renderTarget.createView()},
            {
                binding: 4,
                resource: {
                    buffer: this.gridIteratorBuffer,
                },
            },
            {
                binding: 5,
                resource: {
                    buffer: this.speculativeRayIDBuffer,
                },
            },
            {
                binding: 6,
                resource: {
                    buffer: this.speculativeRayOffsetBuffer,
                },
            },
            {
                binding: 7,
                resource: {
                    buffer: this.rayBlockIDBuffer,
                }
            }
        ],
    });

    this.resetBlockActiveBG = this.device.createBindGroup({
        layout: this.resetBlockActiveBGLayout,
        entries: [
            {binding: 0, resource: {buffer: this.volumeInfoBuffer}},
            {binding: 1, resource: {buffer: this.blockActiveBuffer}},
            {binding: 2, resource: {buffer: this.blockVisibleBuffer}},
        ]
    });

    this.markBlockActiveBG = this.device.createBindGroup({
        layout: this.markBlockActiveBGLayout,
        entries: [
            {binding: 0, resource: {buffer: this.volumeInfoBuffer}},
            {binding: 1, resource: {buffer: this.LODThresholdBuf}},
            {binding: 2, resource: {buffer: this.viewParamBuf}},
            {binding: 3, resource: {buffer: this.blockActiveBuffer}},
            {binding: 4, resource: {buffer: this.rayInformationBuffer}},
            {binding: 5, resource: {buffer: this.blockVisibleBuffer}},
            {binding: 6, resource: {buffer: this.rayBlockIDBuffer}},
        ]
    });

    this.writeRayAndBlockIDBG = this.device.createBindGroup({
        layout: this.writeRayAndBlockIDBGLayout,
        entries: [
            {binding: 0, resource: {buffer: this.volumeInfoBuffer}},
            {binding: 1, resource: {buffer: this.rayBlockIDBuffer}},
            {binding: 2, resource: {buffer: this.rayActiveBuffer}},
        ]
    });

    this.markRayActiveBG = this.device.createBindGroup({
        layout: this.markRayActiveBGLayout,
        entries: [
            {binding: 0, resource: {buffer: this.volumeInfoBuffer}},
            {binding: 1, resource: {buffer: this.rayInformationBuffer}},
            {binding: 2, resource: {buffer: this.rayActiveBuffer}},
        ]
    });
};

VolumeRaycaster.prototype.getMemoryStats = function() {
    // Data from this object
    var memUse = {
        units: "bytes",
        mc: {
            compressedData: this.compressedBuffer.size,
            voxelRanges: this.voxelRangesBuffer.size,
            coarseCellRanges: this.coarseCellRangesBuffer.size,

            // Every app will have view params, we can ignore it
            // viewParam: this.viewParamBuf.size,
            rays: this.rayInformationBuffer.size,
            rayPixelIDs: this.speculativeRayIDBuffer.size,
            sortedRayIDs: this.rayIDBuffer.size,
            compactPixelIDs: this.compactSpeculativeIDBuffer.size,
            rayBlockIDs: this.rayBlockIDBuffer.size,
            compactRayBlockIDs: this.compactRayBlockIDBuffer.size,
            rayActive: this.rayActiveBuffer.size,
            rayActiveCompactOffsets: this.rayActiveCompactOffsetBuffer.size,
            speculativeRayOffsets: this.speculativeRayOffsetBuffer.size,
            rayRGBZ: this.rayRGBZBuffer.size,

            gridIteratorState: this.gridIteratorBuffer.size,

            // block active aliases blockVisibleCompactOffsetBuffer because they aren't needed
            // at the same time in the pipeline
            // blockActive: this.blockActiveBuffer.size,
            blockVisible: this.blockVisibleBuffer.size,
            // This is really based on block visible, not active
            blockVisibleCompactOffsets: this.blockVisibleCompactOffsetBuffer.size,
            blockNumRays: this.blockNumRaysBuffer ? this.blockNumRaysBuffer.size : 0,
            blockRayOffsets: this.blockRayOffsetBuffer ? this.blockRayOffsetBuffer.size : 0,
            visibleBlockIDs: this.visibleBlockIDBuffer ? this.visibleBlockIDBuffer.size : 0,

            // Ignoring the combined block info buffer since this came from a binding count
            // limitation in WebGPU which I think has been addressed, we just haven't updated
            // the code, and it's just # active block size

            // combinedBlockInformation: this.combinedBlockInformationBuffer.size,

            // I think we can also ignore the cube vertices
            // cubeVertices: this.dataBuf.size
        },
        cache: {
            cache: this.lruCache.cacheSize * this.lruCache.elementSize,
            cachedItemSlots: this.lruCache.totalElements * 4,
            needsCaching: this.lruCache.totalElements * 4,
            needsCachingOffsets:
                this.scanPipeline.getAlignedSize(this.lruCache.totalElements) * 4,
            slotAge: this.lruCache.cacheSize * 4,
            slotAvailable: this.lruCache.cacheSize * 4,
            slotAvailableOffsets:
                this.scanPipeline.getAlignedSize(this.lruCache.cacheSize) * 4,
            slotAvailableIDs: this.lruCache.cacheSize * 4,
            slotItemIDs: this.lruCache.cacheSize * 4,
            cacheSizeBuf: 4,
        },
    };
    return memUse;
};

VolumeRaycaster.prototype.reportMemoryUse = function() {
    var formatBytes = function(count) {
        const giga = 1000000000;
        const mega = 1000000;
        const kilo = 1000;
        if (count > giga) {
            return (count / giga).toFixed(2) + " GB";
        } else if (count > mega) {
            return (count / mega).toFixed(2) + " MB";
        } else if (count > kilo) {
            return (count / kilo).toFixed(2) + " KB";
        }
        return count + " B";
    };

    var memUse = this.getMemoryStats();

    var totalMem = 0;
    var rcText = "Raycaster Data:<ul>";
    for (const prop in memUse.mc) {
        totalMem += memUse.mc[prop];
        rcText += "<li>" + prop + ": " + formatBytes(memUse.mc[prop]) + "</li>";
    }
    rcText += "</ul>";

    var cacheText = "LRU Cache Data:<ul>";
    for (const prop in memUse.cache) {
        totalMem += memUse.cache[prop];
        cacheText += "<li>" + prop + ": " + formatBytes(memUse.cache[prop]) + "</li>";
    }
    cacheText += "</ul>";
    return [rcText, cacheText, formatBytes(totalMem), memUse];
};

VolumeRaycaster.prototype.computeBlockRanges = async function() {
    // Note: this could be done by the server for us, but for this prototype
    // it's a bit easier to just do it here
    // Decompress each block and compute its value range, output to the blockRangesBuffer
    // BlockRangesBuffer = purely the ZFP block range
    // TODO: We don't need to keep this buffer long term actually
    var blockRangesBuffer = this.device.createBuffer({
        size: this.totalBlocks * 2 * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    // VoxelRangesBuffer = block range + neighbor cells for the dual grid
    this.voxelRangesBuffer = this.device.createBuffer({
        size: this.totalBlocks * 2 * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    // CoarseCellRanges = 4^3 blocks of ZFP blocks, including neighbor
    this.coarseCellRangesBuffer = this.device.createBuffer({
        size: this.totalCoarseCells * 2 * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    var bindGroup = this.device.createBindGroup({
        layout: this.computeBlockRangeBGLayout,
        entries: [
            {
                binding: 0,
                resource: {
                    buffer: this.compressedBuffer,
                }
            },
            {
                binding: 1,
                resource: {
                    buffer: this.volumeInfoBuffer,
                }
            },
            {
                binding: 2,
                resource: {
                    buffer: blockRangesBuffer,
                }
            }
        ]
    });
    this.voxelBindGroup = this.device.createBindGroup({
        layout: this.computeVoxelRangeBGLayout,
        entries: [{
            binding: 0,
            resource: {
                buffer: this.voxelRangesBuffer,
            }
        }]
    });

    this.macroTraverseRangesBG = this.device.createBindGroup({
        layout: this.macroTraverseRangesBGLayout,
        entries: [
            {
                binding: 0,
                resource: {
                    buffer: this.voxelRangesBuffer,
                },
            },
            {
                binding: 1,
                resource: {
                    buffer: this.coarseCellRangesBuffer,
                }
            }
        ]

    });

    const groupThreadCount = 32;
    var totalWorkGroups = Math.ceil(this.totalBlocks / groupThreadCount);

    var pushConstants = buildPushConstantsBuffer(this.device, totalWorkGroups);

    var blockIDOffsetBG = this.device.createBindGroup({
        layout: this.pushConstantS1B0DynamicLayout,
        entries: [{binding: 0, resource: {buffer: pushConstants.gpuBuffer, size: 4}}]
    });

    var commandEncoder = this.device.createCommandEncoder();
    var pass = commandEncoder.beginComputePass();

    // Decompress each block and compute its range
    pass.setPipeline(this.computeBlockRangePipeline);
    pass.setBindGroup(0, bindGroup);
    for (var i = 0; i < pushConstants.nOffsets; ++i) {
        pass.setBindGroup(1, blockIDOffsetBG, pushConstants.dynamicOffsets, i, 1);
        pass.dispatchWorkgroups(pushConstants.dispatchSizes[i], 1, 1);
    }
    pass.end();

    // Compute each block's range including its neighbors
    var pass = commandEncoder.beginComputePass();
    pass.setPipeline(this.computeVoxelRangePipeline);
    pass.setBindGroup(0, bindGroup);
    pass.setBindGroup(2, this.voxelBindGroup);
    for (var i = 0; i < pushConstants.nOffsets; ++i) {
        pass.setBindGroup(1, blockIDOffsetBG, pushConstants.dynamicOffsets, i, 1);
        pass.dispatchWorkgroups(pushConstants.dispatchSizes[i], 1, 1);
    }
    pass.end();

    var pass = commandEncoder.beginComputePass();
    // Enqueue pass to compute the coarse cell ranges
    var totalWorkGroupsCoarseCell = Math.ceil(this.totalCoarseCells / groupThreadCount);
    var coarsePushConstants = buildPushConstantsBuffer(this.device, totalWorkGroupsCoarseCell);
    var coarseRangeBG = this.device.createBindGroup({
        layout: this.computeCoarseCellRangeBGLayout,
        entries: [
            {
                binding: 0,
                resource: {
                    buffer: this.volumeInfoBuffer,
                }
            },
            {binding: 1, resource: {buffer: coarsePushConstants.gpuBuffer, size: 4}},
            {binding: 2, resource: {buffer: this.voxelRangesBuffer}},
            {binding: 3, resource: {buffer: this.coarseCellRangesBuffer}}
        ]
    });
    pass.setPipeline(this.computeCoarseCellRangePipeline);
    for (var i = 0; i < coarsePushConstants.nOffsets; ++i) {
        pass.setBindGroup(0, coarseRangeBG, coarsePushConstants.dynamicOffsets, i, 1);
        pass.dispatchWorkgroups(coarsePushConstants.dispatchSizes[i], 1, 1);
    }

    pass.end();
    this.device.queue.submit([commandEncoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();

    /*
    {
        var dbgBuffer = this.device.createBuffer({
            size: blockRangesBuffer.size,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });

        var commandEncoder = this.device.createCommandEncoder();
        commandEncoder.copyBufferToBuffer(blockRangesBuffer, 0, dbgBuffer, 0, dbgBuffer.size);
        this.device.queue.submit([commandEncoder.finish()]);
        await this.device.queue.onSubmittedWorkDone();

        await dbgBuffer.mapAsync(GPUMapMode.READ);

        var debugVals = new Float32Array(dbgBuffer.getMappedRange());

        var valRange = [Infinity, -Infinity];

        for (var i = 0; i < debugVals.length; ++i) {
            valRange[0] = Math.min(valRange[0], debugVals[i]);
            valRange[1] = Math.max(valRange[1], debugVals[i]);
        }
        console.log(`value range = ${valRange}`);


        dbgBuffer.unmap();
    }
    */
};

// Progressively compute the surface, returns true when rendering is complete
VolumeRaycaster.prototype.renderSurface = async function(
    isovalue, LODThreshold, viewParamUpload, renderParamsChanged, eyePos, eyeDir, upDir) {
    this.passPerfStats = {};

    if (this.renderComplete && !renderParamsChanged) {
        return this.renderComplete;
    }
    console.log("===== Rendering Surface =======");
    var startPass = performance.now();

    if (renderParamsChanged) {
        this.numVisibleBlocks = 0;
        this.newDecompressed = 0;

        this.totalPassTime = 0;
        this.numPasses = 0;
        this.speculationCount = 1;
        this.speculationEnabled = this.enableSpeculationUI.checked;

        this.surfacePerfStats = [];

        this.recordVisibleBlocks = this.recordVisibleBlocksUI.checked;
        this.recordBlockActiveList = null;
        this.recordBlockVisibleList = null;
        if (this.recordVisibleBlocks) {
            console.log(
                `WARNING: Recording active/visible block statistics may effect performance!`);
            this.recordBlockActiveList = new Uint8Array(this.totalBlocks).fill(0);
            this.recordBlockVisibleList = new Uint8Array(this.totalBlocks).fill(0);
        }

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

        // We need to reset the speculation count
        var uploadSpeculationCount = this.device.createBuffer(
            {size: 4, usage: GPUBufferUsage.COPY_SRC, mappedAtCreation: true});
        new Uint32Array(uploadSpeculationCount.getMappedRange()).set([this.speculationCount]);
        uploadSpeculationCount.unmap();
        commandEncoder.copyBufferToBuffer(
            uploadSpeculationCount, 0, this.viewParamBuf, (16 + 8 + 1 + 1) * 4, 4);

        // Reset active blocks for the new viewpoint/isovalue to allow eviction of old blocks
        var pass = commandEncoder.beginComputePass();
        pass.setPipeline(this.resetBlockActivePipeline);
        pass.setBindGroup(0, this.resetBlockActiveBG);
        pass.dispatchWorkgroups(Math.ceil(this.blockGridDims[0] / 8),
                                this.blockGridDims[1],
                                this.blockGridDims[2]);
        pass.end();
        this.device.queue.submit([commandEncoder.finish()]);

        var start = performance.now();
        await this.computeInitialRays(viewParamUpload);
        var end = performance.now();
        this.passPerfStats["computeInitialRays_ms"] = end - start;
        this.passPerfStats["totalBlocks"] = this.totalBlocks;
        this.passPerfStats["volumeDims"] = this.volumeDims;
        this.passPerfStats["imageSize"] = [this.width, this.height];
        this.passPerfStats["nPixels"] = this.width * this.height;

        // Save camera info as well for reproducibility
        this.passPerfStats["eyePos"] = [eyePos[0], eyePos[1], eyePos[2]];
        this.passPerfStats["eyeDir"] = [eyeDir[0], eyeDir[1], eyeDir[2]];
        this.passPerfStats["upDir"] = [upDir[0], upDir[1], upDir[2]];
    }
    console.log(`++++ Surface pass ${this.numPasses} ++++`);
    this.passPerfStats["passID"] = this.numPasses;
    this.passPerfStats["speculationCount"] = this.speculationCount;
    this.passPerfStats["isovalue"] = isovalue;

    var startPass = performance.now();

    // Reset the number of blocks visible/active each pass to reduce memory usage and skip
    // computing on inactive blocks
    {
        var commandEncoder = this.device.createCommandEncoder();
        var pass = commandEncoder.beginComputePass();
        pass.setPipeline(this.resetBlockActivePipeline);
        pass.setBindGroup(0, this.resetBlockActiveBG);
        pass.dispatchWorkgroups(Math.ceil(this.blockGridDims[0] / 8),
                                this.blockGridDims[1],
                                this.blockGridDims[2]);
        pass.end();
        this.device.queue.submit([commandEncoder.finish()]);
    }

    var start = performance.now();
    await this.macroTraverse();
    var end = performance.now();
    // console.log(`Macro Traverse: ${end - start}ms`);
    this.passPerfStats["macroTraverse_ms"] = end - start;

    start = performance.now();
    await this.markActiveBlocks();
    end = performance.now();
    // console.log(`Mark Active Blocks: ${end - start}ms`);
    this.passPerfStats["markActiveBlocks_ms"] = end - start;

    // Decompress any new blocks needed for the pass
    start = performance.now();
    // TODO: PAss the passPerfStats through to the LRU Cache so it can write stats to it
    var [nBlocksToDecompress, decompressBlockIDs] =
        await this.lruCache.update(this.blockActiveBuffer, this.passPerfStats);
    end = performance.now();
    // console.log(`LRU: ${end - start}ms`);
    this.passPerfStats["lruCacheUpdate_ms"] = end - start;
    this.passPerfStats["nBlocksToDecompress"] = nBlocksToDecompress;

    this.newDecompressed = nBlocksToDecompress;
    if (nBlocksToDecompress != 0) {
        // console.log(`Will decompress ${nBlocksToDecompress} blocks`);
        start = performance.now();
        await this.decompressBlocks(nBlocksToDecompress, decompressBlockIDs);
        end = performance.now();
        // console.log(`Decompress: ${end - start}ms`);
        this.passPerfStats["decompressBlocks_ms"] = end - start;
    }

    // Now at this step we can do the block id compaction, compute # blocks visible
    var numVisibleBlocks = await this.compactVisibleBlockIDs();
    // this.numVisibleBlocks is just for displaying the value in the UI
    this.numVisibleBlocks = numVisibleBlocks;
    this.passPerfStats["nVisibleBlocks"] = numVisibleBlocks;
    var numRaysActive = 0;
    if (numVisibleBlocks > 0) {
        start = performance.now();
        // Then run a pass to write the compacted blockNumRays and blockRayOffsets instead of #
        // block buffers
        numRaysActive = await this.computeBlockRayOffsets(numVisibleBlocks);
        end = performance.now();
        this.passPerfStats["computeBlockRayOffsets_ms"] = end - start;
        this.passPerfStats["nRaysActive"] = numRaysActive;
        // console.log(`Ray active and offsets: ${end - start}ms`);
        // console.log(`numRaysActive = ${numRaysActive}`);
        if (numRaysActive > 0) {
            start = performance.now();
            await this.sortActiveRaysByBlock(numRaysActive);
            end = performance.now();
            this.passPerfStats["sortActiveRaysByBlock_ms"] = end - start;
            // console.log(`Sort active rays by block: ${end - start}ms`);

            start = performance.now();
            await this.raytraceVisibleBlocks(numVisibleBlocks);
            end = performance.now();
            this.passPerfStats["raytraceVisibleBlocks_ms"] = end - start;
            // console.log(`Raytrace blocks: ${end - start}ms`);
            // console.log(`PASS TOOK: ${end - startPass}ms`);

            start = performance.now();
            var commandEncoder = this.device.createCommandEncoder();
            var pass = commandEncoder.beginComputePass();
            pass.setPipeline(this.depthCompositePipeline);
            pass.setBindGroup(0, this.depthCompositeBG);
            pass.setBindGroup(1, this.depthCompositeBG1);
            pass.dispatchWorkgroups(
                Math.ceil(this.width / 32), Math.ceil(this.height / this.speculationCount), 1);
            pass.end();
            this.device.queue.submit([commandEncoder.finish()]);
            await this.device.queue.onSubmittedWorkDone();
            end = performance.now();
            this.passPerfStats["depthComposite_ms"] = end - start;

            start = performance.now();
            var commandEncoder = this.device.createCommandEncoder();
            var pass = commandEncoder.beginComputePass();
            pass.setPipeline(this.markRayActivePipeline);
            pass.setBindGroup(0, this.markRayActiveBG);
            pass.dispatchWorkgroups(Math.ceil(this.width / 32), this.height, 1);
            pass.end();
            // We scan the speculativeRayOffsetBuffer, so copy the ray active information over
            commandEncoder.copyBufferToBuffer(this.rayActiveBuffer,
                                              0,
                                              this.speculativeRayOffsetBuffer,
                                              0,
                                              this.width * this.height * 4);
            this.device.queue.submit([commandEncoder.finish()]);

            numRaysActive = await this.scanRayAfterActive.scan(this.width * this.height);
            end = performance.now();
            this.passPerfStats["countRemainingActiveRays_ms"] = end - start;
            this.passPerfStats["endPassRaysActive_ms"] = numRaysActive;
            // console.log(`num rays active after raytracing: ${numRaysActive}`);

            if (this.speculationEnabled) {
                var commandEncoder = this.device.createCommandEncoder();
                this.speculationCount =
                    Math.min(Math.floor(this.width * this.height / numRaysActive), 64);
                // console.log(`Next pass speculation count is ${this.speculationCount}`);
                var uploadSpeculationCount = this.device.createBuffer(
                    {size: 4, usage: GPUBufferUsage.COPY_SRC, mappedAtCreation: true});
                new Uint32Array(uploadSpeculationCount.getMappedRange()).set([
                    this.speculationCount
                ]);
                uploadSpeculationCount.unmap();
                commandEncoder.copyBufferToBuffer(
                    uploadSpeculationCount, 0, this.viewParamBuf, (16 + 8 + 1 + 1) * 4, 4);
                this.device.queue.submit([commandEncoder.finish()]);
                await this.device.queue.onSubmittedWorkDone();
            }
            console.log(`++++++++++`);
        }
    } else {
        this.passPerfStats["nRaysActive"] = numRaysActive;
    }
    var endPass = performance.now();
    this.passPerfStats["totalPassTime_ms"] = endPass - startPass;
    this.passPerfStats["memory"] = this.getMemoryStats();
    this.surfacePerfStats.push(this.passPerfStats);

    console.log("=============");
    this.totalPassTime += endPass - startPass;
    this.numPasses += 1;
    //}
    this.renderComplete = numRaysActive == 0;
    if (this.renderComplete) {
        if (this.recordVisibleBlocks) {
            var nTotalBlocksActive = 0;
            var nTotalBlocksVisible = 0;
            for (var i = 0; i < this.totalBlocks; ++i) {
                if (this.recordBlockActiveList[i] != 0) {
                    ++nTotalBlocksActive;
                }
                if (this.recordBlockVisibleList[i] != 0) {
                    ++nTotalBlocksVisible;
                }
            }
            console.log(`Total blocks active: ${nTotalBlocksActive} (${
                nTotalBlocksActive / this.totalBlocks * 100})`);
            console.log(`Total blocks visible: ${nTotalBlocksVisible} (${
                nTotalBlocksVisible / this.totalBlocks * 100})`);
            this.passPerfStats["nTotalBlocksActive"] = nTotalBlocksActive;
            this.passPerfStats["nTotalBlocksVisible"] = nTotalBlocksVisible;
        }
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
VolumeRaycaster.prototype.computeInitialRays = async function(viewParamUpload) {
    var commandEncoder = this.device.createCommandEncoder();

    commandEncoder.copyBufferToBuffer(
        viewParamUpload, 0, this.viewParamBuf, 0, (16 + 8 + 1) * 4);

    var resetRaysPass = commandEncoder.beginComputePass(this.resetRaysPipeline);
    resetRaysPass.setBindGroup(0, this.resetRaysBG);
    resetRaysPass.setPipeline(this.resetRaysPipeline);
    resetRaysPass.dispatchWorkgroups(Math.ceil(this.width / 8), this.height, 1);
    resetRaysPass.end();

    var initialRaysPass = commandEncoder.beginRenderPass(this.initialRaysPassDesc);

    initialRaysPass.setPipeline(this.initialRaysPipeline);
    initialRaysPass.setVertexBuffer(0, this.dataBuf);
    initialRaysPass.setBindGroup(0, this.initialRaysBindGroup);
    initialRaysPass.draw(12 * 3, 1, 0, 0);

    initialRaysPass.end();
    this.device.queue.submit([commandEncoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();
};

// Step the active rays forward in the macrocell grid, updating block_id and t.
// Rays that exit the volume will have block_id = UINT_MAX and t = FLT_MAX
VolumeRaycaster.prototype.macroTraverse = async function() {
    // TODO: Would it be worth doing a scan and compact to find just the IDs of the currently
    // active rays? then only advancing them? We'll do that anyways so it would tell us when
    // we're done too (no rays active)
    var commandEncoder = this.device.createCommandEncoder();

    // Reset speculative IDs buffer (here for now but could be moved)
    var resetSpecIDsPass = commandEncoder.beginComputePass();
    resetSpecIDsPass.setBindGroup(0, this.resetSpeculativeIDsBG);
    resetSpecIDsPass.setPipeline(this.resetSpeculativeIDsPipeline);
    resetSpecIDsPass.dispatchWorkgroups(Math.ceil(this.width / 32), this.height, 1);
    resetSpecIDsPass.end();

    // Update the current pass index
    var uploadPassIndex = this.device.createBuffer(
        {size: 4, usage: GPUBufferUsage.COPY_SRC, mappedAtCreation: true});
    new Int32Array(uploadPassIndex.getMappedRange()).set([this.numPasses]);
    uploadPassIndex.unmap();

    commandEncoder.copyBufferToBuffer(this.uploadIsovalueBuf, 0, this.volumeInfoBuffer, 52, 4);
    commandEncoder.copyBufferToBuffer(
        uploadPassIndex, 0, this.viewParamBuf, (16 + 8 + 1) * 4, 4);

    var pass = commandEncoder.beginComputePass();

    pass.setPipeline(this.macroTraversePipeline);
    pass.setBindGroup(0, this.macroTraverseBindGroup);
    pass.setBindGroup(1, this.macroTraverseRangesBG);
    pass.dispatchWorkgroups(Math.ceil(this.width / 64), this.height, 1);

    pass.end();
    this.device.queue.submit([commandEncoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();

    // Log speculative ray IDs buffer
    // var readbackSpeculativeIDBuffer = this.device.createBuffer({
    //     size: this.speculativeRayIDBuffer.size,
    //     usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    // });
    // var commandEncoder = this.device.createCommandEncoder();
    // commandEncoder.copyBufferToBuffer(
    //     this.speculativeRayIDBuffer, 0, readbackSpeculativeIDBuffer, 0,
    //     this.speculativeRayIDBuffer.size);
    // this.device.queue.submit([commandEncoder.finish()]);
    // await this.device.queue.onSubmittedWorkDone();
    // await readbackSpeculativeIDBuffer.mapAsync(GPUMapMode.READ);
    // var specIDs = new Uint32Array(readbackSpeculativeIDBuffer.getMappedRange());
    // console.log(specIDs);

    uploadPassIndex.destroy();
};

// Mark the active and visible blocks for the current viewpoint/isovalue and count the # of
// rays that we need to process for each block
VolumeRaycaster.prototype.markActiveBlocks = async function() {
    var commandEncoder = this.device.createCommandEncoder();
    var pass = commandEncoder.beginComputePass();

    // Compute which blocks are active and how many rays each has
    pass.setPipeline(this.markBlockActivePipeline);
    pass.setBindGroup(0, this.markBlockActiveBG);
    pass.setBindGroup(1, this.renderTargetDebugBG);
    pass.dispatchWorkgroups(Math.ceil(this.width / 32), this.height, 1);

    pass.end();

    this.device.queue.submit([commandEncoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();

    if (this.recordVisibleBlocks) {
        var activeReadback = this.device.createBuffer({
            size: 4 * this.totalBlocks,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        });
        var visibleReadback = this.device.createBuffer({
            size: 4 * this.totalBlocks,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        });

        var commandEncoder = this.device.createCommandEncoder();
        commandEncoder.copyBufferToBuffer(
            this.blockActiveBuffer, 0, activeReadback, 0, activeReadback.size);
        commandEncoder.copyBufferToBuffer(
            this.blockVisibleBuffer, 0, visibleReadback, 0, visibleReadback.size);
        this.device.queue.submit([commandEncoder.finish()]);
        await this.device.queue.onSubmittedWorkDone();

        await activeReadback.mapAsync(GPUMapMode.READ);
        await visibleReadback.mapAsync(GPUMapMode.READ);

        var blockActive = new Uint32Array(activeReadback.getMappedRange());
        var blockVisible = new Uint32Array(visibleReadback.getMappedRange());
        for (var i = 0; i < this.totalBlocks; ++i) {
            this.recordBlockActiveList[i] |= blockActive[i];
            this.recordBlockVisibleList[i] |= blockVisible[i];
        }

        activeReadback.unmap();
        visibleReadback.unmap();

        activeReadback.destroy();
        visibleReadback.destroy();
    }
};

// Compute the number of visible blocks and compact their IDs. Also produces an offset buffer
// mapping from original block index -> compacted location
// - visibleBlockIDBuffer: compact list of active block IDs (visible block IDs)
// - blockVisibleCompactOffsets: mapping from block ID to compacted location
VolumeRaycaster.prototype.compactVisibleBlockIDs = async function() {
    var start = performance.now();
    // Populate the ray ID, ray block ID and ray active buffers
    var commandEncoder = this.device.createCommandEncoder();

    // We scan the active block buffer to produce offsets for compacting active block IDs
    // down This will let us reduce the dispatch size of the ray tracing step to just active
    // blocks
    commandEncoder.copyBufferToBuffer(this.blockVisibleBuffer,
                                      0,
                                      this.blockVisibleCompactOffsetBuffer,
                                      0,
                                      this.totalBlocks * 4);
    this.device.queue.submit([commandEncoder.finish()]);

    // Compact the visible block IDs down
    var numVisibleBlocks = await this.scanBlockVisibleOffsets.scan(this.totalBlocks);
    if (numVisibleBlocks === 0) {
        return numVisibleBlocks;
    }

    this.visibleBlockIDBuffer = this.device.createBuffer({
        size: 4 * numVisibleBlocks,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    });

    await this.streamCompact.compactActiveIDs(this.totalBlocks,
                                              this.blockVisibleBuffer,
                                              this.blockVisibleCompactOffsetBuffer,
                                              this.visibleBlockIDBuffer);
    var end = performance.now();
    // console.log(`compactVisibleBlockIDs: ${end - start}ms`);

    return numVisibleBlocks;
};

// Compute the number of visible rays per block, scan the blockNumRaysBuffer storing the output
// in blockRayOffsetBuffer and return the number of active rays.
VolumeRaycaster.prototype.computeBlockRayOffsets = async function(numVisibleBlocks) {
    this.blockNumRaysBuffer = this.device.createBuffer({
        size: 4 * numVisibleBlocks,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    });

    // Scan result buffer for the block ray offsets (computed by scanning the result in
    // blockNumRaysBuffer)
    this.blockRayOffsetBuffer = this.device.createBuffer({
        size: 4 * this.scanPipeline.getAlignedSize(numVisibleBlocks),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    });
    this.scanBlockRayOffsets = this.scanPipeline.prepareGPUInput(
        this.blockRayOffsetBuffer, this.scanPipeline.getAlignedSize(numVisibleBlocks));

    var countBlockRaysBG = this.device.createBindGroup({
        layout: this.countBlockRaysBGLayout,
        entries: [
            {binding: 0, resource: {buffer: this.volumeInfoBuffer}},
            {binding: 1, resource: {buffer: this.blockNumRaysBuffer}},
            {binding: 2, resource: {buffer: this.rayBlockIDBuffer}},
            {binding: 3, resource: {buffer: this.blockVisibleCompactOffsetBuffer}},
        ]
    });

    // Now here we can run a compute pass to compute # of rays per block, this is image size
    // dispatch so no need to chunk it
    var commandEncoder = this.device.createCommandEncoder();
    var pass = commandEncoder.beginComputePass();
    pass.setPipeline(this.countBlockRaysPipeline);
    pass.setBindGroup(0, countBlockRaysBG);
    pass.dispatchWorkgroups(Math.ceil(this.width / 32), this.height, 1);
    pass.end();

    commandEncoder.copyBufferToBuffer(this.blockNumRaysBuffer,
                                      0,
                                      this.blockRayOffsetBuffer,
                                      0,
                                      this.blockNumRaysBuffer.size);
    await this.device.queue.submit([commandEncoder.finish()]);

    return await this.scanBlockRayOffsets.scan(numVisibleBlocks);
};

// Sort the active ray IDs by their block ID in ascending order (inactive rays will be at the
// end).
VolumeRaycaster.prototype.sortActiveRaysByBlock = async function(numRaysActive) {
    // Populate the ray ID, ray block ID and ray active buffers
    var commandEncoder = this.device.createCommandEncoder();
    var pass = commandEncoder.beginComputePass()
    pass.setPipeline(this.writeRayAndBlockIDPipeline);
    pass.setBindGroup(0, this.writeRayAndBlockIDBG);
    pass.dispatchWorkgroups(Math.ceil(this.width / 8), this.height, 1);
    pass.end();

    // We scan the rayActiveCompactOffsetBuffer, so copy the ray active information over
    commandEncoder.copyBufferToBuffer(this.rayActiveBuffer,
                                      0,
                                      this.rayActiveCompactOffsetBuffer,
                                      0,
                                      this.width * this.height * 4);

    this.device.queue.submit([commandEncoder.finish()]);

    // Scan the active ray buffer and compact the active ray IDs before we sort
    // so that the sort doesn't have to process such a large number of items
    // TODO: This is not matching numRaysActive?
    var nactive = await this.scanRayActive.scan(this.width * this.height);
    // Should match numRaysActive, sanity check here
    if (numRaysActive != nactive) {
        console.log(`nactive ${nactive} doesn't match numRaysActive ${numRaysActive}!?`);
    }
    var startCompacts = performance.now();
    // Compact the active ray IDs and their block IDs down
    await this.streamCompact.compactActive(this.width * this.height,
                                           this.rayActiveBuffer,
                                           this.rayActiveCompactOffsetBuffer,
                                           this.speculativeRayIDBuffer,
                                           this.rayIDBuffer);

    await this.streamCompact.compactActiveIDs(this.width * this.height,
                                              this.rayActiveBuffer,
                                              this.rayActiveCompactOffsetBuffer,
                                              this.compactSpeculativeIDBuffer);

    await this.streamCompact.compactActive(this.width * this.height,
                                           this.rayActiveBuffer,
                                           this.rayActiveCompactOffsetBuffer,
                                           this.rayBlockIDBuffer,
                                           this.compactRayBlockIDBuffer);

    var endCompacts = performance.now();
    // console.log(`sortActiveRaysByBlock: Compacts ${endCompacts - startCompacts}ms`);

    var compactRayBlockIDBufferCopy = this.device.createBuffer({
        size: this.radixSorter.getAlignedSize(this.compactRayBlockIDBuffer.size / 4) * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    var commandEncoder = this.device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(this.compactRayBlockIDBuffer,
                                      0,
                                      compactRayBlockIDBufferCopy,
                                      0,
                                      this.compactRayBlockIDBuffer.size);
    this.device.queue.submit([commandEncoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();

    var start = performance.now();
    // Sort active ray IDs by their block ID
    await this.radixSorter.sort(
        this.compactRayBlockIDBuffer, this.rayIDBuffer, numRaysActive, false);
    await this.radixSorter.sort(
        compactRayBlockIDBufferCopy, this.compactSpeculativeIDBuffer, numRaysActive, false);
    var end = performance.now();
    // console.log(`sortActiveRaysByBlock: Sort rays by blocks: ${end - start}ms`);
};

VolumeRaycaster.prototype.raytraceVisibleBlocks = async function(numVisibleBlocks) {
    /*
    console.log(
        `Raytracing ${numVisibleBlocks} blocks, speculation count = ${this.speculationCount}`);
        */

    // Must recreate each time b/c cache buffer will grow
    var rtBlocksPipelineBG0 = this.device.createBindGroup({
        layout: this.rtBlocksPipelineBG0Layout,
        entries: [
            {binding: 0, resource: {buffer: this.volumeInfoBuffer}},
            {binding: 1, resource: {buffer: this.lruCache.cache}},
            {binding: 2, resource: {buffer: this.lruCache.cachedItemSlots}},
        ]
    });

    this.combinedBlockInformationBuffer = this.device.createBuffer(
        {size: numVisibleBlocks * 4 * 4, usage: GPUBufferUsage.STORAGE});

    this.rtBlocksPipelineBG1 = this.device.createBindGroup({
        layout: this.rtBlocksPipelineBG1Layout,
        entries: [
            {binding: 0, resource: {buffer: this.viewParamBuf}},
            {binding: 1, resource: {buffer: this.rayInformationBuffer}},
            {binding: 2, resource: {buffer: this.rayIDBuffer}},
            {binding: 3, resource: {buffer: this.combinedBlockInformationBuffer}},
            {binding: 4, resource: this.renderTarget.createView()},
            {binding: 5, resource: {buffer: this.compactSpeculativeIDBuffer}},
            {binding: 6, resource: {buffer: this.rayRGBZBuffer}},
        ]
    });

    this.combineBlockInformationBG = this.device.createBindGroup({
        layout: this.combineBlockInformationBGLayout,
        entries: [
            {binding: 0, resource: {buffer: this.combinedBlockInformationBuffer}},
            {binding: 1, resource: {buffer: this.visibleBlockIDBuffer}},
            {binding: 2, resource: {buffer: this.blockRayOffsetBuffer}},
            {binding: 3, resource: {buffer: this.blockNumRaysBuffer}},
            {binding: 4, resource: {buffer: this.blockActiveBuffer}},
        ]
    });

    var commandEncoder = this.device.createCommandEncoder();
    {
        const groupThreadCount = 64;
        const totalWorkGroups = Math.ceil(numVisibleBlocks / groupThreadCount);

        var pushConstants = buildPushConstantsBuffer(
            this.device, totalWorkGroups, new Uint32Array([numVisibleBlocks]));

        var blockIDOffsetBG = this.device.createBindGroup({
            layout: this.pushConstantS1B0DynamicLayout,
            entries: [{binding: 0, resource: {buffer: pushConstants.gpuBuffer, size: 12}}]
        });

        var pass = commandEncoder.beginComputePass();
        // First make the combined block information buffer (to fit in 6 storage buffers)
        // The limit will be bumped up to 8 so we could remove this piece in a bit once
        // the change lands in Chromium
        pass.setPipeline(this.combineBlockInformationPipeline);
        pass.setBindGroup(0, this.combineBlockInformationBG);
        for (var i = 0; i < pushConstants.nOffsets; ++i) {
            pass.setBindGroup(1, blockIDOffsetBG, pushConstants.dynamicOffsets, i, 1);
            pass.dispatchWorkgroups(pushConstants.dispatchSizes[i], 1, 1);
        }
        pass.end();
    }

    {
        var pushConstants = buildPushConstantsBuffer(this.device, numVisibleBlocks);

        var blockIDOffsetBG = this.device.createBindGroup({
            layout: this.pushConstantS1B0DynamicLayout,
            entries: [{binding: 0, resource: {buffer: pushConstants.gpuBuffer, size: 8}}]
        });

        var pass = commandEncoder.beginComputePass();
        // TODO: Might be worth for data sets where many blocks
        // project to a lot of pixels to split up the dispatches,
        // and do multiple dispatch indirect for the blocks touching
        // many pixels so that we don't serialize so badly on them, while
        // still doing a single dispatch for all blocks touching <= 64 pixels
        // since that will be most of the blocks, especially for large data.
        pass.setPipeline(this.raytraceBlocksPipeline);
        pass.setBindGroup(0, rtBlocksPipelineBG0);
        pass.setBindGroup(1, this.rtBlocksPipelineBG1);
        for (var i = 0; i < pushConstants.nOffsets; ++i) {
            pass.setBindGroup(2, blockIDOffsetBG, pushConstants.dynamicOffsets, i, 1);
            pass.dispatchWorkgroups(pushConstants.dispatchSizes[i], 1, 1);
        }
        pass.end();
    }

    this.device.queue.submit([commandEncoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();
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

    var workGroupCount = Math.ceil(nBlocksToDecompress / 64.0);
    var numChunks = Math.ceil(workGroupCount / this.maxDispatchSize);
    var dispatchChunkOffsetsBuf = this.device.createBuffer({
        size: numChunks * 256,
        usage: GPUBufferUsage.UNIFORM,
        mappedAtCreation: true,
    });
    var map = new Uint32Array(dispatchChunkOffsetsBuf.getMappedRange());
    for (var i = 0; i < numChunks; ++i) {
        map[i * 64] = i * this.maxDispatchSize;
        map[i * 64 + 1] = nBlocksToDecompress;
    }
    dispatchChunkOffsetsBuf.unmap();

    // We execute these chunks in separate submissions to avoid having them
    // execute all at once and trigger a TDR if we're decompressing a large amount of data
    for (var i = 0; i < numChunks; ++i) {
        var numWorkGroups =
            Math.min(workGroupCount - i * this.maxDispatchSize, this.maxDispatchSize);
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
                        size: 8,
                        offset: i * 256,
                    },
                },
            ],
        });
        pass.setBindGroup(1, decompressBlocksStartOffsetBG);
        pass.dispatchWorkgroups(Math.ceil(numWorkGroups), 1, 1);
        pass.end();
        this.device.queue.submit([commandEncoder.finish()]);
    }
    await this.device.queue.onSubmittedWorkDone();
    dispatchChunkOffsetsBuf.destroy();
};
