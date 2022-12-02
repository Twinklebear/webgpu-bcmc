var VolumeRaycaster = function(device, canvas) {
    this.device = device;
    this.scanPipeline = new ExclusiveScanPipeline(device);
    this.streamCompact = new StreamCompact(device);
    this.numActiveBlocks = 0;
    this.renderComplete = false;
    this.initialRayTimes = [];
    this.initialRayTimeSum = 0;

    this.canvas = canvas;
    console.log(`canvas size ${canvas.width}x${canvas.height}`);

    // Max dispatch size for more computationally heavy kernels
    // which might hit TDR on lower power devices
    this.maxDispatchSize = device.limits.maxComputeWorkgroupsPerDimension;

    this.numActiveBlocksStorage = 0;

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

    // We'll need a max of canvas.width * canvas.height RayInfo structs in the buffer,
    // so just allocate it once up front
    this.rayInformationBuffer = device.createBuffer({
        size: this.canvas.width * this.canvas.height * 16,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // We need canvas.width * canvas.height RayIDs for speculation,
    // with ray indices repeated as speculation occurs
    this.speculativeRayIDBuffer = device.createBuffer({
        size: this.canvas.width * this.canvas.height * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    this.rayRGBZBuffer = device.createBuffer({
        size: this.canvas.width * this.canvas.height * 4 * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // Each ray stores 2 iterator states, the coarse one followed by the fine one.
    // Each state is 32b
    this.gridIteratorBuffer = device.createBuffer({
        size: this.canvas.width * this.canvas.height * 64,
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
        size: [this.canvas.width, this.canvas.height, 1],
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
            {
                binding: 7,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "storage",
                }
            },
            {
                binding: 8,
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
            {binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: {type: "uniform"}},
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
            {binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: {type: "storage"}},
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
        size: this.radixSorter.getAlignedSize(this.canvas.width * this.canvas.height) * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this.compactSpeculativeIDBuffer = device.createBuffer({
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
    this.speculativeRayOffsetBuffer = device.createBuffer({
        size: this.scanPipeline.getAlignedSize(this.canvas.width * this.canvas.height) * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this.scanRayActive = this.scanPipeline.prepareGPUInput(
        this.rayActiveCompactOffsetBuffer,
        this.scanPipeline.getAlignedSize(this.canvas.width * this.canvas.height));
    this.scanRayAfterActive = this.scanPipeline.prepareGPUInput(
        this.speculativeRayOffsetBuffer,
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
            {
                binding: 7,
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

    console.log(`total blocks ${this.totalBlocks}`);
    const groupThreadCount = 32;
    this.numWorkGroups = Math.ceil(this.totalBlocks / groupThreadCount);
    console.log(`num work groups ${this.numWorkGroups}`);
    var cacheInitialSize = Math.ceil(this.totalBlocks * 0.02);
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
        {size: 4 * this.totalBlocks, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC});

    this.blockVisibleBuffer = this.device.createBuffer(
        {size: 4 * this.totalBlocks, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC});

    this.blockNumRaysBuffer = this.device.createBuffer(
        {size: 4 * this.totalBlocks, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC});

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
                    buffer: this.blockRangesBuffer,
                },
            },
            {
                binding: 3,
                resource: {
                    buffer: this.rayInformationBuffer,
                },
            },
            {binding: 4, resource: this.renderTarget.createView()},
            {
                binding: 5,
                resource: {
                    buffer: this.gridIteratorBuffer,
                },
            },
            {
                binding: 6,
                resource: {
                    buffer: this.speculativeRayIDBuffer,
                },
            },
            {
                binding: 7,
                resource: {
                    buffer: this.speculativeRayOffsetBuffer,
                },
            },
            {
                binding: 8,
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
    this.resetBlockNumRaysBG = this.device.createBindGroup({
        layout: this.resetBlockNumRaysBGLayout,
        entries: [
            {binding: 0, resource: {buffer: this.volumeInfoBuffer}},
            {binding: 1, resource: {buffer: this.blockNumRaysBuffer}}
        ]
    });

    this.markBlockActiveBG = this.device.createBindGroup({
        layout: this.markBlockActiveBGLayout,
        entries: [
            {binding: 0, resource: {buffer: this.volumeInfoBuffer}},
            {binding: 1, resource: {buffer: this.LODThresholdBuf}},
            {binding: 2, resource: {buffer: this.viewParamBuf}},
            {binding: 3, resource: {buffer: this.blockActiveBuffer}},
            {binding: 4, resource: {buffer: this.blockNumRaysBuffer}},
            {binding: 5, resource: {buffer: this.rayInformationBuffer}},
            {binding: 6, resource: {buffer: this.blockVisibleBuffer}},
            {binding: 7, resource: {buffer: this.rayBlockIDBuffer}},
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

    this.combinedBlockInformationBuffer = this.device.createBuffer(
        {size: this.totalBlocks * 4 * 4, usage: GPUBufferUsage.STORAGE});

    this.combineBlockInformationBG = this.device.createBindGroup({
        layout: this.combineBlockInformationBGLayout,
        entries: [
            {binding: 0, resource: {buffer: this.combinedBlockInformationBuffer}},
            {binding: 1, resource: {buffer: this.activeBlockIDBuffer}},
            {binding: 2, resource: {buffer: this.blockRayOffsetBuffer}},
            {binding: 3, resource: {buffer: this.blockNumRaysBuffer}},
            {binding: 4, resource: {buffer: this.blockActiveBuffer}},
        ]
    });

    this.rtBlocksPipelineBG1 = this.device.createBindGroup({
        layout: this.rtBlocksPipelineBG1Layout,
        entries: [
            {binding: 0, resource: {buffer: this.viewParamBuf}},
            {binding: 1, resource: {buffer: this.rayInformationBuffer}},
            {binding: 2, resource: {buffer: this.rayIDBuffer}},
            {binding: 3, resource: {buffer: this.combinedBlockInformationBuffer}},
            {binding: 4, resource: this.renderTarget.createView()},
            {binding: 5, resource: {buffer: this.blockRangesBuffer}},
            {binding: 6, resource: {buffer: this.compactSpeculativeIDBuffer}},
            {binding: 7, resource: {buffer: this.rayRGBZBuffer}},
        ]
    });
};

VolumeRaycaster.prototype.computeBlockRanges = async function() {
    // Note: this could be done by the server for us, but for this prototype
    // it's a bit easier to just do it here
    // Decompress each block and compute its value range, output to the blockRangesBuffer
    this.blockRangesBuffer = this.device.createBuffer({
        // Why is this 10 4byte vals?
        // 8 corner values plus 2 range values
        size: this.totalBlocks * 10 * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    this.voxelRangesBuffer = this.device.createBuffer({
        size: this.totalBlocks * 2 * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
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
                    buffer: this.blockRangesBuffer,
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

    // Compute each block's range including its neighbors
    pass.setPipeline(this.computeVoxelRangePipeline);
    pass.setBindGroup(0, bindGroup);
    pass.setBindGroup(2, this.voxelBindGroup);
    for (var i = 0; i < pushConstants.nOffsets; ++i) {
        pass.setBindGroup(1, blockIDOffsetBG, pushConstants.dynamicOffsets, i, 1);
        pass.dispatchWorkgroups(pushConstants.dispatchSizes[i], 1, 1);
    }

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
};

// Progressively compute the surface, returns true when rendering is complete
VolumeRaycaster.prototype.renderSurface =
    async function(isovalue, LODThreshold, viewParamUpload, perfTracker, renderParamsChanged) {
    if (this.renderComplete && !renderParamsChanged) {
        return this.renderComplete;
    }
    console.log("===== Rendering Surface =======");

    if (renderParamsChanged) {
        this.compactTimes = [];
        this.compactTimeSum = 0;
        this.markTimes = [];
        this.markTimeSum = 0;
        this.macroTimes = [];
        this.macroTimeSum = 0;
        this.raytraceTimes = [];
        this.raytraceTimeSum = 0;
        this.decompressTimes = [];
        this.decompressTimeSum = 0;

        this.totalPassTime = 0;
        this.numPasses = 0;
        this.speculationCount = 1;

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

        // We need to reset the speculation count
        console.log(`Upload new speculation count = ${this.speculationCount}`);
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
        console.log(`Compute initial rays took ${end - start} ms`);
        this.initialRayTimes.push(end - start);
        this.initialRayTimeSum += (end - start);
    }
    // for (var i = 0; i < 50; ++i) {
    console.log(`++++ Surface pass ${this.numPasses} ++++`);
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
    console.log(`Macro Traverse: ${end - start}ms`);
    this.macroTimes.push(end - start);
    this.macroTimeSum += (end - start);

    start = performance.now();
    await this.markActiveBlocks();
    end = performance.now();
    console.log(`Mark Active Blocks: ${end - start}ms`);
    this.markTimes.push(end - start);
    this.markTimeSum += (end - start);

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
        this.decompressTimes.push(end - start);
        this.decompressTimeSum += (end - start);
    }

    start = performance.now();
    var numRaysActive = await this.computeBlockRayOffsets();
    end = performance.now();
    console.log(`Ray active and offsets: ${end - start}ms`);
    console.log(`numRaysActive = ${numRaysActive}`);
    if (numRaysActive > 0) {
        // var commandEncoder = this.device.createCommandEncoder();

        // var pass = commandEncoder.beginComputePass();
        // pass.setPipeline(this.initSpeculativeIDsPipeline);
        // pass.setBindGroup(0, this.initSpeculativeIDsBG);
        // pass.dispatchWorkgroups(Math.ceil(this.canvas.width), this.canvas.height, 1);
        // pass.end();

        // this.device.queue.submit([commandEncoder.finish()]);
        // await this.device.queue.onSubmittedWorkDone();

        // Log speculative ray IDs buffer
        // var readbackSpeculativeIDBuffer = this.device.createBuffer({
        //     size: this.speculativeIDBuffer.size,
        //     usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        // });
        // var commandEncoder = this.device.createCommandEncoder();
        // commandEncoder.copyBufferToBuffer(
        //     this.speculativeIDBuffer, 0, readbackSpeculativeIDBuffer, 0,
        //     this.speculativeIDBuffer.size);
        // this.device.queue.submit([commandEncoder.finish()]);
        // await this.device.queue.onSubmittedWorkDone();
        // await readbackSpeculativeIDBuffer.mapAsync(GPUMapMode.READ);
        // var specIDs = new Uint32Array(readbackSpeculativeIDBuffer.getMappedRange());
        // console.log(specIDs);

        start = performance.now();
        var numActiveBlocks = await this.sortActiveRaysByBlock(numRaysActive);
        end = performance.now();
        console.log(`Sort active rays by block: ${end - start}ms`);

        start = performance.now();
        await this.raytraceVisibleBlocks(numActiveBlocks);
        end = performance.now();
        console.log(`Raytrace blocks: ${end - start}ms`);
        this.raytraceTimes.push(end - start);
        this.raytraceTimeSum += (end - start);
        console.log(`PASS TOOK: ${end - startPass}ms`);
        console.log(`++++++++++`);

        var commandEncoder = this.device.createCommandEncoder();
        var pass = commandEncoder.beginComputePass();
        pass.setPipeline(this.depthCompositePipeline);
        pass.setBindGroup(0, this.depthCompositeBG);
        pass.setBindGroup(1, this.depthCompositeBG1);
        pass.dispatchWorkgroups(Math.ceil(this.canvas.width / 32),
                                Math.ceil(this.canvas.height / this.speculationCount),
                                1);
        pass.end();
        this.device.queue.submit([commandEncoder.finish()]);
        await this.device.queue.onSubmittedWorkDone();

        var commandEncoder = this.device.createCommandEncoder();
        var pass = commandEncoder.beginComputePass();
        pass.setPipeline(this.markRayActivePipeline);
        pass.setBindGroup(0, this.markRayActiveBG);
        pass.dispatchWorkgroups(Math.ceil(this.canvas.width / 32), this.canvas.height, 1);
        pass.end();
        // We scan the speculativeRayOffsetBuffer, so copy the ray active information over
        commandEncoder.copyBufferToBuffer(this.rayActiveBuffer,
                                          0,
                                          this.speculativeRayOffsetBuffer,
                                          0,
                                          this.canvas.width * this.canvas.height * 4);
        this.device.queue.submit([commandEncoder.finish()]);

        numRaysActive =
            await this.scanRayAfterActive.scan(this.canvas.width * this.canvas.height);
        console.log(`num rays active after raytracing: ${numRaysActive}`);

        var commandEncoder = this.device.createCommandEncoder();
        this.speculationCount =
            Math.min(Math.floor(this.canvas.width * this.canvas.height / numRaysActive), 64);
        console.log(`Next pass speculation count is ${this.speculationCount}`);
        var uploadSpeculationCount = this.device.createBuffer(
            {size: 4, usage: GPUBufferUsage.COPY_SRC, mappedAtCreation: true});
        new Uint32Array(uploadSpeculationCount.getMappedRange()).set([this.speculationCount]);
        uploadSpeculationCount.unmap();
        commandEncoder.copyBufferToBuffer(
            uploadSpeculationCount, 0, this.viewParamBuf, (16 + 8 + 1 + 1) * 4, 4);
        this.device.queue.submit([commandEncoder.finish()]);
        await this.device.queue.onSubmittedWorkDone();
    }
    console.log("=============");
    this.totalPassTime += end - startPass;
    this.numPasses += 1;
    //}
    this.renderComplete = numRaysActive == 0;
    if (this.renderComplete) {
        console.log(`Avg time per pass ${this.totalPassTime / this.numPasses}ms`);
        // console.log(`Avg compact time per pass ${this.compactTimeSum /
        // this.compactTimes.length})`); console.log(this.compactTimes); console.log(`Avg mark
        // active block time per pass ${this.markTimeSum / this.markTimes.length})`);
        // console.log(this.markTimes);
        // console.log(`Avg macro traverse time per pass ${this.macroTimeSum /
        // this.macroTimes.length})`); console.log(this.macroTimes); console.log(`Avg compute
        // initial rays time ${this.initialRayTimeSum / this.initialRayTimes.length})`);
        // console.log(this.initialRayTimes);
        // console.log(`Avg raytrace time per pass ${this.raytraceTimeSum /
        // this.raytraceTimes.length})`); console.log(this.raytraceTimes); console.log(`Avg
        // decompress time ${this.decompressTimeSum / this.decompressTimes.length})`);
        // console.log(this.decompressTimes);
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
    resetRaysPass.dispatchWorkgroups(Math.ceil(this.canvas.width / 8), this.canvas.height, 1);
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
    resetSpecIDsPass.dispatchWorkgroups(
        Math.ceil(this.canvas.width / 32), this.canvas.height, 1);
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
    pass.dispatchWorkgroups(Math.ceil(this.canvas.width / 64), this.canvas.height, 1);

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

// Mark the active blocks for the current viewpoint/isovalue and count the # of rays
// that we need to process for each block
VolumeRaycaster.prototype.markActiveBlocks = async function() {
    var commandEncoder = this.device.createCommandEncoder();
    var pass = commandEncoder.beginComputePass();

    // Reset the # of rays for each block
    pass.setPipeline(this.resetBlockNumRaysPipeline);
    pass.setBindGroup(0, this.resetBlockNumRaysBG);
    pass.dispatchWorkgroups(
        Math.ceil(this.blockGridDims[0] / 8), this.blockGridDims[1], this.blockGridDims[2]);

    // Compute which blocks are active and how many rays each has
    pass.setPipeline(this.markBlockActivePipeline);
    pass.setBindGroup(0, this.markBlockActiveBG);
    pass.setBindGroup(1, this.renderTargetDebugBG);
    pass.dispatchWorkgroups(Math.ceil(this.canvas.width / 8), this.canvas.height, 1);

    pass.end();

    this.device.queue.submit([commandEncoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();

    // Debugging readback and view the number of rays per block
    /*
    {
        var readbackBlockNumRaysBuffer = this.device.createBuffer({
            size: 4 * this.totalBlocks,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        });

        var commandEncoder = this.device.createCommandEncoder();
        commandEncoder.copyBufferToBuffer(
            this.blockNumRaysBuffer, 0, readbackBlockNumRaysBuffer, 0, 4 * this.totalBlocks);
        this.device.queue.submit([commandEncoder.finish()]);
        await this.device.queue.onSubmittedWorkDone();

        await readbackBlockNumRaysBuffer.mapAsync(GPUMapMode.READ);
        var raysPerBlock = new Uint32Array(readbackBlockNumRaysBuffer.getMappedRange());

        var activeBlocks = 0;
        var avgRaysPerBlock = 0;
        var minRays = 1e20;
        var maxRays = 0;
        for (var i = 0; i < raysPerBlock.length; ++i) {
            if (raysPerBlock[i] > 0) {
                activeBlocks += 1;
                avgRaysPerBlock += raysPerBlock[i];
                minRays = Math.min(raysPerBlock[i], minRays);
                maxRays = Math.max(raysPerBlock[i], maxRays);
            }
        }
        console.log(raysPerBlock[raysPerBlock.length - 1]);
        if (activeBlocks > 0) {
            console.log(`RPB Avg rays per block ${
                avgRaysPerBlock / activeBlocks} (# active = ${activeBlocks})`);
            console.log(`RPB Min rays per block ${minRays}`);
            console.log(`RPB Max rays per block ${maxRays}`);
        }

        readbackBlockNumRaysBuffer.unmap();

        readbackBlockNumRaysBuffer.destroy();
    }
    */
};

// Scan the blockNumRaysBuffer storing the output in blockRayOffsetBuffer and
// return the number of active rays.
VolumeRaycaster.prototype.computeBlockRayOffsets = async function() {
    var commandEncoder = this.device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(
        this.blockNumRaysBuffer, 0, this.blockRayOffsetBuffer, 0, 4 * this.totalBlocks);
    this.device.queue.submit([commandEncoder.finish()]);

    return await this.scanBlockRayOffsets.scan(this.totalBlocks);
};

// Sort the active ray IDs by their block ID in ascending order (inactive rays will be at the
// end).
VolumeRaycaster.prototype.sortActiveRaysByBlock = async function(numRaysActive) {
    // Populate the ray ID, ray block ID and ray active buffers
    var commandEncoder = this.device.createCommandEncoder();
    var pass = commandEncoder.beginComputePass()
    pass.setPipeline(this.writeRayAndBlockIDPipeline);
    pass.setBindGroup(0, this.writeRayAndBlockIDBG);
    pass.dispatchWorkgroups(Math.ceil(this.canvas.width / 8), this.canvas.height, 1);
    pass.end();

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
    // TODO: This is not matching numRaysActive?
    var nactive = await this.scanRayActive.scan(this.canvas.width * this.canvas.height);
    // Should match numRaysActive, sanity check here
    if (numRaysActive != nactive) {
        console.log(`nactive ${nactive} doesn't match numRaysActive ${numRaysActive}!?`);
    }
    var startCompacts = performance.now();
    // Compact the active ray IDs and their block IDs down
    await this.streamCompact.compactActive(this.canvas.width * this.canvas.height,
                                           this.rayActiveBuffer,
                                           this.rayActiveCompactOffsetBuffer,
                                           this.speculativeRayIDBuffer,
                                           this.rayIDBuffer);

    await this.streamCompact.compactActiveIDs(this.canvas.width * this.canvas.height,
                                              this.rayActiveBuffer,
                                              this.rayActiveCompactOffsetBuffer,
                                              this.compactSpeculativeIDBuffer);

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
    this.compactTimes.push(endCompacts - startCompacts);
    this.compactTimeSum += (endCompacts - startCompacts);

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
        var rayInformationInt = new Uint32Array(rayInfoMapped);

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

VolumeRaycaster.prototype.raytraceVisibleBlocks = async function(numActiveBlocks) {
    console.log(
        `Raytracing ${numActiveBlocks} blocks, speculation count = ${this.speculationCount}`);

    // Must recreate each time b/c cache buffer will grow
    var rtBlocksPipelineBG0 = this.device.createBindGroup({
        layout: this.rtBlocksPipelineBG0Layout,
        entries: [
            {binding: 0, resource: {buffer: this.volumeInfoBuffer}},
            {binding: 1, resource: {buffer: this.lruCache.cache}},
            {binding: 2, resource: {buffer: this.lruCache.cachedItemSlots}},
        ]
    });

    var commandEncoder = this.device.createCommandEncoder();
    {
        const groupThreadCount = 64;
        const totalWorkGroups = Math.ceil(numActiveBlocks / groupThreadCount);

        var pushConstants = buildPushConstantsBuffer(
            this.device, totalWorkGroups, new Uint32Array([numActiveBlocks]));

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
        var pushConstants = buildPushConstantsBuffer(this.device, numActiveBlocks);

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
