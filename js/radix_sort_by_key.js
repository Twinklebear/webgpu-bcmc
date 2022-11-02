var RadixSorter = function(device) {
    this.device = device;

    this.bgLayout = this.device.createBindGroupLayout({
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

    this.radixSortBGLayout = this.device.createBindGroupLayout({
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
        ],
    });

    this.mergeBGLayout = this.device.createBindGroupLayout({
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
        ],
    });

    this.numWorkGroupsBGLayout = this.device.createBindGroupLayout({
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

    this.reverseBGLayout = this.device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "storage",
                }
            },
        ],
    });

    this.sortPipeline = this.device.createComputePipeline({
        layout: this.device.createPipelineLayout({
            bindGroupLayouts: [this.bgLayout, this.radixSortBGLayout],
        }),
        compute: {
            module: this.device.createShaderModule({
                code: radix_sort_chunk_comp_spv,
            }),
            entryPoint: "main",
        },
    });

    this.mergePipeline = this.device.createComputePipeline({
        layout: this.device.createPipelineLayout({
            bindGroupLayouts: [
                this.bgLayout,
                this.mergeBGLayout,
                this.numWorkGroupsBGLayout,
            ],
        }),
        compute: {
            module: this.device.createShaderModule({
                code: merge_sorted_chunks_comp_spv,
            }),
            entryPoint: "main",
        },
    });

    this.reversePipeline = this.device.createComputePipeline({
        layout: this.device.createPipelineLayout({
            bindGroupLayouts: [this.bgLayout, this.reverseBGLayout],
        }),
        compute: {
            module: this.device.createShaderModule({code: reverse_buffer_comp_spv}),
            entryPoint: "main",
        },
    });
};

var nextPow2 = function(x) {
    var a = x - 1;
    a |= a >> 1;
    a |= a >> 2;
    a |= a >> 4;
    a |= a >> 8;
    a |= a >> 16;
    return a + 1;
};

RadixSorter.prototype.getAlignedSize = function(size) {
    var chunkCount = nextPow2(Math.ceil(size / SortChunkSize));
    return chunkCount * SortChunkSize;
};

// Input buffers are assumed to be of size "alignedSize"
RadixSorter.prototype.sort = async function(keys, values, size, reverse) {
    // Has to be a pow2 * chunkSize elements, since we do log_2 merge steps up
    var chunkCount = nextPow2(Math.ceil(size / SortChunkSize));
    var alignedSize = chunkCount * SortChunkSize;
    var numMergeSteps = Math.log2(chunkCount);

    var buffers = {
        keys: keys,
        values: values,
    };

    var scratch = {
        keys: this.device.createBuffer({
            size: alignedSize * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        }),
        values: this.device.createBuffer({
            size: alignedSize * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        }),
    };

    var arrayInfoBuf = this.device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_SRC,
        mappedAtCreation: true,
    });
    new Uint32Array(arrayInfoBuf.getMappedRange()).set([size]);
    arrayInfoBuf.unmap();

    // We'll send the workgroup count through a UBO w/ dynamic offset, so we need
    // to obey the dynamic offset alignment rules as well
    var numWorkGroupsBuf = this.device.createBuffer({
        size: Math.max(numMergeSteps, 1) * 256,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_SRC,
        mappedAtCreation: true,
    });
    {
        var upload = new Uint32Array(numWorkGroupsBuf.getMappedRange());
        for (var i = 0; i < numMergeSteps; ++i) {
            upload[i * 64] = chunkCount / (2 << i);
        }
    }
    numWorkGroupsBuf.unmap();

    var infoBindGroup = this.device.createBindGroup({
        layout: this.bgLayout,
        entries: [
            {
                binding: 0,
                resource: {
                    buffer: arrayInfoBuf,
                },
            },
        ],
    });

    var radixSortBG = this.device.createBindGroup({
        layout: this.radixSortBGLayout,
        entries: [
            {
                binding: 0,
                resource: {
                    buffer: buffers.keys,
                },
            },
            {
                binding: 1,
                resource: {
                    buffer: buffers.values,
                },
            },
        ],
    });

    var mergeBindGroups = [
        this.device.createBindGroup({
            layout: this.mergeBGLayout,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: buffers.keys,
                    },
                },
                {
                    binding: 1,
                    resource: {
                        buffer: buffers.values,
                    },
                },
                {
                    binding: 2,
                    resource: {
                        buffer: scratch.keys,
                    },
                },
                {
                    binding: 3,
                    resource: {
                        buffer: scratch.values,
                    },
                },
            ],
        }),
        this.device.createBindGroup({
            layout: this.mergeBGLayout,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: scratch.keys,
                    },
                },
                {
                    binding: 1,
                    resource: {
                        buffer: scratch.values,
                    },
                },
                {
                    binding: 2,
                    resource: {
                        buffer: buffers.keys,
                    },
                },
                {
                    binding: 3,
                    resource: {
                        buffer: buffers.values,
                    },
                },
            ],
        }),
    ];

    var reverseBG = this.device.createBindGroup({
        layout: this.reverseBGLayout,
        entries: [
            {
                binding: 0,
                resource: {
                    buffer: numMergeSteps % 2 == 0 ? buffers.values : scratch.values,
                },
            },
        ],
    });

    var commandEncoder = this.device.createCommandEncoder();
    var pass = commandEncoder.beginComputePass();
    pass.setPipeline(this.sortPipeline);
    pass.setBindGroup(0, infoBindGroup);
    pass.setBindGroup(1, radixSortBG);
    pass.dispatchWorkgroups(chunkCount, 1, 1);
    pass.end();

    // Merge the chunks up
    var pass = commandEncoder.beginComputePass();
    pass.setPipeline(this.mergePipeline);
    pass.setBindGroup(0, infoBindGroup);
    for (var i = 0; i < numMergeSteps; ++i) {
        var numWorkGroupsBG = this.device.createBindGroup({
            layout: this.numWorkGroupsBGLayout,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: numWorkGroupsBuf,
                        size: 4,
                        offset: i * 256,
                    },
                },
            ],
        });
        pass.setBindGroup(1, mergeBindGroups[i % 2]);
        pass.setBindGroup(2, numWorkGroupsBG);
        pass.dispatchWorkgroups(chunkCount / (2 << i), 1, 1);
    }
    pass.end();
    this.device.queue.submit([commandEncoder.finish()]);

    var commandEncoder = this.device.createCommandEncoder();
    if (reverse) {
        var pass = commandEncoder.beginComputePass();
        pass.setPipeline(this.reversePipeline);
        pass.setBindGroup(0, infoBindGroup);
        pass.setBindGroup(1, reverseBG);
        pass.dispatchWorkgroups(Math.ceil(chunkCount / 2), 1, 1);
        pass.end();
    }

    var readbackOffset = reverse ? alignedSize - size : 0;
    // Copy the sorted real data to the start of the buffer
    if (numMergeSteps % 2 == 0) {
        commandEncoder.copyBufferToBuffer(
            buffers.values, readbackOffset * 4, scratch.values, 0, size * 4);
        commandEncoder.copyBufferToBuffer(scratch.values, 0, buffers.values, 0, size * 4);
    } else {
        commandEncoder.copyBufferToBuffer(
            scratch.values, readbackOffset * 4, buffers.values, 0, size * 4);
    }

    this.device.queue.submit([commandEncoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();

    scratch.keys.destroy();
    scratch.values.destroy();
    arrayInfoBuf.destroy();
    numWorkGroupsBuf.destroy();
};
