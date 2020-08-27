var CompressedMarchingCubes = function(device)
{
    this.device = device;
    this.scanPipeline = new ExclusiveScanPipeline(device);
    this.streamCompact = new StreamCompact(device);
    this.numActiveBlocks = 0;
    this.numVertices = 0;
    this.numBlocksWithVertices = 0;

    // Max dispatch size for more computationally heavy kernels
    // which might hit TDR on lower power devices
    this.maxDispatchSize = 512000;

    this.numActiveBlocksStorage = 0;
    this.numBlocksWithVerticesStorage = 0;
    this.numVerticesStorage = 0;

    this.fence = device.defaultQueue.createFence();
    this.fenceValue = 1;

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
                type: "storage-buffer"
            },
            {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                type: "uniform-buffer"
            },
            {
                binding: 2,
                visibility: GPUShaderStage.COMPUTE,
                type: "storage-buffer"
            },
        ]
    });
    this.computeBlockRangePipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({
            bindGroupLayouts: [this.computeBlockRangeBGLayout]
        }),
        computeStage: {
            module: device.createShaderModule({code: zfp_compute_block_range_comp_spv}),
            entryPoint: "main"
        }
    });

    this.computeActiveBlocksBGLayout = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                type: "uniform-buffer"
            },
            {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                type: "storage-buffer"
            },
            {
                binding: 2,
                visibility: GPUShaderStage.COMPUTE,
                type: "storage-buffer"
            }
        ]
    });
    this.computeActiveBlocksPipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({
            bindGroupLayouts: [this.computeActiveBlocksBGLayout]
        }),
        computeStage: {
            module: device.createShaderModule({code: compute_block_active_comp_spv}),
            entryPoint: "main"
        }
    });

    this.decompressBlocksBGLayout = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                type: "storage-buffer"
            },
            {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                type: "uniform-buffer"
            },
            {
                binding: 2,
                visibility: GPUShaderStage.COMPUTE,
                type: "storage-buffer"
            },
            {
                binding: 3,
                visibility: GPUShaderStage.COMPUTE,
                type: "storage-buffer"
            },
            {
                binding: 4,
                visibility: GPUShaderStage.COMPUTE,
                type: "storage-buffer"
            }
        ]
    });
    this.decompressBlocksStartOffsetBGLayout = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                type: "uniform-buffer",
                hasDynamicOffset: true
            }
        ]
    });

    this.decompressBlocksPipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({
            bindGroupLayouts: [this.decompressBlocksBGLayout, this.decompressBlocksStartOffsetBGLayout]
        }),
        computeStage: {
            module: device.createShaderModule({code: zfp_decompress_block_comp_spv}),
            entryPoint: "main"
        }
    });

    this.computeBlockVertsInfoBGLayout = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                type: "uniform-buffer"
            },
            {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                type: "uniform-buffer"
            },
            {
                binding: 2,
                visibility: GPUShaderStage.COMPUTE,
                type: "storage-buffer"
            },
            {
                binding: 3,
                visibility: GPUShaderStage.COMPUTE,
                type: "storage-buffer"
            },
            {
                binding: 4,
                visibility: GPUShaderStage.COMPUTE,
                type: "storage-buffer"
            },
            {
                binding: 5,
                visibility: GPUShaderStage.COMPUTE,
                type: "storage-buffer"
            }
        ]
    });
    this.computeBlockHasVerticesBGLayout = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                type: "storage-buffer"
            }
        ]
    });
    this.computeBlockVerticesBGLayout = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                type: "storage-buffer"
            },
            {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                type: "storage-buffer"
            }
        ]
    });
    this.computeBlockVerticesOutputBGLayout = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                type: "uniform-buffer",
                hasDynamicOffset: true
            },
            {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                type: "storage-buffer"
            }
        ]
    });

    this.computeBlockHasVerticesPipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({
            bindGroupLayouts: [this.computeBlockVertsInfoBGLayout, this.computeBlockHasVerticesBGLayout],
        }),
        computeStage: {
            module: device.createShaderModule({code: compute_block_has_vertices_comp_spv}),
            entryPoint: "main"
        }
    });
    this.computeBlockVertexCountsPipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({
            bindGroupLayouts: [this.computeBlockVertsInfoBGLayout, this.computeBlockVerticesBGLayout],
        }),
        computeStage: {
            module: device.createShaderModule({code: compute_block_voxel_num_verts_comp_spv}),
            entryPoint: "main"
        }
    });

    // TODO: Exceeds limit of 6 storage buffers, and chrome hasn't implemented GPU limits queries yet
    this.computeBlockVerticesPipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({
            bindGroupLayouts: [this.computeBlockVertsInfoBGLayout,
                this.computeBlockVerticesBGLayout,
                this.computeBlockVerticesOutputBGLayout],
        }),
        computeStage: {
            module: device.createShaderModule({code: compute_block_vertices_comp_spv}),
            entryPoint: "main"
        }
    });
}

CompressedMarchingCubes.prototype.reportMemoryUse = function() {
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

    // Data from this object
    var memUse = {
        mc: {
            compressedData: this.compressedDataSize,
            vertexBuffer: this.numVerticesStorage * 2 * 4,
            blockActive: this.totalBlocks * 4,
            activeBlockOffsets: this.scanPipeline.getAlignedSize(this.totalBlocks) * 4,
            blockRanges: this.totalBlocks * 2 * 4,
            activeBlockIDs: this.numActiveBlocksStorage * 4,
            blockHasVertices: this.numActiveBlocksStorage * 4,
            blockHasVertsOffset: this.scanPipeline.getAlignedSize(this.numActiveBlocksStorage),
            blocksWithVertices: this.numBlocksWithVerticesStorage * 4,
            blockVertexOffsets: this.scanPipeline.getAlignedSize(this.numBlocksWithVerticesStorage),
            triTable: triTable.byteLength,
            volumeInfo: 16 * 4,
        },
        cache: {
            cache: this.lruCache.cacheSize * this.lruCache.elementSize,
            cachedItemSlots: this.lruCache.totalElements * 4,
            needsCaching: this.lruCache.totalElements * 4,
            needsCachingOffsets: this.scanPipeline.getAlignedSize(this.lruCache.totalElements) * 4,
            slotAge: this.lruCache.cacheSize * 4,
            slotAvailable: this.lruCache.cacheSize * 4,
            slotAvailableOffsets: this.scanPipeline.getAlignedSize(this.lruCache.cacheSize) * 4,
            slotAvailableIDs: this.lruCache.cacheSize * 4,
            slotItemIDs: this.lruCache.cacheSize * 4,
            cacheSizeBuf: 4,
        }
    };

    var totalMem = 0;
    var mcText = "Marching Cubes Data:<ul>";
    for (const prop in memUse.mc) {
        totalMem += memUse.mc[prop];
        mcText += "<li>" + prop + ": " + formatBytes(memUse.mc[prop]) + "</li>";
    }
    mcText += "</ul>"

    var cacheText = "LRU Cache Data:<ul>";
    for (const prop in memUse.cache) {
        totalMem += memUse.cache[prop];
        cacheText += "<li>" + prop + ": " + formatBytes(memUse.cache[prop]) + "</li>";
    }
    cacheText += "</ul>";
    return [mcText, cacheText, formatBytes(totalMem)];
}

CompressedMarchingCubes.prototype.setCompressedVolume =
    async function(volume, compressionRate, volumeDims, volumeScale)
{
    // Upload the volume
    this.volumeDims = volumeDims;
    this.paddedDims = [alignTo(volumeDims[0], 4), alignTo(volumeDims[1], 4), alignTo(volumeDims[2], 4)]
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

    var volumeInfoBuffer = this.device.createBuffer({
        size: 16 * 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true
    });
    {
        var mapping = volumeInfoBuffer.getMappedRange();
        var maxBits = (1 << (2 * 3)) * compressionRate;
        var buf = new Uint32Array(mapping);
        buf.set(volumeDims)
        buf.set(this.paddedDims, 4);
        buf.set([maxBits], 12);

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

    // Setup buffers, bind groups and scanner for computing active blocks
    this.blockActiveBuffer = this.device.createBuffer({
        size: this.totalBlocks * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });
    this.activeBlockOffsets = this.device.createBuffer({
        size: this.scanPipeline.getAlignedSize(this.totalBlocks) * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    });
    this.computeActiveBlocksBG = this.device.createBindGroup({
        layout: this.computeActiveBlocksBGLayout,
        entries: [
            {
                binding: 0,
                resource: {
                    buffer: this.volumeInfoBuffer
                }
            },
            {
                binding: 1,
                resource: {
                    buffer: this.blockRangesBuffer
                }
            },
            {
                binding: 2,
                resource: {
                    buffer: this.blockActiveBuffer
                }
            }
        ]
    });
    this.activeBlockScanner = this.scanPipeline.prepareGPUInput(this.activeBlockOffsets,
        this.scanPipeline.getAlignedSize(this.totalBlocks));
}

CompressedMarchingCubes.prototype.computeBlockRanges = async function() {
    // Note: this could be done by the server for us, but for this prototype
    // it's a bit easier to just do it here
    // Decompress each block and compute its value range, output to the blockRangesBuffer
    this.blockRangesBuffer = this.device.createBuffer({
        size: this.totalBlocks * 2 * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });

    var bindGroup = this.device.createBindGroup({
        layout: this.computeBlockRangeBGLayout,
        entries: [
            {
                binding: 0,
                resource: {
                    buffer: this.compressedBuffer
                }
            },
            {
                binding: 1,
                resource: {
                    buffer: this.volumeInfoBuffer
                }
            },
            {
                binding: 2,
                resource: {
                    buffer: this.blockRangesBuffer
                }
            }
        ]
    });

    var commandEncoder = this.device.createCommandEncoder();
    var pass = commandEncoder.beginComputePass();

    // Decompress each block and compute its range
    pass.setPipeline(this.computeBlockRangePipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatch(this.numWorkGroups, 1, 1);

    pass.endPass();
    this.device.defaultQueue.submit([commandEncoder.finish()]);
}

CompressedMarchingCubes.prototype.computeSurface = async function(isovalue, perfTracker)
{
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
    await this.uploadIsovalueBuf.mapAsync(GPUMapMode.WRITE)
    new Float32Array(this.uploadIsovalueBuf.getMappedRange()).set([isovalue]);
    this.uploadIsovalueBuf.unmap();

    var commandEncoder = this.device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(this.uploadIsovalueBuf, 0, this.volumeInfoBuffer, 52, 4);
    this.device.defaultQueue.submit([commandEncoder.finish()]);

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

    // Update cache to get offsets within it where we can decompress the new blocks we need to cache
    var start = performance.now();
    // TODO: want a way to explicitly clear the cache so we can time an empty cache update without
    // having the first launch overhead also timed
    var [nBlocksToDecompress, decompressBlockIDs] = await this.lruCache.update(this.blockActiveBuffer, perfTracker);
    var end = performance.now();
    this.newDecompressed = nBlocksToDecompress;
    console.log(`# Blocks to decompress ${nBlocksToDecompress}`);
    console.log(`Cache update took ${end - start}ms`);
    perfTracker.cacheUpdate.push(end - start); 
    perfTracker.numBlocksDecompressed.push(nBlocksToDecompress);

    if (numActiveBlocks > this.numActiveBlocksStorage) {
        this.numActiveBlocksStorage =
            Math.ceil(Math.min(this.totalBlocks,
                Math.max(numActiveBlocks, this.numActiveBlocksStorage * 1.5)));
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
    // Our compacted output will be the "active block index", which is the location of the actual
    // block id within the block_ids/block_offsets/vertex_offsets buffers
    var start = performance.now();
    var commandEncoder = this.device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(this.blockHasVertices, 0,
        this.blockHasVertsOffsets, 0,
        this.numActiveBlocks * 4);
    this.device.defaultQueue.submit([commandEncoder.finish()]);

    var numBlocksWithVertices = await this.blockWithVerticesScanner.scan(this.numActiveBlocks);
    console.log(`Of ${numActiveBlocks} active, only ${numBlocksWithVertices} will output vertices`);
    if (numBlocksWithVertices > this.numBlocksWithVerticesStorage) {
        this.numBlocksWithVerticesStorage =
            Math.floor(Math.min(this.totalBlocks,
                Math.max(numBlocksWithVertices, this.numBlocksWithVerticesStorage * 1.5)));
        var scanAlignedSize = this.scanPipeline.getAlignedSize(this.numBlocksWithVerticesStorage);

        if (this.blocksWithVertices) {
            this.blocksWithVertices.destroy();
            this.blockVertexOffsets.destroy();
        }
        this.blocksWithVertices = this.device.createBuffer({
            size: this.numBlocksWithVerticesStorage * 4,
            usage: GPUBufferUsage.STORAGE
        });
        this.blockVertexOffsets = this.device.createBuffer({
            size: scanAlignedSize * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
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
        this.numVerticesStorage = Math.floor(Math.max(numVertices, this.numVerticesStorage * 1.5));

        if (this.vertexBuffer) {
            this.vertexBuffer.destroy();
        }
        this.vertexBuffer = this.device.createBuffer({
            size: this.numVerticesStorage * 2 * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_SRC
        });
    }
    this.numVertices = numVertices;

    // For each block:
    // Compute the active voxels within the block. We know there should be at least one,
    // since the block + neighboring halo voxels contain the isovalue
    // The active voxel IDs should be written to a single global buffer, so all blocks basically
    // appear as one giant volume
    // When we do the scan here to compute the total number of active voxels we need to keep
    // each blocks offset in the compacted buffer and its output number of voxels 

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
}

CompressedMarchingCubes.prototype.computeActiveBlocks = async function()
{
    var commandEncoder = this.device.createCommandEncoder();
    var pass = commandEncoder.beginComputePass();
    pass.setPipeline(this.computeActiveBlocksPipeline);
    pass.setBindGroup(0, this.computeActiveBlocksBG);
    pass.dispatch(this.paddedDims[0] / 4, this.paddedDims[1] / 4, this.paddedDims[2] / 4);
    pass.endPass();
    commandEncoder.copyBufferToBuffer(this.blockActiveBuffer, 0,
        this.activeBlockOffsets, 0, this.totalBlocks * 4);
    this.device.defaultQueue.submit([commandEncoder.finish()]);

    // Compute total number of active voxels and offsets for each in the compact buffer
    var start = performance.now();
    var totalActive = await this.activeBlockScanner.scan(this.totalBlocks);
    var end = performance.now();
    console.log(`Active block scan took ${end - start}ms`);
    return totalActive;
}

CompressedMarchingCubes.prototype.decompressBlocks = async function(nBlocksToDecompress, decompressBlockIDs)
{
    var decompressBlocksBG = this.device.createBindGroup({
        layout: this.decompressBlocksBGLayout,
        entries: [
            {
                binding: 0,
                resource: {
                    buffer: this.compressedBuffer
                }
            },
            {
                binding: 1,
                resource: {
                    buffer: this.volumeInfoBuffer
                }
            },
            {
                binding: 2,
                resource: {
                    buffer: this.lruCache.cache
                }
            },
            {
                binding: 3,
                resource: {
                    buffer: decompressBlockIDs
                }
            },
            {
                binding: 4,
                resource: {
                    buffer: this.lruCache.cachedItemSlots
                }
            }
        ]
    });

    var numChunks = Math.ceil(nBlocksToDecompress / this.maxDispatchSize);
    var dispatchChunkOffsetsBuf = this.device.createBuffer({
        size: numChunks * 256,
        usage: GPUBufferUsage.UNIFORM,
        mappedAtCreation: true
    });
    var map = new Uint32Array(dispatchChunkOffsetsBuf.getMappedRange());
    for (var i = 0; i < numChunks; ++i) {
        map[i * 64] = i * this.maxDispatchSize;
    }
    dispatchChunkOffsetsBuf.unmap();

    var decompressBlocksStartOffsetBG = this.device.createBindGroup({
        layout: this.decompressBlocksStartOffsetBGLayout,
        entries: [
            {
                binding: 0,
                resource: {
                    buffer: dispatchChunkOffsetsBuf,
                    size: 4,
                    offset: 0
                }
            }
        ]
    });

    // We execute these chunks in separate submissions to avoid having them
    // execute all at once and trigger a TDR if we're decompressing a large amount of data
    for (var i = 0; i < numChunks; ++i) {
        var numWorkGroups = Math.min(nBlocksToDecompress - i * this.maxDispatchSize, this.maxDispatchSize);
        var commandEncoder = this.device.createCommandEncoder();
        var pass = commandEncoder.beginComputePass();
        pass.setPipeline(this.decompressBlocksPipeline);
        pass.setBindGroup(0, decompressBlocksBG);
        pass.setBindGroup(1, decompressBlocksStartOffsetBG, [i * 256]);
        pass.dispatch(numWorkGroups, 1, 1);
        pass.endPass();
        this.device.defaultQueue.submit([commandEncoder.finish()]);
    }

    this.device.defaultQueue.signal(this.fence, this.fenceValue);
    await this.fence.onCompletion(this.fenceValue);
    dispatchChunkOffsetsBuf.destroy();
    this.fenceValue += 1;
}

CompressedMarchingCubes.prototype.computeBlockHasVertices = async function()
{
    this.computeBlockVertexInfoBG = this.device.createBindGroup({
        layout: this.computeBlockVertsInfoBGLayout,
        entries: [
            {
                binding: 0,
                resource: {
                    buffer: this.volumeInfoBuffer
                }
            },
            {
                binding: 1,
                resource: {
                    buffer: this.triTable
                }
            },
            {
                binding: 2,
                resource: {
                    buffer: this.lruCache.cache
                }
            },
            {
                binding: 3,
                resource: {
                    buffer: this.blockActiveBuffer,
                }
            },
            {
                binding: 4,
                resource: {
                    buffer: this.lruCache.cachedItemSlots
                }
            },
            {
                binding: 5,
                resource: {
                    buffer: this.activeBlockIDs
                }
            }
        ]
    });
    var blockHasVerticesBG = this.device.createBindGroup({
        layout: this.computeBlockHasVerticesBGLayout,
        entries: [
            {
                binding: 0,
                resource: {
                    buffer: this.blockHasVertices
                }
            }
        ]
    });

    var commandEncoder = this.device.createCommandEncoder();
    var pass = commandEncoder.beginComputePass();
    pass.setPipeline(this.computeBlockHasVerticesPipeline);
    pass.setBindGroup(0, this.computeBlockVertexInfoBG);
    pass.setBindGroup(1, blockHasVerticesBG);
    pass.dispatch(this.numActiveBlocks, 1, 1);
    pass.endPass();
    this.device.defaultQueue.submit([commandEncoder.finish()]);

    this.device.defaultQueue.signal(this.fence, this.fenceValue);
    await this.fence.onCompletion(this.fenceValue);
    this.fenceValue += 1;
}

CompressedMarchingCubes.prototype.computeBlockVertexCounts = async function()
{
    this.blockVertexOffsetsBG = this.device.createBindGroup({
        layout: this.computeBlockVerticesBGLayout,
        entries: [
            {
                binding: 0,
                resource: {
                    buffer: this.blockVertexOffsets
                }
            },
            {
                binding: 1,
                resource: {
                    buffer: this.blocksWithVertices
                }
            }
        ]
    });

    var commandEncoder = this.device.createCommandEncoder();
    var pass = commandEncoder.beginComputePass();
    pass.setPipeline(this.computeBlockVertexCountsPipeline);
    pass.setBindGroup(0, this.computeBlockVertexInfoBG);
    pass.setBindGroup(1, this.blockVertexOffsetsBG);
    pass.dispatch(this.numBlocksWithVertices, 1, 1);
    pass.endPass();
    this.device.defaultQueue.submit([commandEncoder.finish()]);

    var total = await this.blockVertexOffsetsScanner.scan(this.numBlocksWithVertices);
    return total;
}

CompressedMarchingCubes.prototype.computeVertices = async function()
{
    var numChunks = Math.ceil(this.numBlocksWithVertices / this.maxDispatchSize);
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

    var bindGroup = this.device.createBindGroup({
        layout: this.computeBlockVerticesOutputBGLayout,
        entries: [
            {
                binding: 0,
                resource: {
                    buffer: dispatchChunkOffsetsBuf,
                    size: 4,
                    offset: 0
                }
            },
            {
                binding: 1,
                resource: {
                    buffer: this.vertexBuffer
                }
            }
        ]
    });

    // Chunk up dispatch to avoid hitting TDR
    for (var i = 0; i < numChunks; ++i) {
        var numWorkGroups = Math.min(this.numBlocksWithVertices - i * this.maxDispatchSize, this.maxDispatchSize);
        var commandEncoder = this.device.createCommandEncoder();
        var pass = commandEncoder.beginComputePass();
        pass.setPipeline(this.computeBlockVerticesPipeline);
        pass.setBindGroup(0, this.computeBlockVertexInfoBG);
        pass.setBindGroup(1, this.blockVertexOffsetsBG);
        pass.setBindGroup(2, bindGroup, [i * 256]);
        pass.dispatch(numWorkGroups, 1, 1);
        pass.endPass();
        this.device.defaultQueue.submit([commandEncoder.finish()]);
    }

    this.device.defaultQueue.signal(this.fence, this.fenceValue);
    await this.fence.onCompletion(this.fenceValue);
    dispatchChunkOffsetsBuf.destroy();
    this.fenceValue += 1;
}

