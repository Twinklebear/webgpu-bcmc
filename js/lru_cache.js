// Create the LRU cache and set an initial size for the cache
// If more data is requested to store in the cache than it can fit, it will
// be grown to accomadate it
var LRUCache = function(
    device, scanPipeline, streamCompact, initialSize, elementSize, totalElements) {
    this.device = device;
    this.scanPipeline = scanPipeline;
    this.streamCompact = streamCompact;

    this.totalElements = totalElements;

    // Round up to the local size of 32 for the cache processing kernels
    this.alignedTotalElements = alignTo(totalElements, 32);
    this.cacheSize = alignTo(initialSize, 32);
    this.elementSize = elementSize;
    this.numNewItems = 0;
    this.maxDispatchSize = device.limits.maxComputeWorkgroupsPerDimension;

    this.sorter = new RadixSorter(this.device);

    // For each element, track if it's in the cache, and if so where
    var buf = this.device.createBuffer({
        size: this.alignedTotalElements * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
    });
    new Int32Array(buf.getMappedRange()).fill(-1);
    buf.unmap();
    this.cachedItemSlots = buf;

    // For each element, if it needs to be added to the cache. This buffer
    // will be 1/0 per total element which could be cached so we can see
    // how many need to be added
    var buf = this.device.createBuffer({
        size: this.alignedTotalElements * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        mappedAtCreation: true,
    });
    new Uint32Array(buf.getMappedRange()).fill(0);
    buf.unmap();
    this.needsCaching = buf;

    this.needsCachingOffsets = this.device.createBuffer({
        size: this.scanPipeline.getAlignedSize(this.alignedTotalElements) * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    // The age of each slot
    var buf = this.device.createBuffer({
        size: this.cacheSize * 4 * 3,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
    });
    var m = new Int32Array(buf.getMappedRange());
    for (var i = 0; i < this.cacheSize; ++i) {
        m[i * 3] = 2 + i;   // For slot age
        m[i * 3 + 1] = 1;   // For slot availability
        m[i * 3 + 2] = -1;  // For slot ID
    }
    buf.unmap();
    this.slotData = buf;

    this.slotAvailableOffsets = this.device.createBuffer({
        size: this.scanPipeline.getAlignedSize(this.cacheSize) * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    // A temp buffer to hold the available slot IDs for compaction
    this.slotAvailableForCompact = this.device.createBuffer({
        size: this.cacheSize * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    this.slotAvailableIDs = this.device.createBuffer({
        size: this.cacheSize * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    // Buffer used to pass the previous cache size to lru_cache_init.comp
    this.cacheSizeBuf = this.device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // The actual cached data
    this.cache = this.device.createBuffer({
        size: this.cacheSize * elementSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    this.scanNeededSlots = this.scanPipeline.prepareGPUInput(
        this.needsCachingOffsets, this.scanPipeline.getAlignedSize(this.totalElements));

    this.slotAvailableScanner = this.scanPipeline.prepareGPUInput(
        this.slotAvailableOffsets, this.scanPipeline.getAlignedSize(this.cacheSize));

    this.lruCacheBGLayout = this.device.createBindGroupLayout({
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
        ],
    });

    this.lruCacheBG = this.device.createBindGroup({
        layout: this.lruCacheBGLayout,
        entries: [
            {
                binding: 0,
                resource: {
                    buffer: this.cachedItemSlots,
                },
            },
            {
                binding: 1,
                resource: {
                    buffer: this.slotAvailableIDs,
                },
            },
            {
                binding: 2,
                resource: {
                    buffer: this.slotData,
                },
            },
        ],
    });

    this.markNewItemsBGLayout = this.device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "read-only-storage",
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

    this.cacheUpdateBGLayout = this.device.createBindGroupLayout({
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

    this.singleUniformBGLayout = this.device.createBindGroupLayout({
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
    this.cacheInitBG = this.device.createBindGroup({
        layout: this.singleUniformBGLayout,
        entries: [
            {
                binding: 0,
                resource: {
                    buffer: this.cacheSizeBuf,
                },
            },
        ],
    });

    this.pushConstantsBGLayout = this.device.createBindGroupLayout({
        entries: [{
            binding: 0,
            visibility: GPUShaderStage.COMPUTE,
            buffer: {
                hasDynamicOffset: true,
                type: "uniform",
            }
        }]
    });

    this.ageCacheSlotsPipeline = this.device.createComputePipeline({
        layout: this.device.createPipelineLayout({
            bindGroupLayouts: [this.lruCacheBGLayout],
        }),
        compute: {
            module: device.createShaderModule({code: lru_cache_age_slots_comp_spv}),
            entryPoint: "main",
        },
    });

    this.markNewItemsPipeline = this.device.createComputePipeline({
        layout: this.device.createPipelineLayout({
            bindGroupLayouts:
                [this.lruCacheBGLayout, this.markNewItemsBGLayout, this.pushConstantsBGLayout],
        }),
        compute: {
            module: device.createShaderModule({
                code: lru_cache_mark_new_items_comp_spv,
            }),
            entryPoint: "main",
        },
    });

    this.cacheInitPipeline = this.device.createComputePipeline({
        layout: this.device.createPipelineLayout({
            bindGroupLayouts: [this.lruCacheBGLayout, this.singleUniformBGLayout],
        }),
        compute: {
            module: device.createShaderModule({code: lru_cache_init_comp_spv}),
            entryPoint: "main",
        },
    });

    this.cacheUpdatePipeline = this.device.createComputePipeline({
        layout: this.device.createPipelineLayout({
            // The last BG layout passes the number of new items
            bindGroupLayouts:
                [this.lruCacheBGLayout, this.cacheUpdateBGLayout, this.singleUniformBGLayout],
        }),
        compute: {
            module: device.createShaderModule({code: lru_cache_update_comp_spv}),
            entryPoint: "main",
        },
    });

    this.copyAvailableSlotAgePipeline = this.device.createComputePipeline({
        layout: this.device.createPipelineLayout({
            bindGroupLayouts:
                [this.lruCacheBGLayout, this.cacheUpdateBGLayout, this.singleUniformBGLayout],
        }),
        compute: {
            module: device.createShaderModule({
                code: lru_copy_available_slot_age_comp_spv,
            }),
            entryPoint: "main",
        },
    });

    this.outputSlotAvailableBG = this.device.createBindGroup({
        layout: this.cacheUpdateBGLayout,
        entries: [{
            binding: 0,
            resource: {
                buffer: this.slotAvailableOffsets,
            }
        }]
    });

    this.outputSlotAvailableForCompact = this.device.createBindGroup({
        layout: this.cacheUpdateBGLayout,
        entries: [{
            binding: 0,
            resource: {
                buffer: this.slotAvailableForCompact,
            }
        }]
    });

    this.extractSlotAvailablePipeline = this.device.createComputePipeline({
        layout: this.device.createPipelineLayout({
            bindGroupLayouts: [this.lruCacheBGLayout, this.cacheUpdateBGLayout],
        }),
        compute: {
            module: device.createShaderModule({
                code: lru_cache_extract_slot_available_comp_spv,
            }),
            entryPoint: "main",
        },
    });
};

// Update the cache based on the externally updated "needsCaching" list
// Items which need to stay in the cache should be marked as needs caching,
// regardless of whether they are currently cached.
// Returns the number of new items which need to be decompressed
// and their IDs
LRUCache.prototype.update = async function(itemNeeded, perfTracker) {
    if (!(itemNeeded instanceof GPUBuffer)) {
        alert("itemNeeded info must be a GPUbuffer");
    }
    perfTracker["lru"] = {};

    var markNewItemsBG = this.device.createBindGroup({
        layout: this.markNewItemsBGLayout,
        entries: [
            {
                binding: 0,
                resource: {
                    buffer: itemNeeded,
                },
            },
            {
                binding: 1,
                resource: {
                    buffer: this.needsCaching,
                },
            },
        ],
    });

    // Update the old cache size UBO so we know where to start initializing the data
    var uploadBuf = this.device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.COPY_SRC,
        mappedAtCreation: true,
    });
    new Uint32Array(uploadBuf.getMappedRange()).set([this.cacheSize]);
    uploadBuf.unmap();

    var start = performance.now();
    // Pass through the needs caching buffer to see which items need caching
    // and are in the cache to unmark them, and which items no longer need
    // caching but are in the cache with age > 2 to mark their slot available.
    var commandEncoder = this.device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(uploadBuf, 0, this.cacheSizeBuf, 0, 4);

    var pass = commandEncoder.beginComputePass();

    /*
    console.log(`LRU: cacheSize = ${this.cacheSize}, totalElements = ${
        this.totalElements}, aligned total = ${this.alignedTotalElements}`);
        */

    // Age all slots in the cache
    pass.setPipeline(this.ageCacheSlotsPipeline);
    pass.setBindGroup(0, this.lruCacheBG);
    pass.dispatchWorkgroups(this.cacheSize / 32, 1, 1);

    // For testing purposes
    /*
    pass.end();
    var start = performance.now();
    this.device.queue.submit([commandEncoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();
    var end = performance.now();
    console.log(`LRU: Age cache slots took ${end - start}ms`);
    var commandEncoder = this.device.createCommandEncoder();
    var pass = commandEncoder.beginComputePass();
    */

    {
        var totalWorkGroups = this.alignedTotalElements / 32;
        var pushConstants = buildPushConstantsBuffer(this.device, totalWorkGroups);
        var pushConstantsBG = this.device.createBindGroup({
            layout: this.pushConstantsBGLayout,
            entries: [{
                binding: 0,
                resource: {
                    buffer: pushConstants.gpuBuffer,
                    size: 8,
                }
            }]
        });

        pass.setPipeline(this.markNewItemsPipeline);
        pass.setBindGroup(0, this.lruCacheBG);
        pass.setBindGroup(1, markNewItemsBG);
        for (var i = 0; i < pushConstants.nOffsets; ++i) {
            pass.setBindGroup(2, pushConstantsBG, pushConstants.dynamicOffsets, i, 1);
            pass.dispatchWorkgroups(pushConstants.dispatchSizes[i], 1, 1);
        }
    }
    // For testing purposes
    /*
    pass.end();
    this.device.queue.submit([commandEncoder.finish()]);
    var start = performance.now();
    await this.device.queue.onSubmittedWorkDone();
    var end = performance.now();
    console.log(`LRU: Mark new items took ${end - start}ms`);
    var commandEncoder = this.device.createCommandEncoder();
    var pass = commandEncoder.beginComputePass();
    pass.setBindGroup(0, this.lruCacheBG);
    */

    // We need a kernel to copy the slotAvailable member out of the structs instead of using
    // copyBufferToBuffer, since it's stored AoS to reduce our buffer use
    pass.setPipeline(this.extractSlotAvailablePipeline);
    pass.setBindGroup(1, this.outputSlotAvailableBG);
    pass.dispatchWorkgroups(this.cacheSize / 32, 1, 1);

    pass.end();
    commandEncoder.copyBufferToBuffer(
        this.needsCaching, 0, this.needsCachingOffsets, 0, this.totalElements * 4);

    this.device.queue.submit([commandEncoder.finish()]);
    // var start = performance.now()
    await this.device.queue.onSubmittedWorkDone();
    var end = performance.now();
    // console.log(`LRU: Extract slots available took ${end - start}ms`);
    perfTracker["lru"]["markNewItems_ms"] = end - start;

    uploadBuf.destroy();

    // Scan the needsCaching buffer to get a count of which items need caching
    // and the available slot offset that they'll be assigned
    // The scan output will need to be written to a separate buffer so I can compact,
    // or I can just merge the compact of both together (like a multi-compact step)
    // where we compact both the slot offset assignments and the item for that
    // new slot together in one step, since the offset to write to in both
    // arrays is the same (the output of the scan on needsCaching)
    // This doesn't really need changes to do a multi-compact, because we can just pass
    // the same compact offset buffer twice
    var start = performance.now();
    var numNewItems = await this.scanNeededSlots.scan(this.totalElements);
    var end = performance.now();
    // console.log(`LRU: Scan needed slots took ${end - start}ms`);
    perfTracker["lru"]["scanNeededSlots_ms"] = end - start;
    perfTracker["lru"]["nNewItems"] = numNewItems;

    // console.log(`LRU: num new items ${numNewItems}`);
    if (numNewItems == 0) {
        return [0, undefined];
    }

    // Compact the IDs of the elements we need into a list of new elements we'll add
    var newItemIDs = this.device.createBuffer({
        size: numNewItems * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    var start = performance.now();
    await this.streamCompact.compactActiveIDs(
        this.totalElements, this.needsCaching, this.needsCachingOffsets, newItemIDs);
    var end = performance.now();
    // console.log(`Compact new item ids took ${end - start}ms`);
    perfTracker["lru"]["compactNewItems_ms"] = end - start;

    // Scan the slotAvailable buffer to get a count of the slots we currently
    // have available, based on the items we can evict from the cache. This scan
    // should output to the slotAvailableIDs buffer
    var start = performance.now();
    var numSlotsAvailable = await this.slotAvailableScanner.scan(this.cacheSize);
    var end = performance.now();
    // console.log(`LRU: Scan slots available took ${end - start}ms`);
    perfTracker["lru"]["scanSlotsAvailable_ms"] = end - start;
    perfTracker["lru"]["nSlotsAvailable"] = numSlotsAvailable;

    // If there aren't enough slots to hold the new items we need to cache,
    // we have to grow the cache: expand ages, slotAvailable, slotAvailableOffsets,
    // slotAvailableIDs, cache to the new size and copy in the old data.
    // Then run a pass over the slotAvailable buffer to mark the slots as available
    if (numSlotsAvailable < numNewItems) {
        var startGrow = performance.now();

        // We don't need to preserve these buffer's contents so release them first to free
        // space
        this.slotAvailableIDs.destroy();
        this.slotAvailableOffsets.destroy();
        this.slotAvailableForCompact.destroy();

        var newSize =
            Math.min(this.cacheSize + Math.ceil((numNewItems - numSlotsAvailable) * 1.5),
                     this.totalElements);
        newSize = alignTo(newSize, 32);

        var slotData = this.device.createBuffer({
            size: newSize * 4 * 3,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        var cache = this.device.createBuffer({
            size: newSize * this.elementSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });

        this.slotAvailableForCompact = this.device.createBuffer({
            size: newSize * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });

        this.slotAvailableIDs = this.device.createBuffer({
            size: newSize * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });

        // Update the bindgroups
        this.lruCacheBG = this.device.createBindGroup({
            layout: this.lruCacheBGLayout,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: this.cachedItemSlots,
                    },
                },
                {
                    binding: 1,
                    resource: {
                        buffer: this.slotAvailableIDs,
                    },
                },
                {
                    binding: 2,
                    resource: {
                        buffer: slotData,
                    },
                },
            ],
        });

        this.outputSlotAvailableForCompact = this.device.createBindGroup({
            layout: this.cacheUpdateBGLayout,
            entries: [{
                binding: 0,
                resource: {
                    buffer: this.slotAvailableForCompact,
                }
            }]
        });

        var commandEncoder = this.device.createCommandEncoder();

        // Initialize the new parts of the buffers
        var pass = commandEncoder.beginComputePass();
        pass.setPipeline(this.cacheInitPipeline);
        pass.setBindGroup(0, this.lruCacheBG);
        pass.setBindGroup(1, this.cacheInitBG);
        // Probably needs to chunk for very large volumes
        pass.dispatchWorkgroups((newSize - this.cacheSize) / 32, 1, 1);
        pass.end();

        // Copy in the old contents of the buffers to the new ones
        commandEncoder.copyBufferToBuffer(
            this.slotData, 0, slotData, 0, this.cacheSize * 4 * 3);
        commandEncoder.copyBufferToBuffer(
            this.cache, 0, cache, 0, this.cacheSize * this.elementSize);

        this.device.queue.submit([commandEncoder.finish()]);
        await this.device.queue.onSubmittedWorkDone();

        this.slotData.destroy();
        this.slotData = slotData;

        this.cache.destroy();
        this.cache = cache;

        // Update the slot available offsets data and re-scan it to compute for our larger
        // cache size
        this.slotAvailableOffsets = this.device.createBuffer({
            size: this.scanPipeline.getAlignedSize(newSize) * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });

        this.outputSlotAvailableBG = this.device.createBindGroup({
            layout: this.cacheUpdateBGLayout,
            entries: [{
                binding: 0,
                resource: {
                    buffer: this.slotAvailableOffsets,
                }
            }]
        });

        var commandEncoder = this.device.createCommandEncoder();

        var pass = commandEncoder.beginComputePass()
        // We need a kernel to copy the slotAvailable member out of the structs instead of
        // using copyBufferToBuffer, since it's stored AoS to reduce our buffer use
        pass.setPipeline(this.extractSlotAvailablePipeline);
        pass.setBindGroup(0, this.lruCacheBG);
        pass.setBindGroup(1, this.outputSlotAvailableBG);
        pass.dispatchWorkgroups(newSize / 32, 1, 1);
        pass.end();

        this.device.queue.submit([commandEncoder.finish()]);

        // Update available slot IDs w/ a new scan result
        this.slotAvailableScanner = this.scanPipeline.prepareGPUInput(
            this.slotAvailableOffsets, this.scanPipeline.getAlignedSize(newSize));

        await this.device.queue.onSubmittedWorkDone();
        var end = performance.now();
        // console.log(`cache resize took ${end - startGrow}ms`);

        var start = performance.now();
        numSlotsAvailable = await this.slotAvailableScanner.scan(newSize);
        var end = performance.now();
        // console.log(`LRU: Resize and scan new cache took ${end - startGrow}ms`);

        perfTracker["lru"]["growCache_ms"] = end - startGrow;
        this.cacheSize = newSize;
    }
    this.displayNumSlotsAvailable = numSlotsAvailable - numNewItems;

    // Compact the slot IDs to get the slots we'll assign to the new
    // data which needs to be cached, and fill those out in the cached item slots
    // For the items which are evicted from the cache by assigning their slot
    // to another item, we have to mark that they're no longer cached
    var start = performance.now();

    // We need a kernel to copy the slotAvailable member out of the structs into a temp array
    // for use by stream compact. Not as great for perf, but a bit lazy here ideally we can
    // undo this buffer use reduction once the limits API is added.
    var commandEncoder = this.device.createCommandEncoder();
    var pass = commandEncoder.beginComputePass()
    pass.setPipeline(this.extractSlotAvailablePipeline);
    pass.setBindGroup(0, this.lruCacheBG);
    pass.setBindGroup(1, this.outputSlotAvailableForCompact);
    // Probably needs to chunk for very large volumes
    pass.dispatchWorkgroups(this.cacheSize / 32, 1, 1);
    pass.end();
    this.device.queue.submit([commandEncoder.finish()]);
    // I don't think we need an await here since it's all on the same queue
    // await this.device.queue.onSubmittedWorkDone();

    await this.streamCompact.compactActiveIDs(this.cacheSize,
                                              this.slotAvailableForCompact,
                                              this.slotAvailableOffsets,
                                              this.slotAvailableIDs);
    var end = performance.now();
    // console.log(`LRU: Compact available slot IDs took ${end - start}ms`);
    perfTracker["lru"]["compactAvailableSlots_ms"] = end - start;

    var start = performance.now();
    // Sort the available slots by their age
    var slotKeys = this.device.createBuffer({
        size: this.sorter.getAlignedSize(numSlotsAvailable) * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    var sortedIDs = this.device.createBuffer({
        size: this.sorter.getAlignedSize(numSlotsAvailable) * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    var outputAgeBG = this.device.createBindGroup({
        layout: this.cacheUpdateBGLayout,
        entries: [
            {
                binding: 0,
                resource: {
                    buffer: slotKeys,
                },
            },
        ],
    });

    var numSlotsAvailableBuf = this.device.createBuffer(
        {size: 4, usage: GPUBufferUsage.UNIFORM, mappedAtCreation: true});
    {
        var uploadArray = new Uint32Array(numSlotsAvailableBuf.getMappedRange());
        uploadArray[0] = numSlotsAvailable;
        numSlotsAvailableBuf.unmap();
    }

    var outputAgeBGSize = this.device.createBindGroup({
        layout: this.singleUniformBGLayout,
        entries: [
            {
                binding: 0,
                resource: {
                    buffer: numSlotsAvailableBuf,
                },
            },
        ],
    });

    // Note: Bit of trick/hack, we only sort the number of entries we need since the
    // sort implementation needs improvement. This helps the LRU cache update cost
    // scale with the number of new items instead of the cache size, at the cost of making
    // it not quite guaranteed that the oldest items are always evicted first.
    var numItemsToSort = numNewItems;
    // var numItemsToSort = numSlotsAvailable;
    console.log(`LRU: numNewItems = ${numNewItems}, numSlotsAvailable = ${numSlotsAvailable}`);
    var commandEncoder = this.device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(
        this.slotAvailableIDs, 0, sortedIDs, 0, numSlotsAvailable * 4);
    // Run pass to copy the slot ages over
    var pass = commandEncoder.beginComputePass();
    pass.setPipeline(this.copyAvailableSlotAgePipeline);
    pass.setBindGroup(0, this.lruCacheBG);
    pass.setBindGroup(1, outputAgeBG);
    pass.setBindGroup(2, outputAgeBGSize);
    pass.dispatchWorkgroups(Math.ceil(numSlotsAvailable / 32), 1, 1);
    pass.end();
    this.device.queue.submit([commandEncoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();
    var end = performance.now();
    // console.log(`LRU: Prep key/value pairs for sort: ${end - start}ms`);
    perfTracker["lru"]["prepKeyValue_ms"] = end - start;

    var start = performance.now();
    await this.sorter.sort(slotKeys, sortedIDs, numItemsToSort, true);
    var end = performance.now();
    // console.log(`LRU: Sorting ${numItemsToSort} ages/slots took ${end - start}ms`);
    perfTracker["lru"]["totalSortTime_ms"] = end - start;

    // Update the bindgroup
    var sortedSlotsBG = this.device.createBindGroup({
        layout: this.lruCacheBGLayout,
        entries: [
            {
                binding: 0,
                resource: {
                    buffer: this.cachedItemSlots,
                },
            },
            {
                binding: 1,
                resource: {
                    buffer: sortedIDs,
                },
            },
            {
                binding: 2,
                resource: {
                    buffer: this.slotData,
                },
            },
        ],
    });

    var cacheUpdateBG = this.device.createBindGroup({
        layout: this.cacheUpdateBGLayout,
        entries: [
            {
                binding: 0,
                resource: {
                    buffer: newItemIDs,
                },
            },
        ],
    });

    var numNewItemsBuf = this.device.createBuffer(
        {size: 4, usage: GPUBufferUsage.UNIFORM, mappedAtCreation: true});
    {
        var uploadArray = new Uint32Array(numNewItemsBuf.getMappedRange());
        uploadArray[0] = numNewItems;
        numNewItemsBuf.unmap();
    }

    var numNewItemsBG = this.device.createBindGroup({
        layout: this.singleUniformBGLayout,
        entries: [
            {
                binding: 0,
                resource: {
                    buffer: numNewItemsBuf,
                },
            },
        ],
    });

    var start = performance.now();
    // Update the slot item IDs with the new items which will be stored in the cache
    var commandEncoder = this.device.createCommandEncoder();
    var pass = commandEncoder.beginComputePass();
    pass.setPipeline(this.cacheUpdatePipeline);
    pass.setBindGroup(0, sortedSlotsBG);
    pass.setBindGroup(1, cacheUpdateBG);
    pass.setBindGroup(2, numNewItemsBG);
    // TODO: Probably needs to chunk for large volumes
    pass.dispatchWorkgroups(Math.ceil(numNewItems / 32), 1, 1);
    pass.end();
    this.device.queue.submit([commandEncoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();
    var end = performance.now();
    // console.log(`LRU: Writing new item slots took ${end - start}ms`);
    perfTracker["lru"]["writeNewItems_ms"] = end - start;

    slotKeys.destroy();
    sortedIDs.destroy();

    // console.log("------");
    // Return the list of blocks which need to be decompressed into the cache
    // The location to write them is found in cachedItemSlots[itemID]
    return [numNewItems, newItemIDs];
};

// Reset the cache to clear items and force them to be decompressed again for benchmarking
LRUCache.prototype.reset = async function() {
    // We just run the same init pipeline used when we grow the cache but
    // just say the cache size is 0 when we run it to clear the whole thing
    var uploadBuf = this.device.createBuffer({
        size: 8,
        usage: GPUBufferUsage.COPY_SRC,
        mappedAtCreation: true,
    });
    new Uint32Array(uploadBuf.getMappedRange()).set([0, this.cacheSize]);
    uploadBuf.unmap();

    var commandEncoder = this.device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(uploadBuf, 0, this.cacheSizeBuf, 0, 4);
    var pass = commandEncoder.beginComputePass();
    pass.setPipeline(this.cacheInitPipeline);
    pass.setBindGroup(0, this.lruCacheBG);
    pass.setBindGroup(1, this.cacheInitBG);
    pass.dispatchWorkgroups(this.cacheSize / 32, 1, 1);
    pass.end();

    // Also need to clear the cached item slots array, just copy the slot item
    // ID array over it, which is also filled with -1
    var buf = this.device.createBuffer({
        size: this.cacheSize * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        mappedAtCreation: true,
    });
    new Int32Array(buf.getMappedRange()).fill(-1);
    buf.unmap();
    negativeBuffer = buf;
    var numCopies = Math.ceil(this.totalElements / this.cacheSize);
    for (var i = 0; i < numCopies; ++i) {
        var copySize = Math.min(this.totalElements - i * this.cacheSize, this.cacheSize);
        commandEncoder.copyBufferToBuffer(
            negativeBuffer, 0, this.cachedItemSlots, i * this.cacheSize * 4, copySize * 4);
    }

    // Copy back over the original cache size to the cache size buffer
    commandEncoder.copyBufferToBuffer(uploadBuf, 4, this.cacheSizeBuf, 0, 4);

    this.device.queue.submit([commandEncoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();

    uploadBuf.destroy();
};
