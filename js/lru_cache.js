// Create the LRU cache and set an initial size for the cache
// If more data is requested to store in the cache than it can fit, it will
// be grown to accomadate it
var LRUCache = function (
  device,
  scanPipeline,
  streamCompact,
  initialSize,
  elementSize,
  totalElements
) {
  this.device = device;
  this.scanPipeline = scanPipeline;
  this.streamCompact = streamCompact;
  this.totalElements = totalElements;
  this.cacheSize = initialSize;
  this.elementSize = elementSize;
  this.numNewItems = 0;

  this.sorter = new RadixSorter(this.device);

  // For each element, track if it's in the cache, and if so where
  var buf = this.device.createBuffer({
    size: totalElements * 4,
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
    size: totalElements * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    mappedAtCreation: true,
  });
  new Uint32Array(buf.getMappedRange()).fill(0);
  buf.unmap();
  this.needsCaching = buf;

  this.needsCachingOffsets = this.device.createBuffer({
    size: this.scanPipeline.getAlignedSize(totalElements) * 4,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_DST |
      GPUBufferUsage.COPY_SRC,
  });

  // The age of each slot
  var buf = this.device.createBuffer({
    size: initialSize * 4,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  // TODO: Just fill with a high value??
  var m = new Uint32Array(buf.getMappedRange());
  for (var i = 0; i < initialSize; ++i) {
    m[i] = 2 + i;
  }
  buf.unmap();
  this.slotAge = buf;

  var buf = this.device.createBuffer({
    size: initialSize * 4,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Uint32Array(buf.getMappedRange()).fill(1);
  buf.unmap();
  this.slotAvailable = buf;

  this.slotAvailableOffsets = this.device.createBuffer({
    size: this.scanPipeline.getAlignedSize(initialSize) * 4,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
  });

  this.slotAvailableIDs = this.device.createBuffer({
    size: initialSize * 4,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
  });

  // List of which item is currently in the cache slot, -1 if unnoccupied
  var buf = this.device.createBuffer({
    size: initialSize * 4,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_DST |
      GPUBufferUsage.COPY_SRC,
    mappedAtCreation: true,
  });
  new Int32Array(buf.getMappedRange()).fill(-1);
  buf.unmap();
  this.slotItemIDs = buf;

  // Buffer used to pass the previous cache size to lru_cache_init.comp
  this.cacheSizeBuf = this.device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // The actual cached data
  this.cache = this.device.createBuffer({
    size: initialSize * elementSize,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_DST |
      GPUBufferUsage.COPY_SRC,
  });

  this.scanNeededSlots = this.scanPipeline.prepareGPUInput(
    this.needsCachingOffsets,
    this.scanPipeline.getAlignedSize(this.totalElements)
  );

  this.slotAvailableScanner = this.scanPipeline.prepareGPUInput(
    this.slotAvailableOffsets,
    this.scanPipeline.getAlignedSize(this.cacheSize)
  );

  this.lruCacheBGLayout = this.device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        type: "storage-buffer",
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        type: "storage-buffer",
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        type: "storage-buffer",
      },
      {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        type: "storage-buffer",
      },
      {
        binding: 4,
        visibility: GPUShaderStage.COMPUTE,
        type: "storage-buffer",
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
          buffer: this.slotAge,
        },
      },
      {
        binding: 2,
        resource: {
          buffer: this.slotAvailable,
        },
      },
      {
        binding: 3,
        resource: {
          buffer: this.slotAvailableIDs,
        },
      },
      {
        binding: 4,
        resource: {
          buffer: this.slotItemIDs,
        },
      },
    ],
  });

  this.markNewItemsBGLayout = this.device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        type: "readonly-storage-buffer",
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        type: "storage-buffer",
      },
    ],
  });

  this.cacheUpdateBGLayout = this.device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        type: "storage-buffer",
      },
    ],
  });

  this.cacheInitBGLayout = this.device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        type: "uniform-buffer",
      },
    ],
  });
  this.cacheInitBG = this.device.createBindGroup({
    layout: this.cacheInitBGLayout,
    entries: [
      {
        binding: 0,
        resource: {
          buffer: this.cacheSizeBuf,
        },
      },
    ],
  });

  this.ageCacheSlotsPipeline = this.device.createComputePipeline({
    layout: this.device.createPipelineLayout({
      bindGroupLayouts: [this.lruCacheBGLayout],
    }),
    compute: {
      module: device.createShaderModule({ code: lru_cache_age_slots_comp_spv }),
      entryPoint: "main",
    },
  });

  // TODO: Exceeds limit of 6 storage buffers, and chrome hasn't implemented
  // GPU limits queries yet
  this.markNewItemsPipeline = this.device.createComputePipeline({
    layout: this.device.createPipelineLayout({
      bindGroupLayouts: [this.lruCacheBGLayout, this.markNewItemsBGLayout],
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
      bindGroupLayouts: [this.lruCacheBGLayout, this.cacheInitBGLayout],
    }),
    compute: {
      module: device.createShaderModule({ code: lru_cache_init_comp_spv }),
      entryPoint: "main",
    },
  });

  this.cacheUpdatePipeline = this.device.createComputePipeline({
    layout: this.device.createPipelineLayout({
      bindGroupLayouts: [this.lruCacheBGLayout, this.cacheUpdateBGLayout],
    }),
    compute: {
      module: device.createShaderModule({ code: lru_cache_update_comp_spv }),
      entryPoint: "main",
    },
  });

  this.copyAvailableSlotAgePipeline = this.device.createComputePipeline({
    layout: this.device.createPipelineLayout({
      bindGroupLayouts: [this.lruCacheBGLayout, this.cacheUpdateBGLayout],
    }),
    compute: {
      module: device.createShaderModule({
        code: lru_copy_available_slot_age_comp_spv,
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
LRUCache.prototype.update = async function (itemNeeded, perfTracker) {
  // TODO WILL: This should also manage the decompression step too,
  // it makes more sense for it to be managed by the cache
  if (!(itemNeeded instanceof GPUBuffer)) {
    alert("itemNeeded info must be a GPUbuffer");
  }
  console.log("------\nCache Update");

  if (perfTracker.lruMarkNewItems === undefined) {
    perfTracker.lruMarkNewItems = [];
    perfTracker.lruScanNeededSlots = [];
    perfTracker.lruNumNewItems = [];

    perfTracker.lruCompactNewItems = [];
    perfTracker.lruScanSlotsAvailable = [];
    perfTracker.lruNumSlotsAvailable = [];
    perfTracker.lruGrowCache = [];
    perfTracker.lruCompactAvailableSlots = [];
    perfTracker.lruPrepKeyValue = [];
    perfTracker.lruTotalSortTime = [];
    perfTracker.lruWriteNewItems = [];
  }

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

  // Age all slots in the cache
  pass.setPipeline(this.ageCacheSlotsPipeline);
  pass.setBindGroup(0, this.lruCacheBG);
  pass.dispatch(this.cacheSize, 1, 1);

  pass.setPipeline(this.markNewItemsPipeline);
  pass.setBindGroup(0, this.lruCacheBG);
  pass.setBindGroup(1, markNewItemsBG);
  pass.dispatch(this.totalElements, 1, 1);
  pass.endPass();
  commandEncoder.copyBufferToBuffer(
    this.needsCaching,
    0,
    this.needsCachingOffsets,
    0,
    this.totalElements * 4
  );
  commandEncoder.copyBufferToBuffer(
    this.slotAvailable,
    0,
    this.slotAvailableOffsets,
    0,
    this.cacheSize * 4
  );
  this.device.queue.submit([commandEncoder.finish()]);
  await this.device.queue.onSubmittedWorkDone();
  var end = performance.now();
  console.log(`Initial aging and mark new items took ${end - start}ms`);
  perfTracker.lruMarkNewItems.push(end - start);

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
  console.log(`Scan needed slots took ${end - start}ms`);
  perfTracker.lruScanNeededSlots.push(end - start);
  perfTracker.lruNumNewItems.push(numNewItems);

  if (numNewItems == 0) {
    console.log("------");
    // Push 0's for unexecuted steps
    perfTracker.lruCompactNewItems.push(0);
    perfTracker.lruScanSlotsAvailable.push(0);
    perfTracker.lruNumSlotsAvailable.push(0);
    perfTracker.lruGrowCache.push(0);
    perfTracker.lruCompactAvailableSlots.push(0);
    perfTracker.lruPrepKeyValue.push(0);
    perfTracker.lruTotalSortTime.push(0);
    perfTracker.lruWriteNewItems.push(0);
    return [0, undefined];
  }
  console.log(`num new items ${numNewItems}`);

  // Compact the IDs of the elements we need into a list of new elements we'll add
  var newItemIDs = this.device.createBuffer({
    size: numNewItems * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  var start = performance.now();
  await this.streamCompact.compactActiveIDs(
    this.totalElements,
    this.needsCaching,
    this.needsCachingOffsets,
    newItemIDs
  );
  var end = performance.now();
  console.log(`Compact new item ids took ${end - start}ms`);
  perfTracker.lruCompactNewItems.push(end - start);

  // Scan the slotAvailable buffer to get a count of the slots we currently
  // have available, based on the items we can evict from the cache. This scan
  // should output to the slotAvailableIDs buffer
  var start = performance.now();
  var numSlotsAvailable = await this.slotAvailableScanner.scan(this.cacheSize);
  var end = performance.now();
  console.log(`slots available ${numSlotsAvailable}`);
  console.log(`Scan slots available took ${end - start}ms`);
  perfTracker.lruScanSlotsAvailable.push(end - start);
  perfTracker.lruNumSlotsAvailable.push(numSlotsAvailable);

  // If there aren't enough slots to hold the new items we need to cache,
  // we have to grow the cache: expand ages, slotAvailable, slotAvailableOffsets,
  // slotAvailableIDs, cache to the new size and copy in the old data.
  // Then run a pass over the slotAvailable buffer to mark the slots as available
  if (numSlotsAvailable < numNewItems) {
    var startGrow = performance.now();

    // We don't need to preserve these buffer's contents so release them first to free space
    this.slotAvailableIDs.destroy();
    this.slotAvailableOffsets.destroy();

    var newSize = Math.min(
      this.cacheSize + Math.ceil((numNewItems - numSlotsAvailable) * 1.5),
      this.totalElements
    );

    var slotAge = this.device.createBuffer({
      size: newSize * 4,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
    });
    var slotAvailable = this.device.createBuffer({
      size: newSize * 4,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
    });
    var slotItemIDs = this.device.createBuffer({
      size: newSize * 4,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
    });
    var cache = this.device.createBuffer({
      size: newSize * this.elementSize,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
    });

    this.slotAvailableIDs = this.device.createBuffer({
      size: newSize * 4,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
    });

    // Update the bindgroup
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
            buffer: slotAge,
          },
        },
        {
          binding: 2,
          resource: {
            buffer: slotAvailable,
          },
        },
        {
          binding: 3,
          resource: {
            buffer: this.slotAvailableIDs,
          },
        },
        {
          binding: 4,
          resource: {
            buffer: slotItemIDs,
          },
        },
      ],
    });

    var commandEncoder = this.device.createCommandEncoder();

    // Initialize the new parts of the buffers
    var pass = commandEncoder.beginComputePass();
    pass.setPipeline(this.cacheInitPipeline);
    pass.setBindGroup(0, this.lruCacheBG);
    pass.setBindGroup(1, this.cacheInitBG);
    pass.dispatch(newSize - this.cacheSize, 1, 1);
    pass.endPass();

    // Copy in the old contents of the buffers to the new ones
    commandEncoder.copyBufferToBuffer(
      this.slotAge,
      0,
      slotAge,
      0,
      this.cacheSize * 4
    );
    commandEncoder.copyBufferToBuffer(
      this.slotAvailable,
      0,
      slotAvailable,
      0,
      this.cacheSize * 4
    );
    commandEncoder.copyBufferToBuffer(
      this.slotItemIDs,
      0,
      slotItemIDs,
      0,
      this.cacheSize * 4
    );
    commandEncoder.copyBufferToBuffer(
      this.cache,
      0,
      cache,
      0,
      this.cacheSize * this.elementSize
    );

    this.device.queue.submit([commandEncoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();

    this.slotAge.destroy();
    this.slotAge = slotAge;

    this.slotAvailable.destroy();
    this.slotAvailable = slotAvailable;

    this.slotItemIDs.destroy();
    this.slotItemIDs = slotItemIDs;

    this.cache.destroy();
    this.cache = cache;

    // Update the slot available offsets data and re-scan it to compute for our larger cache size
    this.slotAvailableOffsets = this.device.createBuffer({
      size: this.scanPipeline.getAlignedSize(newSize) * 4,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
    });

    var commandEncoder = this.device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(
      slotAvailable,
      0,
      this.slotAvailableOffsets,
      0,
      newSize * 4
    );
    this.device.queue.submit([commandEncoder.finish()]);

    // Update available slot IDs w/ a new scan result
    this.slotAvailableScanner = this.scanPipeline.prepareGPUInput(
      this.slotAvailableOffsets,
      this.scanPipeline.getAlignedSize(newSize)
    );
    var end = performance.now();
    console.log(`cache resize took ${end - startGrow}ms`);

    console.log(`prev avail ${numSlotsAvailable}`);
    var start = performance.now();
    numSlotsAvailable = await this.slotAvailableScanner.scan(newSize);
    var end = performance.now();
    console.log(`new avail ${numSlotsAvailable}`);
    console.log(`Scan new cache took ${end - start}ms`);

    perfTracker.lruGrowCache.push(end - startGrow);
    this.cacheSize = newSize;
  } else {
    perfTracker.lruGrowCache.push(0);
  }
  this.displayNumSlotsAvailable = numSlotsAvailable - numNewItems;

  // Compact the slot IDs to get the slots we'll assign to the new
  // data which needs to be cached, and fill those out in the cached item slots
  // For the items which are evicted from the cache by assigning their slot
  // to another item, we have to mark that they're no longer cached
  var start = performance.now();
  await this.streamCompact.compactActiveIDs(
    this.cacheSize,
    this.slotAvailable,
    this.slotAvailableOffsets,
    this.slotAvailableIDs
  );
  var end = performance.now();
  console.log(`num Available ${numSlotsAvailable}`);
  console.log(`cache size ${this.cacheSize}`);
  console.log(`Compact available slot IDs took ${end - start}ms`);
  perfTracker.lruCompactAvailableSlots.push(end - start);

  var start = performance.now();
  // Sort the available slots by their age
  var slotKeys = this.device.createBuffer({
    size: this.sorter.getAlignedSize(numSlotsAvailable) * 4,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_DST |
      GPUBufferUsage.COPY_SRC,
  });
  var sortedIDs = this.device.createBuffer({
    size: this.sorter.getAlignedSize(numSlotsAvailable) * 4,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_DST |
      GPUBufferUsage.COPY_SRC,
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

  var commandEncoder = this.device.createCommandEncoder();
  commandEncoder.copyBufferToBuffer(
    this.slotAvailableIDs,
    0,
    sortedIDs,
    0,
    numSlotsAvailable * 4
  );
  // Run pass to copy the slot ages over
  var pass = commandEncoder.beginComputePass();
  pass.setPipeline(this.copyAvailableSlotAgePipeline);
  pass.setBindGroup(0, this.lruCacheBG);
  pass.setBindGroup(1, outputAgeBG);
  pass.dispatch(numSlotsAvailable, 1, 1);
  pass.endPass();
  this.device.queue.submit([commandEncoder.finish()]);
  await this.device.queue.onSubmittedWorkDone();
  var end = performance.now();
  console.log(`Prep key/value pairs for sort: ${end - start}ms`);
  perfTracker.lruPrepKeyValue.push(end - start);

  var start = performance.now();
  await this.sorter.sort(slotKeys, sortedIDs, numSlotsAvailable, true);
  var end = performance.now();
  console.log(`Sorting ${numSlotsAvailable} ages/slots took ${end - start}ms`);
  perfTracker.lruTotalSortTime.push(end - start);

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
          buffer: this.slotAge,
        },
      },
      {
        binding: 2,
        resource: {
          buffer: this.slotAvailable,
        },
      },
      {
        binding: 3,
        resource: {
          buffer: sortedIDs,
        },
      },
      {
        binding: 4,
        resource: {
          buffer: this.slotItemIDs,
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

  var start = performance.now();
  // Update the slot item IDs with the new items which will be stored in the cache
  var commandEncoder = this.device.createCommandEncoder();
  var pass = commandEncoder.beginComputePass();
  pass.setPipeline(this.cacheUpdatePipeline);
  pass.setBindGroup(0, sortedSlotsBG);
  pass.setBindGroup(1, cacheUpdateBG);
  pass.dispatch(numNewItems, 1, 1);
  pass.endPass();
  this.device.queue.submit([commandEncoder.finish()]);
  await this.device.queue.onSubmittedWorkDone();
  var end = performance.now();
  console.log(`Writing new item slots took ${end - start}ms`);
  perfTracker.lruWriteNewItems.push(end - start);

  slotKeys.destroy();
  sortedIDs.destroy();

  console.log("------");
  // Return the list of blocks which need to be decompressed into the cache
  // The location to write them is found in cachedItemSlots[itemID]
  return [numNewItems, newItemIDs];
};

// Reset the cache to clear items and force them to be decompressed again for benchmarking
LRUCache.prototype.reset = async function () {
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
  pass.dispatch(this.cacheSize, 1, 1);
  pass.endPass();

  // Also need to clear the cached item slots array, just copy the slot item
  // ID array over it, which is also filled with -1
  var numCopies = Math.ceil(this.totalElements / this.cacheSize);
  for (var i = 0; i < numCopies; ++i) {
    var copySize = Math.min(
      this.totalElements - i * this.cacheSize,
      this.cacheSize
    );
    commandEncoder.copyBufferToBuffer(
      this.slotItemIDs,
      0,
      this.cachedItemSlots,
      i * this.cacheSize * 4,
      copySize * 4
    );
  }

  // Copy back over the original cache size to the cache size buffer
  commandEncoder.copyBufferToBuffer(uploadBuf, 4, this.cacheSizeBuf, 0, 4);

  this.device.queue.submit([commandEncoder.finish()]);
  await this.device.queue.onSubmittedWorkDone();

  uploadBuf.destroy();
};
