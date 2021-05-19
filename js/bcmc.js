(async () => {
    var adapter = await navigator.gpu.requestAdapter();

    // TODO: Waiting on Chrome Canary to support passing these limits through
    var gpuDeviceDesc = {
        nonGuaranteedLimits: {
            maxStorageBuffersPerShaderStage: 8,
        },
    };
    var device = await adapter.requestDevice(gpuDeviceDesc);

    var canvas = document.getElementById("webgpu-canvas");
    var context = canvas.getContext("gpupresent");

    var dataset = datasets.skull;
    if (window.location.hash) {
        var name = decodeURI(window.location.hash.substr(1));
        console.log(`Linked to data set ${name}`);
        dataset = datasets[name];
    }
    console.log(`Compression rate ${dataset.compressionRate}`);

    var volumeDims = getVolumeDimensions(dataset.name);
    var zfpDataName = dataset.name + ".zfp";
    var compressedData = await fetch("/models/" + zfpDataName)
                             .then((res) => res.arrayBuffer().then(function(arr) {
                                 return new Uint8Array(arr);
                             }));

    if (compressedData == null) {
        alert(`Failed to load compressed data`);
        return;
    }

    var compressedMC = new CompressedMarchingCubes(device);
    await compressedMC.setCompressedVolume(
        compressedData, dataset.compressionRate, volumeDims, dataset.scale);
    compressedData = null;

    var mcInfo = document.getElementById("mcInfo");
    var cacheInfo = document.getElementById("cacheInfo");
    var totalMemDisplay = document.getElementById("totalMemDisplay");
    var mcMemDisplay = document.getElementById("mcMemDisplay");
    var cacheMemDisplay = document.getElementById("cacheMemDisplay");
    var fpsDisplay = document.getElementById("fps");
    var camDisplay = document.getElementById("camDisplay");

    var enableCache = document.getElementById("enableCache");
    enableCache.checked = true;

    var isovalueSlider = document.getElementById("isovalue");
    isovalueSlider.min = dataset.range[0];
    isovalueSlider.max = dataset.range[1];
    if (dataset.step !== undefined) {
        isovalueSlider.step = dataset.step;
    } else {
        isovalueSlider.step = (isovalueSlider.max - isovalueSlider.min) / 255.0;
    }
    isovalueSlider.value = (dataset.range[0] + dataset.range[1]) / 2.0;
    var currentIsovalue = isovalueSlider.value;

    // We don't keep the perf results from the first run
    var start = performance.now();
    var totalVerts = await compressedMC.computeSurface(currentIsovalue, {});
    var end = performance.now();
    console.log(`total vertices ${totalVerts} in ${end - start}ms`);

    var displayMCInfo = function() {
        mcInfo.innerHTML = `Extracted surface with ${
      totalVerts / 3
    } triangles in ${(end - start).toFixed(2)}ms.
            Isovalue = ${currentIsovalue}`;
    };
    var displayCacheInfo = function() {
        var percentActive = (compressedMC.numActiveBlocks / compressedMC.totalBlocks) * 100;
        var percentWithVerts =
            (compressedMC.numBlocksWithVertices / compressedMC.numActiveBlocks) * 100;
        cacheInfo.innerHTML = `Cache Space: ${
      compressedMC.lruCache.cacheSize
    } blocks
            (${(
              (compressedMC.lruCache.cacheSize / compressedMC.totalBlocks) *
              100
            ).toFixed(2)} %
            of ${compressedMC.totalBlocks} total blocks)<br/>
            # Newly Decompressed: ${compressedMC.newDecompressed}<br/>
            # Active Blocks: ${
              compressedMC.numActiveBlocks
            } (${percentActive.toFixed(2)}%)<br/>
            # Active with Vertices: ${
              compressedMC.numBlocksWithVertices
            } (${percentWithVerts.toFixed(2)}%)<br/>
            # Cache Slots Available ${
              compressedMC.lruCache.displayNumSlotsAvailable
            }`;
    };
    displayMCInfo();
    displayCacheInfo();

    var memUse = compressedMC.reportMemoryUse();
    mcMemDisplay.innerHTML = memUse[0];
    cacheMemDisplay.innerHTML = memUse[1];
    totalMemDisplay.innerHTML = `Total Memory: ${memUse[2]}`;

    // Render it!
    const defaultEye = vec3.set(vec3.create(), 0.0, 0.0, 1.0);
    const center = vec3.set(vec3.create(), 0.0, 0.0, 0.0);
    const up = vec3.set(vec3.create(), 0.0, 1.0, 0.0);
    /*
      const defaultEye = vec3.set(vec3.create(), -0.256, -0.364, -0.009);
      const defaultDir = vec3.set(vec3.create(), 0.507, 0.869, 0.0469);
      const center = vec3.add(vec3.create(), defaultEye, defaultDir);
      const up = vec3.set(vec3.create(), -0.0088, -0.0492, 0.999);
      */
    var camera = new ArcballCamera(defaultEye, center, up, 2, [
        canvas.width,
        canvas.height,
    ]);
    var proj = mat4.perspective(
        mat4.create(), (50 * Math.PI) / 180.0, canvas.width / canvas.height, 0.01, 1000);
    var projView = mat4.create();

    var numFrames = 0;
    var totalTimeMS = 0;
    var cameraChanged = true;

    var controller = new Controller();
    controller.mousemove = function(prev, cur, evt) {
        if (evt.buttons == 1) {
            cameraChanged = true;
            camera.rotate(prev, cur);
            numFrames = 0;
            totalTimeMS = 0;
        } else if (evt.buttons == 2) {
            cameraChanged = true;
            camera.pan([cur[0] - prev[0], prev[1] - cur[1]]);
            numFrames = 0;
            totalTimeMS = 0;
        }
    };
    controller.wheel = function(amt) {
        cameraChanged = true;
        camera.zoom(amt * 0.05);
        numFrames = 0;
        totalTimeMS = 0;
    };
    controller.pinch = controller.wheel;
    controller.twoFingerDrag = function(drag) {
        cameraChanged = true;
        camera.pan(drag);
        numFrames = 0;
        totalTimeMS = 0;
    };
    controller.registerForCanvas(canvas);

    var swapChainFormat = "bgra8unorm";
    var swapChain = context.configureSwapChain({
        device: device,
        format: swapChainFormat,
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });

    var depthTexture = device.createTexture({
        size: {
            width: canvas.width,
            height: canvas.height,
            depth: 1,
        },
        format: "depth24plus-stencil8",
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });

    var renderPassDesc = {
        colorAttachments: [
            {
                view: undefined,
                loadValue: [1.0, 1.0, 1.0, 1],
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

    var viewParamsLayout = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                buffer: {
                    type: "uniform",
                }
            },
            {
                binding: 1,
                visibility: GPUShaderStage.VERTEX,
                buffer: {
                    type: "uniform",
                }
            },
        ],
    });

    // The proj_view matrix and eye position
    var viewParamSize = (16 + 4) * 4;
    var viewParamBuf = device.createBuffer({
        size: viewParamSize,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    var viewParamsBindGroup = device.createBindGroup({
        layout: viewParamsLayout,
        entries: [
            {
                binding: 0,
                resource: {
                    buffer: viewParamBuf,
                },
            },
            {
                binding: 1,
                resource: {
                    buffer: compressedMC.volumeInfoBuffer,
                },
            },
        ],
    });

    var renderPipeline = device.createRenderPipeline({
        layout: device.createPipelineLayout({
            bindGroupLayouts: [viewParamsLayout],
        }),
        vertex: {
            module: device.createShaderModule({code: mc_isosurface_vert_spv}),
            entryPoint: "main",
            buffers: [{
                arrayStride: 2 * 4,
                attributes: [
                    {
                        format: "uint32x2",
                        offset: 0,
                        shaderLocation: 0,
                    },
                ]
            }]
        },
        fragment: {
            module: device.createShaderModule({code: mc_isosurface_frag_spv}),
            entryPoint: "main",
            targets: [{format: swapChainFormat}]
        },
        depthStencil: {
            format: "depth24plus-stencil8",
            depthWriteEnabled: true,
            depthCompare: "less",
        },
    });

    var animationFrame = function() {
        var resolve = null;
        var promise = new Promise((r) => (resolve = r));
        window.requestAnimationFrame(resolve);
        return promise;
    };

    requestAnimationFrame(animationFrame);

    var upload = device.createBuffer({
        size: viewParamSize,
        usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC,
    });

    var currentBenchmark = null;

    // Other elements to track are added by the different objects
    var perfResults = {
        isovalue: [],
        totalTime: [],
    };
    var once = true;
    while (true) {
        projView = mat4.mul(projView, proj, camera.camera);
        await upload.mapAsync(GPUMapMode.WRITE);
        var uploadArray = new Float32Array(upload.getMappedRange());
        uploadArray.set(projView);
        uploadArray.set(camera.eyePos(), 16);
        upload.unmap();

        if (cameraChanged) {
            cameraChanged = false;
            var eyePos = camera.eyePos();
            var eyeDir = camera.eyeDir();
            var upDir = camera.upDir();
            camDisplay.innerHTML = `eye = ${eyePos[0].toFixed(4)} ${eyePos[1].toFixed(
        4
      )} ${eyePos[2].toFixed(4)}<br/>
                dir = ${eyeDir[0].toFixed(4)} ${eyeDir[1].toFixed(
        4
      )} ${eyeDir[2].toFixed(4)}<br/>
                up = ${upDir[0].toFixed(4)} ${upDir[1].toFixed(
        4
      )} ${upDir[2].toFixed(4)}`;
        }

        await animationFrame();
        var start = performance.now();

        if (requestBenchmark && !currentBenchmark) {
            perfResults = {
                isovalue: [],
                totalTime: [],
            };
            await compressedMC.lruCache.reset();
            if (requestBenchmark == "random") {
                currentBenchmark =
                    new RandomIsovalueBenchmark(isovalueSlider, perfResults, dataset.range);
            } else if (requestBenchmark == "sweepUp") {
                currentBenchmark = new SweepIsovalueBenchark(
                    isovalueSlider, perfResults, dataset.range, true);
            } else {
                currentBenchmark = new SweepIsovalueBenchark(
                    isovalueSlider, perfResults, dataset.range, false);
            }
            requestBenchmark = null;
        }

        if (currentBenchmark) {
            if (!currentBenchmark.run()) {
                currentBenchmark = null;
            }
        }

        if (!enableCache.checked) {
            await compressedMC.lruCache.reset();
        }

        if (isovalueSlider.value != currentIsovalue || requestRecompute) {
            currentIsovalue = parseFloat(isovalueSlider.value);

            var start = performance.now();
            totalVerts = await compressedMC.computeSurface(currentIsovalue, perfResults);
            var end = performance.now();
            console.log(`Computation took ${end - start}ms`);

            perfResults.isovalue.push(currentIsovalue);
            perfResults.totalTime.push(end - start);

            displayMCInfo();
            displayCacheInfo();

            var memUse = compressedMC.reportMemoryUse();
            mcMemDisplay.innerHTML = memUse[0];
            cacheMemDisplay.innerHTML = memUse[1];
            totalMemDisplay.innerHTML = `Total Memory: ${memUse[2]}`;

            requestRecompute = false;
            numFrames = 0;
            totalTimeMS = 0;

            // TODO: We'll want to only print this after the benchmark run is done
            /*
                  console.log(JSON.stringify(perfResults));
                  for (const prop in perfResults) {
                      var sum = perfResults[prop].reduce(function(acc, x) { return acc + x; });
                      console.log(`${prop} average = ${(sum /
               perfResults[prop].length).toFixed(3)}`);
                  }
                  */
        }

        renderPassDesc.colorAttachments[0].view = swapChain.getCurrentTexture().createView();

        var commandEncoder = device.createCommandEncoder();

        commandEncoder.copyBufferToBuffer(upload, 0, viewParamBuf, 0, viewParamSize);

        var renderPass = commandEncoder.beginRenderPass(renderPassDesc);
        if (totalVerts > 0) {
            renderPass.setPipeline(renderPipeline);
            renderPass.setBindGroup(0, viewParamsBindGroup);
            renderPass.setVertexBuffer(0, compressedMC.vertexBuffer);
            renderPass.draw(totalVerts, 1, 0, 0);
        }
        renderPass.endPass();
        device.queue.submit([commandEncoder.finish()]);

        // Measure render time by waiting for the work done
        await device.queue.onSubmittedWorkDone();
        var end = performance.now();
        numFrames += 1;
        totalTimeMS += end - start;
        fpsDisplay.innerHTML = `Avg. FPS ${Math.round((1000.0 * numFrames) / totalTimeMS)}`;
    }
})();
