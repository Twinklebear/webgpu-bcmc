(async () => {
    var adapter = await navigator.gpu.requestAdapter();

    var device = await adapter.requestDevice();

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
    var volumeURL = null;
    if (window.location.hostname == "www.willusher.io") {
        volumeURL = "https://lab.wushernet.com/data/bcmc/" + zfpDataName;
    } else {
        volumeURL = "/models/" + zfpDataName;
    }
    var compressedData =
        await fetch(volumeURL).then((res) => res.arrayBuffer().then(function(arr) {
            return new Uint8Array(arr);
        }));

    if (compressedData == null) {
        alert(`Failed to load compressed data`);
        return;
    }

    var volumeRC = new VolumeRaycaster(device, canvas);
    await volumeRC.setCompressedVolume(
        compressedData, dataset.compressionRate, volumeDims, dataset.scale);
    compressedData = null;

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

    const defaultEye = vec3.set(vec3.create(), 0.0, 0.0, 2.0);
    const center = vec3.set(vec3.create(), 0.0, 0.0, 0.0);
    const up = vec3.set(vec3.create(), 0.0, 1.0, 0.0);
    /*
      const defaultEye = vec3.set(vec3.create(), -0.256, -0.364, -0.009);
      const defaultDir = vec3.set(vec3.create(), 0.507, 0.869, 0.0469);
      const center = vec3.add(vec3.create(), defaultEye, defaultDir);
      const up = vec3.set(vec3.create(), -0.0088, -0.0492, 0.999);
      */
    var camera = new ArcballCamera(defaultEye, center, up, 4, [
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

    var animationFrame = function() {
        var resolve = null;
        var promise = new Promise((r) => (resolve = r));
        window.requestAnimationFrame(resolve);
        return promise;
    };

    requestAnimationFrame(animationFrame);

    var upload = device.createBuffer({
        size: 20 * 4,
        usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC,
    });

    /* We need a render pass to blit the image that is computed by the volume
     * raycaster to the screen. This just draws a quad to the screen and loads
     * the corresponding texel from the render to show on the screen
     */
    var swapChainFormat = "bgra8unorm";
    var swapChain = context.configureSwapChain({
        device: device,
        format: swapChainFormat,
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });

    var vertModule = device.createShaderModule({code: display_render_vert_spv});
    var fragModule = device.createShaderModule({code: display_render_frag_spv});

    var renderBGLayout = device.createBindGroupLayout({
        entries: [{
            binding: 0,
            visibility: GPUShaderStage.FRAGMENT,
            storageTexture: {access: "read-only", format: "rgba8unorm"}
        }]
    });

    var renderPipeline = device.createRenderPipeline({
        layout: device.createPipelineLayout({bindGroupLayouts: [renderBGLayout]}),
        vertex: {
            module: vertModule,
            entryPoint: "main",
        },
        fragment:
            {module: fragModule, entryPoint: "main", targets: [{format: swapChainFormat}]}
    });

    var renderPipelineBG = device.createBindGroup({
        layout: renderBGLayout,
        entries: [{binding: 0, resource: volumeRC.renderTarget.createView()}]
    });

    var renderPassDesc = {
        colorAttachments: [{attachment: undefined, loadValue: [0.3, 0.3, 0.3, 1]}],
    };

    var currentBenchmark = null;

    // Other elements to track are added by the different objects
    var perfResults = {
        isovalue: [],
        totalTime: [],
    };

    var recomputeSurface = true;
    var surfaceDone = false;
    var averageComputeTime = 0;
    while (true) {
        projView = mat4.mul(projView, proj, camera.camera);
        await upload.mapAsync(GPUMapMode.WRITE);
        var uploadArray = new Float32Array(upload.getMappedRange());
        uploadArray.set(projView);
        uploadArray.set(camera.eyePos(), 16);
        upload.unmap();

        if (cameraChanged) {
            cameraChanged = false;
            recomputeSurface = true;

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
            await volumeRC.lruCache.reset();
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
            await volumeRC.lruCache.reset();
        }

        if (isovalueSlider.value != currentIsovalue || requestRecompute) {
            recomputeSurface = true;
            currentIsovalue = parseFloat(isovalueSlider.value);

            /*
            perfResults.isovalue.push(currentIsovalue);
            perfResults.totalTime.push(end - start);

            //displayMCInfo();
            //displayCacheInfo();

            var memUse = compressedMC.reportMemoryUse();
            mcMemDisplay.innerHTML = memUse[0];
            cacheMemDisplay.innerHTML = memUse[1];
            totalMemDisplay.innerHTML = `Total Memory: ${memUse[2]}`;

            requestRecompute = false;
            numFrames = 0;
            totalTimeMS = 0;
            */

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

        if (recomputeSurface || !surfaceDone) {
            var perfTracker = {};
            var start = performance.now();
            surfaceDone = await volumeRC.renderSurface(
                currentIsovalue, upload, perfTracker, recomputeSurface);
            var end = performance.now();
            averageComputeTime = Math.round(volumeRC.totalPassTime / volumeRC.numPasses);
            recomputeSurface = false;
        }

        // Blit the image rendered by the raycaster onto the screen
        var commandEncoder = device.createCommandEncoder();

        renderPassDesc.colorAttachments[0].view = swapChain.getCurrentTexture().createView();
        var renderPass = commandEncoder.beginRenderPass(renderPassDesc);

        renderPass.setPipeline(renderPipeline);
        renderPass.setBindGroup(0, renderPipelineBG);
        // Draw a full screen quad
        renderPass.draw(6, 1, 0, 0);
        renderPass.endPass();
        device.queue.submit([commandEncoder.finish()]);

        // Measure render time by waiting for the work done
        await device.queue.onSubmittedWorkDone();
        var end = performance.now();
        numFrames += 1;
        totalTimeMS += end - start;
        fpsDisplay.innerHTML = `Avg. FPS ${Math.round((1000.0 * numFrames) / totalTimeMS)}<br/>
            Avg. pass time: ${averageComputeTime}ms<br/>
            Total pipeline time: ${volumeRC.totalPassTime}`;
    }
})();
