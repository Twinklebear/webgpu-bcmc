// Generate the work group ID offset buffer and the dynamic offset buffer to use for chunking
// up a large compute dispatch. The start of the push constants data will be:
// {
//      u32: global work group id offset
//      u32: totalWorkGroups
//      ...: up to 248 bytes additional data (if any) from the pushConstants parameter,
//           passed as an ArrayBuffer
// }
// ID offset (u32),
function buildPushConstantsBuffer(device, totalWorkGroups, pushConstants)
{
    var dynamicOffsets = [];
    var dispatchSizes = [];

    var numDynamicOffsets =
        Math.ceil(totalWorkGroups / device.limits.maxComputeWorkgroupsPerDimension);
    var idOffsetsBuffer = device.createBuffer({
        size: 256 * numDynamicOffsets,
        usage: GPUBufferUsage.UNIFORM,
        mappedAtCreation: true,
    });
    {
        var pushConstantsView = null;
        if (pushConstants) {
            if (pushConstants.byteLength > 248) {
                throw Error("Push constants must be at most 248b");
            }
            pushConstantsView =
                new Uint8Array(pushConstants.buffer, 0, pushConstants.byteLength);
        }
        var mapping = idOffsetsBuffer.getMappedRange();
        for (var i = 0; i < numDynamicOffsets; ++i) {
            dynamicOffsets.push(i * 256);

            if (i + 1 < numDynamicOffsets) {
                dispatchSizes.push(device.limits.maxComputeWorkgroupsPerDimension);
            } else {
                dispatchSizes.push(totalWorkGroups %
                                   device.limits.maxComputeWorkgroupsPerDimension);
            }

            // Write the push constants data
            var u32view = new Uint32Array(mapping, i * 256, 2);
            u32view.set([device.limits.maxComputeWorkgroupsPerDimension * i, totalWorkGroups]);

            // Copy in any additional push constants data if provided
            if (pushConstantsView) {
                var u8view = new Uint8Array(mapping, i * 256 + 8, 248);
                u8view.set(pushConstantsView);
            }
        }
        idOffsetsBuffer.unmap();
    }
    dynamicOffsets = new Uint32Array(dynamicOffsets);

    return {
        nOffsets: numDynamicOffsets,
        gpuBuffer: idOffsetsBuffer,
        dynamicOffsets: dynamicOffsets,
        dispatchSizes: dispatchSizes,
    };
}

