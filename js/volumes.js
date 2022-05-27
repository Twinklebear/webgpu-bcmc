var requestRecompute = false;
var requestBenchmark = null;

var datasets = {
    plane_x: {
        compressionRate: 2,
        name: "plane_x_32x32x32_float32.gen.crate2",
        range: [0, 1],
        scale: [1, 1, 1],
    },
    sphere: {
        compressionRate: 2,
        name: "sphere_32x32x32_float32.gen.crate2",
        range: [0, 1],
        scale: [1, 1, 1],
    },
    quarter_sphere: {
        compressionRate: 2,
        name: "quarter_sphere_32x32x32_float32.gen.crate2",
        range: [0, 1],
        scale: [1, 1, 1],
    },
    wavelet: {
        compressionRate: 2,
        name: "wavelet_32x32x32_float32.gen.crate2",
        range: [-3, 3],
        scale: [1, 1, 1],
    },
    fuel: {
        compressionRate: 4,
        name: "fuel_64x64x64_uint8.raw.crate4",
        range: [10, 255],
        scale: [1, 1, 1],
    },
    aneurism: {
        compressionRate: 4,
        name: "vertebra_512x512x512_uint16.raw.crate4",
        range: [550, 2100],
        scale: [1, 1, 1],
    },
    // For benchmarks:
    skull: {
        compressionRate: 2,
        name: "skull_256x256x256_uint8.raw.crate2",
        range: [10, 255],
        scale: [1, 1, 1],
    },
    foot: {
        compressionRate: 2,
        name: "foot_256x256x256_uint8.raw.crate2",
        range: [10, 255],
        scale: [1, 1, 1],
    },
    backpack: {
        compressionRate: 4,
        name: "backpack_512x512x373_uint16.raw.crate4",
        range: [200, 4000],
        scale: [1, 1, 1],
        step: 1.0 / 3800.0,
    },
    magnetic: {
        compressionRate: 4,
        name: "magnetic_reconnection_512x512x512_float32.raw.crate4",
        range: [0.1, 3.5],
        scale: [1, 1, 1],
        step: 1.0 / 8192,
    },
    stagbeetle: {
        compressionRate: 2,
        name: "stag_beetle_832x832x494_uint16.raw.crate2",
        range: [100, 4096],
        scale: [1, 1, 1],
        step: 1.0 / 4096,
    },
    kingsnake: {
        compressionRate: 2,
        name: "kingsnake_1024x1024x795_uint8.raw.crate2",
        range: [100, 150],
        scale: [1, 1, 1],
    },
    chameleon: {
        compressionRate: 2,
        name: "chameleon_1024x1024x1080_uint16.raw.crate2",
        range: [11000, 52000],
        scale: [1, 1, 1],
        step: 1.0 / 8192,
    },
    miranda: {
        compressionRate: 4,
        name: "miranda_1024x1024x1024_float32.raw.crate4",
        range: [1.05, 2.9],
        scale: [1, 1, 1],
        step: 1.0 / 8192,
    },
    dns_large: {
        compressionRate: 2,
        name: "dns_1920x1440x288_float64.raw.crate2",
        range: [0.75, 1.15],
        scale: [1, 1440 / 1920, 288 / 1920],
        step: 1.0 / 8192,
    },
};

var fileRegex = /(\w+)_(\d+)x(\d+)x(\d+)_(\w+)\.*/;

var getVolumeDimensions = function(filename) {
    var m = filename.match(fileRegex);
    return [parseInt(m[2]), parseInt(m[3]), parseInt(m[4])];
};

function recomputeSurface()
{
    requestRecompute = true;
}

function runBenchmark(benchmark)
{
    requestBenchmark = benchmark;
}
