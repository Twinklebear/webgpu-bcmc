var requestRecompute = false;
var requestBenchmark = null;

var volumes = {
    // TODO: replace with compressed versions
    "Fuel": "7d87jcsh0qodk78/fuel_64x64x64_uint8.raw",
    "Neghip": "zgocya7h33nltu9/neghip_64x64x64_uint8.raw",
    "HydrogenAtom": "jwbav8s3wmmxd5x/hydrogen_atom_128x128x128_uint8.raw",
    "BostonTeapot": "w4y88hlf2nbduiv/boston_teapot_256x256x178_uint8.raw",
    "Engine": "ld2sqwwd3vaq4zf/engine_256x256x128_uint8.raw",
    "Bonsai": "rdnhdxmxtfxe0sa/bonsai_256x256x256_uint8.raw",
    "Foot": "ic0mik3qv4vqacm/foot_256x256x256_uint8.raw",
    "Skull": "5rfjobn0lvb7tmo/skull_256x256x256_uint8.raw",
    "Aneurysm": "3ykigaiym8uiwbp/aneurism_256x256x256_uint8.raw",
};

var fileRegex = /.*\/(\w+)_(\d+)x(\d+)x(\d+)_(\w+)\.*/;

var makeVolumeURL = function(name) {
    if (name.startsWith("local")) {
        return volumes[name];
    }
	return "https://www.dl.dropboxusercontent.com/s/" + volumes[name] + "?dl=1";
}

var getVolumeDimensions = function(name) {
	var m = volumes[name].match(fileRegex);
	return [parseInt(m[2]), parseInt(m[3]), parseInt(m[4])];
}

var getVolumeType = function(name) {
	var m = volumes[name].match(fileRegex);
	return m[5];
}

function recomputeSurface() {
    requestRecompute = true;
}

function runBenchmark(benchmark) {
    requestBenchmark = benchmark;
}

