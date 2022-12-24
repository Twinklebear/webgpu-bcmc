const benchmarkIterations = 100;
const cameraIterations = 25;

var RandomIsovalueBenchmark = function(isovalueSlider, range) {
    this.name = "random";
    this.iteration = 0;
    this.isovalueSlider = isovalueSlider;
    this.range = range;
    this.numIterations = benchmarkIterations;
};

RandomIsovalueBenchmark.prototype.run = function() {
    if (this.iteration == this.numIterations) {
        return false;
    }
    var range = this.range[1] - this.range[0];
    this.isovalueSlider.value = Math.random() * range + this.range[0];
    this.iteration += 1;
    return true;
};

RandomIsovalueBenchmark.prototype.reset = function() {
    this.iteration = 0;
};

var SweepIsovalueBenchark = function(isovalueSlider, range, sweepUp) {
    this.iteration = 0;
    this.isovalueSlider = isovalueSlider;
    this.range = range;
    this.sweepUp = sweepUp;
    this.numIterations = benchmarkIterations;
    if (this.sweepUp) {
        this.name = "sweepUp";
        this.currentValue = range[0];
    } else {
        this.name = "sweepDown";
        this.currentValue = range[1];
    }
};

SweepIsovalueBenchark.prototype.run = function() {
    if (this.iteration == this.numIterations) {
        return false;
    }
    var step = (this.range[1] - this.range[0]) / benchmarkIterations;
    if (this.sweepUp) {
        this.currentValue += step;
    } else {
        this.currentValue -= step;
    }
    this.isovalueSlider.value = this.currentValue;
    this.iteration += 1;
    return true;
};

SweepIsovalueBenchark.prototype.reset = function() {
    this.iteration = 0;
};

var CameraOrbitBenchmark = function(radius) {
    this.iteration = 0;
    this.name = "cameraOrbit";
    this.numIterations = cameraIterations;
    this.radius = radius;
};

CameraOrbitBenchmark.prototype.run = function() {
    if (this.iteration == this.numIterations) {
        return false;
    }
    const increment = Math.PI * (3.0 - Math.sqrt(5.0));
    const offset = 2.0 / this.numIterations;

    var y = ((this.iteration * offset) - 1.0) + offset / 2.0;
    const r = Math.sqrt(1.0 - y * y);
    const phi = this.iteration * increment;
    var x = r * Math.cos(phi);
    var z = r * Math.sin(phi);

    x *= this.radius;
    y *= this.radius;
    z *= this.radius;

    this.currentPoint = vec3.set(vec3.create(), x, y, z);
    this.iteration += 1;
    return true;
};

CameraOrbitBenchmark.prototype.reset = function() {
    this.iteration = 0;
};

var NestedBenchmark = function(outerLoop, innerLoop) {
    this.name = outerLoop.name + "-" + innerLoop.name;
    this.outerLoop = outerLoop;
    this.innerLoop = innerLoop;
    this.iteration = 0;
};

NestedBenchmark.prototype.run = function() {
    if (this.iteration == 0) {
        this.outerLoop.run();
    }
    if (!this.innerLoop.run()) {
        if (!this.outerLoop.run()) {
            return false;
        }
        this.innerLoop.reset();
        this.innerLoop.run();
    }
    this.iteration += 1;
    return true;
}

