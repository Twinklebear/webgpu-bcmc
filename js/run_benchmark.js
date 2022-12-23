const benchmarkIterations = 10;

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
}
