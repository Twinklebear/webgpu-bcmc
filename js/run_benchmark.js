var RandomIsovalueBenchmark =
    function(isovalueSlider, perfResults, range) {
    this.iteration = 0;
    this.isovalueSlider = isovalueSlider;
    this.perfResults = perfResults;
    this.range = range;
}

    RandomIsovalueBenchmark.prototype.run =
        function() {
    if (this.iteration == 100) {
        console.log(JSON.stringify(this.perfResults));
        for (const prop in this.perfResults) {
            var sum = this.perfResults[prop].reduce(function(acc, x) {
                return acc + x;
            });
            console.log(
                `${prop} average = ${(sum / this.perfResults[prop].length).toFixed(3)}`);
        }
        return false;
    }
    var range = this.range[1] - this.range[0];
    this.isovalueSlider.value = Math.random() * range + this.range[0];
    this.iteration += 1;
    return true;
}

var SweepIsovalueBenchark =
    function(isovalueSlider, perfResults, range, sweepUp) {
    this.iteration = 0;
    this.isovalueSlider = isovalueSlider;
    this.perfResults = perfResults;
    this.range = range;
    this.sweepUp = sweepUp;
    if (this.sweepUp) {
        this.currentValue = range[0];
    } else {
        this.currentValue = range[1];
    }
}

    SweepIsovalueBenchark.prototype.run = function() {
    if (this.iteration == 100) {
        console.log(JSON.stringify(this.perfResults));
        for (const prop in this.perfResults) {
            var sum = this.perfResults[prop].reduce(function(acc, x) {
                return acc + x;
            });
            console.log(
                `${prop} average = ${(sum / this.perfResults[prop].length).toFixed(3)}`);
        }
        return false;
    }
    var step = (this.range[1] - this.range[0]) / 100;
    if (this.sweepUp) {
        this.currentValue += step;
    } else {
        this.currentValue -= step;
    }
    this.isovalueSlider.value = this.currentValue;
    this.iteration += 1;
    return true;
}
