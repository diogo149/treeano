"use strict";

// NOT containing in a function, so that we can access global variables
// (function() {

// TODO parameterize?
var EPSILON = 1e-6;

var defaultSettings = [
  {
    title: "cost",
    onlyRelevantTooltipKeys: true,
    rollingMeanWindow: 1,
    numXTicks: 11,
    numYTicks: 11,
    scales: {
      idx: {
        keys: ["_idx"],
        lower: "min",
        upper: "max",
        scale: "linear"
      },
      cost: {
        keys: ["cost$"],
        scale: "linear",
        lower: "min",
        upper: "max"
      }
    },
    x: {
      key: "_idx",
      scale: "idx"
    },
    ys: [
      {
        keys: ["cost$"],
        scale: "cost",
        color: "category10"
      }
    ]
  },
  {
    title: "time",
    onlyRelevantTooltipKeys: true,
    rollingMeanWindow: 1,
    numXTicks: 11,
    numYTicks: 11,
    scales: {
      idx: {
        keys: ["_idx"],
        lower: "min",
        upper: "max",
        scale: "linear"
      },
      time: {
        keys: ["\\w+_time$"],
        scale: "log",
        lower: "min",
        upper: "max"
      }
    },
    x: {
      key: "_idx",
      scale: "idx"
    },
    ys: [
      {
        keys: ["\\w+_time$"],
        scale: "time",
        color: "category10"
      }
    ]
  },
  {
    title: "hyperparameters",
    onlyRelevantTooltipKeys: true,
    rollingMeanWindow: 1,
    numXTicks: 11,
    numYTicks: 11,
    scales: {
      idx: {
        keys: ["_idx"],
        lower: "min",
        upper: "max",
        scale: "linear"
      },
      hyperparameter: {
        keys: [":hyperparameter$"],
        scale: "linear",
        lower: "min",
        upper: "max"
      }
    },
    x: {
      key: "_idx",
      scale: "idx"
    },
    ys: [
      {
        keys: [":hyperparameter$"],
        scale: "hyperparameter",
        color: "category10"
      }
    ]
  }
];


var monitorData = [];
var allKeys = [];
var settings;

var $window = $(window);
var $mainView = $("#main-view");

/*
modified from:
http://stackoverflow.com/questions/11963352/plot-rolling-moving-average-in-d3-js
*/
function movingAvg(n) {
  return function (points) {
    points = _.filter(points, function(elem) { return !_.isNaN(elem[1]); } );
    points = points.map(function(each, index, array) {
      var to = index + n - 1;
      var subSeq, sum;
      if (to < points.length) {
        subSeq = array.slice(index, to + 1);
        sum = subSeq.reduce(function(a,b) {
          return [a[0] + b[0], a[1] + b[1]];
        });
        return sum.map(function(each) { return each / n; });
      }
      return undefined;
    });
    points = points.filter(function(each) { return typeof each !== 'undefined'; });
    // Note that one could re-interpolate the points
    // to form a basis curve (I think...)
    return points.join("L");
  };
};


function loadMonitorData(callback) {
  $.get(
    "monitor.jsonl",
    undefined,
    undefined,
    "text"
  ).done(
    function(data) {
      // split by new line and remove empty strings
      var jsonStrings = data.split("\n").filter(Boolean);
      var arrayLength = jsonStrings.length;
      // re-read monitor data
      monitorData = [];
      for (var idx = 0; idx < arrayLength; idx++) {
        var elem = jsonStrings[idx];
        var d = JSON.parse(elem);
        d._idx = idx;
        monitorData.push(d);
      }
      allKeys = _.union.apply(null, _.map(monitorData, _.keys));
      if (!_.isUndefined(callback)) {
        callback();
      }
    }
  ).fail(
    function(err) {
      console.log("error loading monitor data");
      console.log(err);
    }
  );
}

function matchedKeys(patterns) {
  /*
   returns all of the keys of monitor data that match any of the input
   list of strings, treated as regexps
   */
  var regexps = _.map(patterns, function(pattern) {
    return new RegExp(pattern);
  });
  return _.filter(allKeys, function(key) {
    return _.some(regexps, function(regexp) {
      return key.match(regexp);
    });
  });
}

function makeScaleFn(scaleData) {
  return function(range) {
    var lower;
    var upper;

    var scaleKeys = matchedKeys(scaleData.keys);
    var vals = _(monitorData)
          .map(function(d) {
            return _.map(scaleKeys, function(k) { return d[k]; });
          })
          .flatten()
          .filter(_.isNumber)
          .value();

    // TODO add option for percentiles
    switch (scaleData.lower) {
    case "min":
      lower = _.min(vals);
      break;
    default:
      if (_.isNumber(scaleData.lower)) {
        lower = scaleData.lower;
      } else {
        throw "Incorrect lower bound: " + scaleData.lower;
      }
    }

    switch (scaleData.upper) {
    case "max":
      upper = _.max(vals);
      break;
    default:
      if (_.isNumber(scaleData.upper)) {
        upper = scaleData.upper;
      } else {
        throw "Incorrect upper bound: " + scaleData.upper;
      }
    }

    var baseScale;
    switch (scaleData.scale) {
    case "linear":
      baseScale = d3.scale.linear();
      break;
    case "log":
      baseScale = d3.scale.log();
      if (lower === 0) {
        lower += EPSILON;
      }
      // need to clamp to not compute very negative values
      baseScale.clamp(true);
      break;
    default:
      throw "Incorrect scale: " + scaleData.scale;
    }
    return baseScale.domain([lower, upper]).range(range);
  };
}

function createChartView() {
  $mainView.empty();

  var $tabs = $("<div/>");
  $mainView.append($tabs);
  var $tabsList = $("<ul/>");
  $tabs.append($tabsList);

  _.forEach(settings, function(chartData, chartIdx) {
    var chartId = "chart-" + chartIdx;
    $tabsList.append(
      $("<li><a href=\"#$ID\">$TITLE</a></li>"
        .replace(/\$ID/g, chartId)
        .replace(/\$TITLE/g, chartData.title))
    );
    var $tab = $("<div/>");
    $tab.attr("id", chartId);
    var $chart = $("<div/>");
    $tabs.append($tab);
    $tab.append($chart);
    $tab.append($("<button/>")
                 .text("Refresh data")
                .click(function() {
                   loadMonitorData(refreshChart);
                 }));
    $tab.append($("<button/>")
                 .text("Refresh chart")
                 .click(function() {
                   refreshChart();
                 }));

    var refreshChart = function() {
      $chart.empty();
      console.log("refreshing chart");

      var margin = {top: 20, right: 20, bottom: 30, left: 40},
          width = 600 - margin.left - margin.right,
          height = 400 - margin.top - margin.bottom;

      var xRange = [0, width];
      var yRange = [height, 0];
      var scales = _.mapValues(chartData.scales, makeScaleFn);
      var x = scales[chartData.x.scale](xRange);
      // use first y value as axis
      var y = scales[chartData.ys[0].scale](yRange);

      var xAxis = d3.svg.axis()
            .scale(x)
            .orient("bottom")
            .ticks(chartData.numXTicks)
            .tickSize(-height);

      var yAxis = d3.svg.axis()
            .scale(y)
            .orient("left")
            .ticks(chartData.numYTicks)
            .tickSize(-width);

      // TODO add axis labels

      var vis = d3.select($chart[0])
        .selectAll("svg")
            .data([monitorData])
        .enter().append("svg:svg")
            .attr("width", width + margin.left + margin.right)
            .attr("height", height + margin.top + margin.bottom)
        .append("g")
            .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

      // gray background
      vis.append("rect")
        .style("fill", "#ddd")
        .attr("width", width)
        .attr("height", height);

      // x axis ticks + vertical lines
      vis.append("g")
        .attr("class", "x axis")
        .attr("transform", "translate(0," + height + ")") // at the bottom
        .call(xAxis);

      // y axis ticks + horizontal lines
      vis.append("g")
        .attr("class", "y axis")
        .call(yAxis);

      vis.selectAll(".line").remove();

      var dataToX = function(d) { return x(d[chartData.x.key]); };

      var legendData = [];
      var relevantTooltipKeys = [chartData.x.key];

      var category10ColorScale = d3.scale.category10();
      _.forEach(chartData.ys, function(yMap) {
        var yKeys = matchedKeys(yMap.keys);
        relevantTooltipKeys = _.union(relevantTooltipKeys, yKeys);
        _.forEach(yKeys, function(yKey) {
          // TODO have each y have it's own window
          var rollingMeanWindow = chartData.rollingMeanWindow;
          var yScale = scales[yMap.scale](yRange);
          var yVis = vis.append("g");

          var color;
          switch (yMap.color) {
          case "random":
            color = "hsl(" + Math.random() * 360 + ",100%,50%)";
            break;
          case "category10":
            color = category10ColorScale(yKey);
            break;
          default:
            color = yMap.color;
          }
          var dataToY = function(d) { return yScale(d[yKey]); };
          var ySelectors = [];
          ySelectors.push(yVis.append("svg:path")
                          .attr("class", "line")
                          .attr("fill", "none")
                          .attr("stroke", color)
                          .attr("stroke-width", 2)
                          .attr("d", d3.svg.line()
                                .x(dataToX)
                                .y(dataToY)
                                .interpolate(movingAvg(rollingMeanWindow))));

          // only show dots if not doing rolling mean
          if (rollingMeanWindow === 1) {
            var validPoints = _.filter(monitorData, function(d) {
              return !(_.isNaN(dataToX(d)) || _.isNaN(dataToY(d)));
            });
            ySelectors.push(yVis.selectAll("circle.line")
                            .data(validPoints)
                          .enter().append("svg:circle")
                            .attr("class", "line")
                            .attr("fill", color)
                            .attr("cx", dataToX)
                            .attr("cy", dataToY)
                            .attr("r", 3));
          }
          // add legend data
          legendData.push({
            key: yKey,
            color: color,
            selectors: ySelectors
          });
        });
      });

      var $tooltip = $("<pre/>").css("position", "absolute");
      $chart.prepend($tooltip);

      var focusLine = vis.append("line")
            .attr("y1", y.range()[0])
            .attr("y2", y.range()[1])
            .style("stroke-width", 2)
            .style("stroke", "black")
            .style("opacity", 0.5)
            .style("fill", "none");

      vis.append("rect")
        .attr("class", "overlay")
        .attr("fill", "none")
        .attr("pointer-events", "all")
          .attr("width", width)
        .attr("height", height)
        .on("mousemove", mousemove);

      var focusX, focusY, focusData;

      var updateFocus = function() {
        // update line
        focusLine.attr("x1", focusX).attr("x2", focusX);

        // update tooltip
        var chartPos = $chart.position();
        // take margin into account because this is raw html
        var translateX = focusX + margin.left;
        var translateY = focusY + margin.right;
        $tooltip.css("transform", "translate(" + translateX + "px," + translateY + "px)");

        // optionally show only the relevant keys for the tooltip
        var jsonData;
        if (chartData.onlyRelevantTooltipKeys) {
          jsonData = {};
          _.forEach(relevantTooltipKeys, function(k) {
            jsonData[k] = focusData[k];
          });
        } else {
          jsonData = focusData;
        }

        var newText = JSON.stringify(jsonData, undefined, 1);
        if ($tooltip.text() !== newText) {
          $tooltip.text(newText);
        }
      };

      var updateFocusThrottled = _.throttle(updateFocus, 100);

      function mousemove(event) {
        var mouse = d3.mouse(this),
            mouseX = mouse[0],
            mouseY = mouse[1],
            x0 = x.invert(mouseX),
            bisector = d3.bisector(function(d) { return d[chartData.x.key]; }).left,
            // set minimum of 1
            i = bisector(monitorData, x0, 1, monitorData.length - 1),
            d0 = monitorData[i - 1],
            d1 = monitorData[i];
        // figure out which of the 2 surrounding points is closer
        if (Math.abs(dataToX(d0) - mouseX) > Math.abs(dataToX(d1) - mouseX)) {
          focusData = d1;
        } else {
          focusData = d0;
        }
        // put focus on x location aligned with samples
        focusX = x(focusData._idx);
        // put focus on y at location of mouse
        focusY = d3.mouse(this)[1];

        updateFocusThrottled();
      }

      // legend
      // NOTE: this should be defined after the overlay for focusing
      // make the legend draggable
      var legendX = height * 0.01;
      var legendY = width * 0.01;
      // TODO save drag state, so legend doesn't move when refreshing chart
      var dragmove = function(d) {
        legendX = legendX + d3.event.dx;
        legendY = legendY + d3.event.dy;
        legend.attr("transform", "translate(" + legendX + "," + legendY + ")");
      };
      var legend = vis.append("g")
            .attr("transform", "translate(" + legendX + "," + legendY + ")")
            .call(d3.behavior.drag().on("drag", dragmove))
            .style("opacity", 0.5)
            .on("mouseenter", function() { legend.style("opacity", 0.8); })
            .on("mouseleave", function() { legend.style("opacity", 0.5); });

      var legendFontSize = 15;  // TODO parameterize?
      _.forEach(legendData, function(d, idx) {
        // based on: http://www.d3noob.org/2014/07/d3js-multi-line-graph-with-automatic.html
        var isActive = true;
        legend.append("g")
          .attr("transform", "translate(0, "+  legendFontSize * idx + ")")
        .append("text")
          .style("fill", d.color)
          .attr("dy", "1em")
          .style("font-size", legendFontSize + "px")
          .style("font-weight", 900)
          .text(d.key)
          .on("click", function() {
            var newOpacity = isActive ? 0.0 : 1.0;
            isActive = !isActive;
            _.forEach(d.selectors, function(s) {
              s.transition(1000).style("opacity", newOpacity);
            });
          });
      });

      var legendBoundingRect = legend[0][0].getBoundingClientRect();
      legend.insert("rect", ":first-child")
        .style("fill", "white")
      // .attr("transform", "translate(0, " + -legendBoundingRect.height + ")")
        .attr("width", legendBoundingRect.width)
        .attr("height", legendBoundingRect.height);

    };

    refreshChart();
  });

  $tabs.tabs({event: "mouseover"});
}

function createEditView() {
  $mainView.empty();
  var textarea = $("<textarea/>");
  textarea.height($window.height() * 0.8);
  textarea.width($window.width() * 0.8);
  textarea.val(JSON.stringify(settings, undefined, 2));
  $mainView.append($("<button/>")
                   .text("Save")
                   .css("display", "block")
                   .click(function() {
                     settings = JSON.parse(textarea.val());

                     // TODO save in local storage (or something like that)
                     createChartView();
                   }));
  $mainView.append(textarea);
}

function createDataView() {
  $mainView.empty();
  $mainView.append($("<h2/>").text("All keys:"));
  $mainView.append($("<pre/>").text(JSON.stringify(allKeys, undefined, 2)));
  $mainView.append($("<h2/>").text("All data:"));
  $mainView.append($("<pre/>").text(JSON.stringify(monitorData, undefined, 2)));
}

$.getJSON("default_settings.json").done(function(data) {
  settings = data;
  // if view is saved, default to chart view
  loadMonitorData(createChartView);
}).fail(function(err) {
  console.log("no default settings found");
  settings = defaultSettings;
  loadMonitorData(createEditView);
});

// })();
