import { Component, OnInit, ViewChild, ElementRef } from '@angular/core';
import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";
import * as Papa from "papaparse";
declare var Plotly: any;

Papa.parsePromise = function (file) {
  return new Promise(function (complete, error) {
    Papa.parse(file, {
      header: true,
      download: true,
      dynamicTyping: true,
      complete,
      error
    });
  });
};
@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit {
  title = 'dashboard';
  @ViewChild("Graph", { static: true })
  private Graph: ElementRef; 

  ngOnInit() {
    this.run();
  }



  
  private getData = async () => {
    const csv = await Papa.parsePromise(
      "https://docs.google.com/spreadsheets/d/1C947mrOPa2JvFIXOhR8_8jN7-TsE1MWzEF_h3gdMJs4/export?format=csv"
    );
    const cleaned = csv.data
      .map(car => ({
        cars: car.cars,
        time: car.time
      }))
      .filter(car => car.cars != null && car.time != null);

      this.renderScatter("qual-cont", cleaned, ["time", "cars"], {
        title: "Miles/Gallon vs HorsePower",
        xLabel: "Time",
        yLabel: "Cars"
      });
    return cleaned;
  }


  private run = async () => {
    // Load and plot the original input data that we are going to train on.
    const data = await this.getData();
    const values = data.map(d => ({
      x: d.time,
      y: d.cars
    }));

    tfvis.render.scatterplot(
      { name: "time v cars" },
      { values },
      {
        xLabel: "time",
        yLabel: "cars",
        height: 300
      }
    );

    const model = this.createModel();
    // tfvis.show.modelSummary({ name: "Model Summary" }, model);

    const tensorData = this.convertToTensor(data);
    const { inputs, labels } = tensorData;

    // Train the model
    await this.trainModel(model, inputs, labels);
    console.log("Done Training");

    this.testModel(model, data, tensorData);
    // More code will be added below
  }

  createModel = () => {
    const model: tf.Sequential = tf.sequential();
    model.add(tf.layers.dense({ inputShape: [1], units: 50 }));
    model.add(tf.layers.dense({ units: 100, activation: 'sigmoid' }));
    model.add(tf.layers.dense({ units: 1, useBias: true, activation: 'sigmoid' }));
  
    return model;
  }

  private convertToTensor = (data) => {
    return tf.tidy(() => {
      tf.util.shuffle(data);
  
      const inputs = data.map(d => d.time);
      const labels = data.map(d => d.cars);
  
      const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
      const labelTensor = tf.tensor2d(labels, [labels.length, 1]);
  
      const inputMax = inputTensor.max();
      const inputMin = inputTensor.min();
      const labelMax = labelTensor.max();
      const labelMin = labelTensor.min();
  
      const normalizedInputs = inputTensor
        .sub(inputMin)
        .div(inputMax.sub(inputMin));
      const normalizedLabels = labelTensor
        .sub(labelMin)
        .div(labelMax.sub(labelMin));
  
      return {
        inputs: normalizedInputs,
        labels: normalizedLabels,
        inputMax,
        inputMin,
        labelMax,
        labelMin
      };
    });
  }
  private trainModel = async (model, inputs, labels) => {
    model.compile({
      optimizer: tf.train.adam(0.001),
      loss: tf.losses.meanSquaredError,
      metrics: ["mse"]
    });
  
    const batchSize = 32;
    const epochs = 30;
  
    return await model.fit(inputs, labels, {
      batchSize,
      epochs,
      shuffle: true,
      callbacks: tfvis.show.fitCallbacks(
        { name: "Training Performance" },
        ["loss", "mse"],
        { height: 200, callbacks: ["onEpochEnd"] }
      )
    });
  }
  testModel (model, inputData, normalizationData) {
    const { inputMax, inputMin, labelMin, labelMax } = normalizationData;
  
    const [xs, preds] = tf.tidy(() => {
      const xs = tf.linspace(0, 1, 100);
      const preds = model.predict(xs.reshape([100, 1]));
  
      const unNormXs = xs.mul(inputMax.sub(inputMin)).add(inputMin);
  
      const unNormPreds = preds.mul(labelMax.sub(labelMin)).add(labelMin);
  
      return [unNormXs.dataSync(), unNormPreds.dataSync()];
    });
  
    const predictedPoints = Array.from(xs).map((val, i) => {
      return { x: val, y: preds[i] };
    });
  
    const originalPoints = inputData.map(d => ({
      x: d.time,
      y: d.cars
    }));
  
    tfvis.render.scatterplot(
      { name: "Model Predictions vs Original Data" },
      {
        values: [originalPoints, predictedPoints],
        series: ["original", "predicted"]
      },
      {
        xLabel: "time",
        yLabel: "cars",
        height: 300
      }
    );
  }
  


  
renderHistogram = (container, data, column, config) => {
  const columnData = data.map(r => r[column]);

  const columnTrace = {
    name: column,
    x: columnData,
    type: "histogram",
    opacity: 0.7,
    marker: {
      color: "dodgerblue"
    }
  };

  Plotly.newPlot(container, [columnTrace], {
    xaxis: {
      title: config.xLabel,
      range: config.range
    },
    yaxis: { title: "Count" },
    title: config.title
  });
};

renderScatter = (container, data, columns, config) => {
  var trace = {
    x: data.map(r => r[columns[0]]),
    y: data.map(r => r[columns[1]]),
    mode: "markers",
    type: "scatter",
    opacity: 0.7,
    marker: {
      color: "dodgerblue"
    }
  };

  var chartData = [trace];
  this.Graph = Plotly.newPlot( 
    this.Graph.nativeElement, //our viewchild element
    chartData, //data provided
    { 
  // Plotly.newPlot(container, chartData, {
    title: config.title,
    dragmode: 'select',
    hovermode:'closest',
    xaxis: {
      title: config.xLabel
    },
    yaxis: { title: config.yLabel }
  });

  // let myPlot: any;
  // myPlot = document.getElementById(container);
  // myPlot.on('plotly_selected', function(selecteddata) {
  //   const x = [];
  //   const y = [];
  //   alert("You clicked this Plotly chart!");
  //   selecteddata.points.forEach(data => {
  //     x.push(data.x);
  //     y.push(data.y);
  //   });
  //   // document.getElementById('qual-data').innerText = this.mean(x);
  //   console.log(data);
  // });
};

private mean(numbers) {
  var total = 0, i;
  for (i = 0; i < numbers.length; i += 1) {
      total += numbers[i];
  }
  return total / numbers.length;
}

}
