async function getData(){
  const carsDataResponse = await fetch('https://storage.googleapis.com/tfjs-tutorials/carsData.json');
  const carsData = await carsDataResponse.json();

  const cleaned = carsData.map(car => ({
      wil: car.Weight_in_lbs,
      acceleration: car.Acceleration 
  }))    
  .filter(car => (car.wil != null && car.acceleration != null))

  return cleaned;
}

function createModel() {
  const model = tf.sequential();

  model.add(tf.layers.dense({inputShape: [1], units:1, useBias: true}));

  model.add(tf.layers.dense({units:50,activation: 'sigmoid'}));

  model.add(tf.layers.dense({units:50,activation: 'sigmoid'}));

  model.add(tf.layers.dense({units:40,activation: 'sigmoid'}));

  model.add(tf.layers.dense({units:1, useBias: true}));

  return model;
}

function convertToTensor(data){
  return tf.tidy(()=> {
      tf.util.shuffle(data);

      const inputs = data.map(d => d.acceleration);
      const labels = data.map(d => d.wil);

      const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
      const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

      const inputMax = inputTensor.max();
      const inputMin = inputTensor.min();
      const labelMax = labelTensor.max();
      const labelMin = labelTensor.min();

      const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
      const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

      return {
          inputs: normalizedInputs,
          labels: normalizedLabels,
          inputMax,
          inputMin,
          labelMax,
          labelMin,
      };
  })
}

async function trainModel(model, inputs, labels){
  model.compile({
      optimizer: tf.train.adam(),
      loss: tf.losses.meanSquaredError,
      metrics: ["mse"], 
  });


  const batchSize = 32;
  const epochs = 150;


  return await model.fit(inputs, labels,{
      batchSize,
      epochs,
      shuffle: true,
      callbacks: tfvis.show.fitCallbacks(
          {name: "Performance de Treinamento"},
          ["loss","mse"],
          {height: 200, callbacks: ["onEpochEnd"]}
      ),
  });
}

function testModel(model, inputData, normalizationData){
  const {inputMax, inputMin, labelMax, labelMin} = normalizationData;

  const [xs, preds] = tf.tidy(() =>{
      const xs = tf.linspace(0, 1, 100);
      const preds = model.predict(xs.reshape([100, 1]));

      const unNormXs = xs.mul(inputMax.sub(inputMin)).add(inputMin);
      const unNormPreds = preds.mul(labelMax.sub(labelMin)).add(labelMin);
      return [unNormXs.dataSync(), unNormPreds.dataSync()];
  });

  const predictedPoints = Array.from(xs).map((val, i) => {
      return { x: val, y: preds[i] };
  });

  const originalPoints = inputData.map((d) => {
      return {
          x: d.acceleration,
          y: d.wil,
      }
  });

  tfvis.render.scatterplot(
      {
          name: "Provisões vs Valores originais",
      },
      {
          values: [originalPoints, predictedPoints],
          series: ["Originais", "Predicted"],

      },
      {
          xLabel: "Acceleration",
          yLabel: "WeightInLbs",
          height: 300
      }
  );

  return [originalPoints, predictedPoints]
}

async function run() {
  const data = await getData();
  const values = data.map(d => ({
      x: d.acceleration,
      y: d.wil
  }));

  tfvis.render.scatterplot(
      {name: "acceleration vs wil"},
      {values},
      {
          xLabel: 'Acceleration',
          yLabel: 'WeightInLbs',
          height: 300
      }
  );

  const tensorData = convertToTensor(data);
  const { inputs, labels } = tensorData;

  await trainModel(model, inputs, labels);

  console.log("Treinamento completo");

  let array = testModel(model, data, tensorData);

  let sumx = 0, sumy = 0;
  for(let i = 0; i < 100; i++){
    sumx += Math.abs(array[0][i].x - array[1][i].x); 
    sumy += Math.abs(array[0][i].y - array[1][i].y); 
  }

  console.log(`A disparidade entre as predições do valor x é ${sumx/100}`);
  console.log(`A disparidade entre as predições do valor y é ${sumy/100}`);
}

const model = createModel();

document.addEventListener('DOMContentLoaded', run);

tfvis.show.modelSummary({name:'Modelo'},model);