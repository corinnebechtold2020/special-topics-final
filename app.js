// LSTM Sine Wave Dataset Explorer - app.js

// Utility: seeded RNG (mulberry32)
function mulberry32(seed) {
  return function() {
    seed |= 0;
    seed = seed + 0x6D2B79F5 | 0;
    var t = Math.imul(seed ^ seed >>> 15, 1 | seed);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  }
}

// Gaussian sample using Box-Muller, with RNG function
function gaussian(rng, mean=0, std=1){
  let u = 0, v = 0;
  while(u === 0) u = rng();
  while(v === 0) v = rng();
  let z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  return z * std + mean;
}

// Generate dataset
function generateSineDataset(config){
  const {
    numSamples=1000,
    seqLen=50,
    ampMin=0.5, ampMax=1.5,
    freqMin=0.5, freqMax=2.0,
    noiseStd=0.05,
    seed=null
  } = config;

  const rng = seed != null ? mulberry32(seed) : Math.random;
  const xs = [];
  const ys = [];
  const meta = {curves:[], xFull:[]};

  // x coordinates: seqLen + 1 points inclusive from -1.5 to 1. Target is last point x=1
  const totalPoints = seqLen + 1;
  const xFull = new Array(totalPoints);
  for(let i=0;i<totalPoints;i++) xFull[i] = -1.5 + (i/(totalPoints-1))*(1 - (-1.5));

  for(let s=0;s<numSamples;s++){
    const amp = ampMin + rng()*(ampMax-ampMin);
    const freq = freqMin + rng()*(freqMax-freqMin);
    const phase = rng()*Math.PI*2;

    const curve = new Array(totalPoints);
    for(let i=0;i<totalPoints;i++){
      const t = xFull[i];
      const val = amp * Math.sin(2*Math.PI*freq*t + phase);
      const noisy = val + gaussian(rng,0,noiseStd);
      curve[i] = noisy;
    }

    // inputs are first seqLen points, target is last
    const input = curve.slice(0, seqLen).map(v=>[v]); // shape [seqLen,1]
    const target = [curve[seqLen]]; // scalar in array for shape [1]

    xs.push(input);
    ys.push(target);
    meta.curves.push(curve);
  }

  meta.xFull = xFull.slice();

  // Convert to tensors for TF
  // `xs` is already a number[][][]: [numSamples][seqLen][1]
  const xsTensor = tf.tensor3d(xs, [numSamples, seqLen, 1]);
  const ysTensor = tf.tensor2d(ys, [numSamples, 1]);

  return {xs: xsTensor, ys: ysTensor, meta, raw: {xsArray: xs, ysArray: ys}};
}

// Build model
function buildModel(n, units, lr){
  const model = tf.sequential();
  model.add(tf.layers.lstm({units: units, returnSequences:false, inputShape:[n,1]}));
  model.add(tf.layers.dense({units:1}));
  model.compile({optimizer: tf.train.adam(lr), loss: 'meanSquaredError'});
  return model;
}

// Plot samples with Plotly
function plotSamples(meta, startIndex=0, count=1){
  if(!meta || !meta.xFull){
    console.warn('plotSamples: meta or meta.xFull is null');
    return;
  }
  const seqLen = meta.xFull.length - 1; // inputs length
  const plots = [];
  for(let i=0;i<count;i++){
    const idx = (startIndex + i) % meta.curves.length;
    const curve = meta.curves[idx];
    const xIn = meta.xFull.slice(0, seqLen);
    const yIn = curve.slice(0, seqLen);
    plots.push({x:xIn, y:yIn, mode:'lines', name:`sample ${idx}`});
    // target as red dot
    plots.push({x:[meta.xFull[seqLen]], y:[curve[seqLen]], mode:'markers', marker:{color:'red',size:10}, name:`target ${idx}`});
  }

  const layout = {title:`Samples ${startIndex}..${startIndex+count-1}`, xaxis:{range:[-1.5,1], title:'x'}, yaxis:{title:'y'}};
  Plotly.newPlot('samplesPlot', plots, layout, {responsive:true});
}

// Training metrics chart
function updateTrainingMetricsChart(metricsHistory){
  const epochs = metricsHistory.length;
  const epochIdx = Array.from({length:epochs}, (_,i)=>i+1);
  const trainLoss = metricsHistory.map(d=>d.loss);
  const valLoss = metricsHistory.map(d=>d.valLoss==null?null:d.valLoss);

  const traces = [
    {x:epochIdx, y:trainLoss, mode:'lines+markers', name:'train loss'},
    {x:epochIdx, y:valLoss, mode:'lines+markers', name:'val loss'}
  ];
  const layout = {title:'Training Loss', xaxis:{title:'epoch'}, yaxis:{title:'loss'}};
  Plotly.react('metricsPlot', traces, layout);
}

// Predictions vs targets chart (for fixed val subset)
async function updatePredictionsVsTargetsChart(valXsArray, valYsArray, model){
  // valXsArray: array of inputs, shape [nVal][seqLen][1]
  // valYsArray: array of targets [nVal][1]
  const toPredictCount = valXsArray.length;
  const xsTensor = tf.tensor3d(valXsArray, [toPredictCount, valXsArray[0].length, 1]);
  const predsTensor = model.predict(xsTensor);
  const preds = await predsTensor.array();
  predsTensor.dispose(); xsTensor.dispose();

  const trueVals = valYsArray.map(v=>v[0]);
  const predVals = preds.map(v=>v[0]);

  const idx = Array.from({length:toPredictCount}, (_,i)=>i);
  const traces = [
    {x:idx, y:trueVals, mode:'markers', name:'true', marker:{color:'red', size:8}},
    {x:idx, y:predVals, mode:'markers+lines', name:'pred', marker:{color:'blue', size:6}}
  ];
  const layout = {title:'Predictions vs Targets (validation subset)', xaxis:{title:'sample index'}, yaxis:{title:'y'}};
  Plotly.react('predsPlot', traces, layout);
}

// ---- App State and wiring ----
const state = {
  ds: null,
  model: null,
  meta: null,
  raw: null,
  metricsHistory: [],
  stopRequested: false,
  valSubsetCount: 10,
  valXsSubset: null,
  valYsSubset: null
};

// UI helpers
function $(id){return document.getElementById(id)}

function serializeDSInfo(){
  const info = state.ds ? `samples=${state.ds.xs.shape[0]} seqLen=${state.ds.xs.shape[1]}` : 'No dataset';
  $('dsInfo').textContent = info;
}

// Setup event listeners
function setupUI(){
  // Collapsible
  document.querySelectorAll('.collapsible-btn').forEach(btn=>{
    btn.addEventListener('click', ()=>{
      const body = btn.nextElementSibling;
      body.style.display = body.style.display === 'none' ? 'block' : 'none';
    });
  });

  $('btnGen').addEventListener('click', ()=>{
    const cfg = {
      numSamples: parseInt($('numSamples').value),
      seqLen: parseInt($('seqLen').value),
      ampMin: parseFloat($('ampMin').value),
      ampMax: parseFloat($('ampMax').value),
      freqMin: parseFloat($('freqMin').value),
      freqMax: parseFloat($('freqMax').value),
      noiseStd: parseFloat($('noiseStd').value),
      seed: $('seed').value === '' ? null : parseInt($('seed').value)
    };
    if(state.ds){ state.ds.xs.dispose(); state.ds.ys.dispose(); }
    const ds = generateSineDataset(cfg);
    state.ds = {xs: ds.xs, ys: ds.ys};
    state.meta = ds.meta;
    state.raw = ds.raw;
    serializeDSInfo();
    // reset plots
    plotSamples(state.meta, 0, Math.min(3, parseInt($('showCount').value)||1));
  });

  $('btnRender').addEventListener('click', ()=>{
    const start = parseInt($('startIndex').value) || 0;
    const count = Math.min(10, Math.max(1, parseInt($('showCount').value)||1));
    plotSamples(state.meta, start, count);
  });

  $('btnPrev').addEventListener('click', ()=>{
    const start = Math.max(0, (parseInt($('startIndex').value)||0) - (parseInt($('showCount').value)||1));
    $('startIndex').value = start;
    $('btnRender').click();
  });
  $('btnNext').addEventListener('click', ()=>{
    const count = parseInt($('showCount').value)||1;
    const start = (parseInt($('startIndex').value)||0) + count;
    $('startIndex').value = start;
    $('btnRender').click();
  });
  $('btnRandom').addEventListener('click', ()=>{
    const maxStart = Math.max(0, (state.meta?state.meta.curves.length:1)- (parseInt($('showCount').value)||1));
    const r = Math.floor(Math.random()*(maxStart+1));
    $('startIndex').value = r;
    $('btnRender').click();
  });

  $('btnBuild').addEventListener('click', ()=>{
    if(!state.ds) return alert('Generate dataset first');
    if(state.model){ state.model.dispose(); state.model = null; }
    const units = parseInt($('units').value)||32;
    const lr = parseFloat($('learningRate').value)||0.001;
    const seqLen = parseInt($('seqLen').value)||50;
    state.model = buildModel(seqLen, units, lr);
    $('trainLog').textContent = 'Model built.';
    state.metricsHistory = [];
    Plotly.purge('metricsPlot');
    Plotly.purge('predsPlot');
  });

  $('btnStart').addEventListener('click', async ()=>{
    if(!state.model) return alert('Build model first');
    if(!state.ds) return alert('Generate dataset first');
    state.stopRequested = false;

    const batchSize = parseInt($('batchSize').value)||32;
    const epochs = parseInt($('epochs').value)||20;
    const valSplit = parseFloat($('valSplit').value)||0.1;

    // manual split so we can keep a fixed validation subset
    const total = state.ds.xs.shape[0];
    const valCount = Math.max(1, Math.floor(total * valSplit));
    const trainCount = total - valCount;

    // Extract raw arrays to split
    const xsArr = state.raw.xsArray; // [num][seqLen][1]
    const ysArr = state.raw.ysArray; // [num][1]

    const trainXsArr = xsArr.slice(0, trainCount);
    const trainYsArr = ysArr.slice(0, trainCount);
    const valXsArr = xsArr.slice(trainCount);
    const valYsArr = ysArr.slice(trainCount);

    state.valXsSubset = valXsArr.slice(0, Math.min(state.valSubsetCount, valXsArr.length));
    state.valYsSubset = valYsArr.slice(0, Math.min(state.valSubsetCount, valYsArr.length));

    // Convert to tensors
    // trainXsArr and valXsArr are number[][][] already ([num][seqLen][1])
    const trainXsTensor = tf.tensor3d(trainXsArr, [trainXsArr.length, trainXsArr[0].length, 1]);
    const trainYsTensor = tf.tensor2d(trainYsArr, [trainYsArr.length, 1]);

    const valXsTensor = tf.tensor3d(valXsArr, [valXsArr.length, valXsArr[0].length, 1]);
    const valYsTensor = tf.tensor2d(valYsArr, [valYsArr.length, 1]);

    $('trainLog').textContent = 'Training started.';

    state.metricsHistory = [];

    // Callback
    const onEpochEnd = async (epoch, logs) => {
      const e = epoch + 1;
      state.metricsHistory.push({loss: logs.loss, valLoss: logs.val_loss});
      updateTrainingMetricsChart(state.metricsHistory);
      if(state.model && state.valXsSubset && state.valYsSubset){
        await updatePredictionsVsTargetsChart(state.valXsSubset, state.valYsSubset, state.model);
      }

      // Update progress UI
      const totalEpochs = epochs;
      const pct = Math.round((e / totalEpochs) * 100);
      $('trainProgressFill').style.width = pct + '%';
      $('epochStatus').textContent = `Epoch: ${e} / ${totalEpochs}`;

      // Append to log (keep recent entries)
      const logEl = $('trainLog');
      const line = document.createElement('div');
      line.textContent = `Epoch ${e}: loss=${logs.loss.toFixed(6)} val=${(logs.val_loss||0).toFixed(6)}`;
      logEl.prepend(line);
      // trim log lines to 200
      while(logEl.childElementCount > 200) logEl.removeChild(logEl.lastChild);

      if(state.stopRequested){
        if(state.model) state.model.stopTraining = true;
      }
    };

    // Start training
    try{
      await state.model.fit(trainXsTensor, trainYsTensor, {
        batchSize, epochs,
        validationData: [valXsTensor, valYsTensor],
        callbacks: {onEpochEnd}
      });
      $('trainLog').textContent = 'Training finished.';
    }catch(err){
      console.error(err);
      $('trainLog').textContent = 'Training stopped or error: '+err.message;
    }

    // cleanup
    trainXsTensor.dispose(); trainYsTensor.dispose(); valXsTensor.dispose(); valYsTensor.dispose();
  });

  $('btnStop').addEventListener('click', ()=>{
    state.stopRequested = true;
    $('trainLog').textContent = 'Stop requested.';
  });

  // initial dataset generation
  $('btnGen').click();
}

// Initialize
window.addEventListener('load', ()=>{
  setupUI();
});
