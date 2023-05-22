import { createCanvas, loadImage } from 'canvas';
import * as fs from 'fs';

import { neuron } from './neuron';

async function readPixelsFromImage(imagePath: string): Promise<number[]> {
  const image = await loadImage(imagePath);

  const canvas = createCanvas(image.width, image.height);
  const context = canvas.getContext('2d');

  context.drawImage(image, 0, 0);

  const imageData = context.getImageData(0, 0, image.width, image.height);
  const { data, width, height } = imageData;

  const pixels: number[] = [];

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      // const red = data[((width * y) + x) * 4];
      // const green = data[((width * y) + x) * 4 + 1];
      // const blue = data[((width * y) + x) * 4 + 2];
      const alpha = data[((width * y) + x) * 4 + 3];

      pixels.push(alpha === 255 ? 1 : 0);
    }
  }

  return pixels;
}

function setHighestToOne(arr: number[]): number[] {
  const max = Math.max(...arr);
  return arr.map(num => num === max ? 1 : 0);
}

async function run() {
  function NN(_inputs: number[], target: number[], epochs: number): number[] {
    const state: {
      weights: {
        inputs: number[][],
        hiddenI: number[][],
        hiddenII: number[][],
        output: number[][],
      };
      inputs: {
        inputs: number[][],
        hiddenI: number[][],
        hiddenII: number[][],
        output: number[][],
      };
    } = {
      weights: { inputs: [], hiddenI: [], hiddenII: [], output: [] },
      inputs: { inputs: [], hiddenI: [], hiddenII: [], output: [] },
    };

    // inputs
    const inputs = Array(_inputs.length).fill('').map((_, index) => {
      const weights = [Math.random()]; 
      state.weights.inputs.push(weights);
      state.inputs.inputs.push([_inputs[index]]);

      return neuron([_inputs[index]], weights);
    });

    // hidden I
    const hiddensI = Array(16).fill('').map(() => {
      const weights = Array(inputs.length).fill('').map(() => Math.random());

      state.weights.hiddenI.push(weights);
      state.inputs.hiddenI.push(inputs);

      return neuron(inputs, weights);
    });

    // hidden II
    const hiddensII = Array(16).fill('').map(() => {
      const weights = Array(hiddensI.length).fill('').map(() => Math.random());
      
      state.weights.hiddenII.push(weights);
      state.inputs.hiddenII.push(hiddensI);

      return neuron(hiddensI, weights);
    });

    // output
    const outputs = Array(9).fill('').map(() => {
      const weights = Array(hiddensII.length).fill('').map(() => Math.random());
      
      state.weights.output.push(weights);
      state.inputs.output.push(hiddensII);
      
      return neuron(hiddensII, weights);
    });

    
    outputs.map((output, index) => {
      const W3 = hiddensII.map(hiddenII => {
        return (output - target[index]) * output * (1 - output) * hiddenII;
      });

      console.log(W3)
    });
    
    console.log(setHighestToOne(outputs))


    /* outputs.map((output, index) => {
      const W3 = outputOfHiddenII.map(outputHiddenII => {
        return (output - target[index]) * output * (1 - output) * outputHiddenII;
      });

      console.log(W3);
    }) */

    // for (let epoch = 0; epoch < epochs; epoch++) {
    // }

    fs.writeFileSync('train.json', JSON.stringify(state));

    
    
    // return output;
    return [];
  }

  const pixelsArray = await readPixelsFromImage('imgs/1.png');
  
  const TARGET = [1, 0, 0, 0, 0, 0, 0, 0, 0];

  const EPOCHS = 100;

  const result = setHighestToOne(NN(pixelsArray, TARGET, EPOCHS));

  console.log(result.findIndex(i => i === 1) + 1);
}

// run();

/*
  a21 = 0.5 -> 1
  a11 = 
*/

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

const sumArr = (arr: number[]): number => arr.reduce((acc, i) => acc + i, 0);

function sumNestedArrayItems(arr: number[][]): number[] {
  const result: number[] = [];

  for (let i = 0; i < arr[0].length; i++) {
    let sum = 0;

    for (let j = 0; j < arr.length; j++) {
      sum += arr[j][i];
    }

    result.push(sum);
  }

  return result;
}

// backpropagation 1 layer manual
function v1() {
  let {
    a11,
    a12,
    b21,
    b22,
    w5,
    w6,
  } = {
    a11: 0.8461403894202115,
    a12: 0.5973245545943462,
    // a21: 0.767572100985218
    // a21: 0.767572100985218
    
    b21: 0.466101764690122,
    b22: 0.09832927369697009,
    w5: 0.7890118830786776,
    w6: 0.8823302616194728,
  };
  
  for (let epoch = 0; epoch < 2000; epoch++) {
    const a21 = sigmoid(w5 * a11 + w6 * a12 + b21);
    const y11 = 1;
  
    const a22 = sigmoid(w5 * a11 + w6 * a12 + b22);
    const y12 = 0;
    

    const calcErrW = (target: number, output: number, input: number) =>
    //       MLS                 SIGMOID'           I
      (output - target) * output * (1 - output) * input;
  
    const errA21W5 = calcErrW(y11, a21, a11);
    const errA21W6 = calcErrW(y11, a21, a12);
    const errB21 = calcErrW(y11, a21, b21);

    const errA22W5 = calcErrW(y12, a22, a11);
    const errA22W6 = calcErrW(y12, a22, a12);
    const errB22 = calcErrW(y12, a22, b22);
  
    w5 -= errA21W5 + errA22W5;
    w6 -= errA21W6 + errA22W6;
    b21 -= errB21;
    b22 -= errB22;
    
    console.log(a21, a22)
    return;
  }
}

// backpropagation 1 layer automatico
function v2() {
  let {
    a1,
    b2,
    w2,
  } = {
    a1: [
      0.8461403894202115, // a11
      0.5973245545943462, // a12
    ],
    w2: [
      0.7890118830786776, // w5
      0.8823302616194728, // w6
    ],
    
    // 1 BIAS POR OUTPUT_NEURON
    b2: [
      0.466101764690122,  // b21
      0.09832927369697009,// b22
      0.17237488295630943,
    ],
  };

  const OUTPUT_NEURONS = 3;
  const TARGET = [0, 0, 1];

  for (let epoch = 0; epoch < 1000; epoch++) {
    const a2 = Array(OUTPUT_NEURONS).fill('').map((_, idx) => {
      const weightsByA1 = a1.reduce((total, _a1, idxA1) => {
        return total + (w2[idxA1] * _a1);
      }, 0);
      
      return sigmoid(weightsByA1 + b2[idx]);
    });

    const calcErr = (target: number, output: number, input: number) =>
    //       MLS                 SIGMOID'           I
      (output - target) * output * (1 - output) * input;

      
    // const errA21W5 = calcErrW(y11, a21, a11);
    const errW2 = sumNestedArrayItems(
      a2.map((_a2, idx) => {
        return a1.map(_a1 => calcErr(TARGET[idx], _a2, _a1));
      }),
    );
    
    // const errB21 = calcErrW(y11, a21, b21);
    const errB2 = a2.map((_a2, idx) => calcErr(TARGET[idx], _a2, b2[idx]));

    w2 = w2.map((_w1, idx) => _w1 - errW2[idx]);
    b2 = b2.map((_b2, idx) => _b2 - errB2[idx]);

    console.log(a2);
  }
}

// backpropagation 1 layer delta
function v3() {
  let {
    a1,
    b2,
    w2,
  } = {
    a1: [
      0.8461403894202115, // a11
      0.5973245545943462, // a12
    ],
    w2: [
      0.7890118830786776, // w5
      0.8823302616194728, // w6
    ],
    
    // 1 BIAS POR OUTPUT_NEURON
    b2: [
      0.466101764690122,  // b21
      0.09832927369697009,// b22
      0.17237488295630943,
    ],
  };

  const OUTPUT_NEURONS = 3;
  const TARGET = [0, 0, 1];

  for (let epoch = 0; epoch < 1000; epoch++) {
    // feed forward
    const a2 = Array(OUTPUT_NEURONS).fill('').map((_, idx) => {
      const weightsByA1 = a1.reduce((total, _a1, idxA1) => {
        return total + (w2[idxA1] * _a1);
      }, 0);
      
      return sigmoid(weightsByA1 + b2[idx]);
    });

    // backpropagation
    const d2 = a2.map((_a2, idx) => (_a2 - TARGET[idx]) * _a2 * (1 - _a2));

    const errW2 = sumNestedArrayItems(d2.map(_d2 => a1.map(_a1 => _d2 * _a1)));
    const errB2 = d2.map((_d2, idx) => _d2 * b2[idx]);

    w2 = w2.map((_w1, idx) => _w1 - errW2[idx]);
    b2 = b2.map((_b2, idx) => _b2 - errB2[idx]);

    console.log(a2);
  }
}

// backpropagation 2 layer delta
function v4() {
  let {
    x,
    w1,
    b1,
    b2,
    w2,
  } = {
    x: [
      0.7762731534822029,
      0.5905773556464473,
    ],
    w1: [
      [0.1865583054234718, 0.1917077944995373],
      [0.9968795636140959, 0.7416680399277247],
    ],
    b1: [
      0.0182253190251917,
      0.4148609916812549,
    ],
    w2: [
      0.7890118830786776,
      0.8823302616194728,
      // [0.7890118830786776],
      // [0.8823302616194728],
    ],
    
    // 1 BIAS POR OUTPUT_NEURON
    b2: [
      0.46610176469012245,
      0.09832927369697009,
      0.17237488295630943,
    ],
  };

  const HIDDEN_NEURONS = 2;
  const OUTPUT_NEURONS = 3;
  const TARGET = [0, 0, 1];

  for (let epoch = 0; epoch < 1000; epoch++) {
    // feed forward
    const a1 = Array(HIDDEN_NEURONS).fill('').map((_, idx) => {
      return sigmoid(sumArr(w1[idx].map((_w11, idx) => x[idx] * _w11)));
    });

    const a2 = Array(OUTPUT_NEURONS).fill('').map((_, idx) => {
      const weightsByA1 = a1.reduce((total, _a1, idxA1) => {
        return total + (w2[idxA1] * _a1);
      }, 0);
      
      return sigmoid(weightsByA1 + b2[idx]);
    });

    // backpropagation
    const d2 = a2.map((_a2, idx) => (_a2 - TARGET[idx]) * _a2 * (1 - _a2));

    const errW2 = sumNestedArrayItems(d2.map(_d2 => a1.map(_a1 => _d2 * _a1)));
    const errB2 = d2.map((_d2, idx) => _d2 * b2[idx]);

    w2 = w2.map((_w1, idx) => _w1 - errW2[idx]);
    b2 = b2.map((_b2, idx) => _b2 - errB2[idx]);

    console.log(a2);
  }
}

v4()