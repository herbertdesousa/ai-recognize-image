import { createCanvas, loadImage } from 'canvas';
import * as fs from 'fs';

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

const sumArr = (arr: number[]): number => arr.reduce((acc, i) => acc + i, 0);

type CalcPropagationOpt = {
  weights: number[][];
  input: number[];
  bias: number[];
  outputNeuronsQuantity: number;
}
function calcPropagation(opt: CalcPropagationOpt): number[] {
  const { outputNeuronsQuantity, weights, input, bias } = opt;

  return Array(outputNeuronsQuantity).fill('').map((_, x) => {
    const allWeights = weights.map((weight, y) => weight[x] * input[y]);

    return sigmoid(sumArr(allWeights) + bias[x]);
  });
}

function createRandomList(x: number) {
  return Array(x).fill('').map(() => Math.random());
}

function createRandomMatrix(x: number, y: number) {
  return Array(x).fill('').map(() => Array(y).fill('').map(() => Math.random()));
}

function transposeMatrix(matrix: number[][]): number[][] {
  const rows = matrix.length;
  const cols = matrix[0].length;

  // Create a new matrix with swapped rows and columns
  const result: number[][] = [];
  for (let j = 0; j < cols; j++) {
    const newRow: number[] = [];
    for (let i = 0; i < rows; i++) {
      newRow.push(matrix[i][j]);
    }
    result.push(newRow);
  }

  return result;
}

function sumMatrix(matrix1: number[][], matrix2: number[][]): number[][] {
  const rows = matrix1.length;
  const cols = matrix1[0].length;

  // Create a new matrix to store the sum
  const result: number[][] = [];
  for (let i = 0; i < rows; i++) {
    const newRow: number[] = [];
    for (let j = 0; j < cols; j++) {
      newRow.push(matrix1[i][j] + matrix2[i][j]);
    }
    result.push(newRow);
  }

  return result;
}

function sumLists(list1: number[], list2: number[]): number[] {
  const sum: number[] = [];

  const maxLength = Math.max(list1.length, list2.length);

  for (let i = 0; i < maxLength; i++) {
    const num1 = list1[i] || 0; // If index is out of bounds, use 0
    const num2 = list2[i] || 0; // If index is out of bounds, use 0

    const sumOfElements = num1 + num2;
    sum.push(sumOfElements);
  }

  return sum;
}

function reduceMatrix(matrix: number[][]): number[] {
  const rows = matrix.length;
  const cols = matrix[0].length;

  // Initialize an array to store the sum of each column
  const columnSums: number[] = new Array(cols).fill(0);

  // Compute the sum of each column
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      columnSums[j] += matrix[i][j];
    }
  }

  return columnSums;
}

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

async function run() {
  const INPUT_NEURONS = 64;
  const HIDDEN_NEURONS = 8;
  const OUTPUT_NEURONS = 9;
  const EPOCHS = 5000;

  const TRAIN_FILE_PATH = 'nn-train.json';

  let {
    w1, // ROW === INPUT(X) || COL === HIDDEN_NEURON
    w2, // ROW === HIDDEN_NEURON || COL === OUTPUT_NEURON
    b1, // HIDDEN_NEURON/EACH
    b2, // OUTPUT_NEURON/EACH
  } = (() => {
    const findTrainFile = fs.existsSync(TRAIN_FILE_PATH);

    if (findTrainFile) {
      const file = JSON.parse(
        fs.readFileSync(TRAIN_FILE_PATH, { encoding: 'utf-8' }) || '{}'
      );

      return {
        w1: file.w1 as number[][],
        w2: file.w2 as number[][],
        b1: file.b1 as number[],
        b2: file.b2 as number[],
      }
    }

    return {
      w1: createRandomMatrix(INPUT_NEURONS, HIDDEN_NEURONS),
      w2: createRandomMatrix(HIDDEN_NEURONS, OUTPUT_NEURONS),
      b1: createRandomList(HIDDEN_NEURONS),
      b2: createRandomList(OUTPUT_NEURONS),
    }
  })();

  function feedForward(input: number[], target: number[]) {
    // feed forward
    const a1 = calcPropagation({
      outputNeuronsQuantity: HIDDEN_NEURONS,
      bias: b1,
      input: input,
      weights: w1,
    });

    const a2 = calcPropagation({
      outputNeuronsQuantity: OUTPUT_NEURONS,
      bias: b2,
      input: a1,
      weights: w2,
    });

    // backpropagation
    const d2 = a2.map((_a2, idx) => (_a2 - target[idx]) * _a2 * (1 - _a2));

    const errW2 = a1.map(_a1 => d2.map(_d2 => _d2 * _a1));
    const errB2 = d2.map((_d2, idx) => _d2 * b2[idx]);

    const d1 = (() => {
      const rawErrW1 = errW2.map((_w2, x) => {
        return _w2.map((_w2i, y) => d2[y] * _w2i * a1[x] * (1 - a1[x]))
      });

      return reduceMatrix(transposeMatrix(rawErrW1));
    })();

    const errW1 = input.map(_x => d1.map(_d1 => _d1 * _x));
    const errB1 = d1.map((_d1, idx) => _d1 * b1[idx]);

    return {
      output: a2,
      errors: {
        w1: errW1,
        w2: errW2,
        b1: errB1,
        b2: errB2,
      },
    }
  }

  // console.log(feedForward(await readPixelsFromImage('imgs/1.png'), []).output);

  // return;

  const imagePaths = [
    'imgs/1.png',
    'imgs/2.png',
  ];

  for (let epoch = 0; epoch <= EPOCHS; epoch++) {
    // feed forward
    let errW1: number[][] = [];
    let errW2: number[][] = [];
    let errB1: number[] = [];
    let errB2: number[] = [];

    for (let imagePathIdx = 0; imagePathIdx < imagePaths.length; imagePathIdx++) {
      const img = await readPixelsFromImage(imagePaths[imagePathIdx]);


      const target = img.slice(0, 9);
      const image = img.slice(16);

      console.log(target, image.length)

      const train = feedForward(image, target);

      if (errW1.length === 0) {
        errW1 = train.errors.w1;
      } else {
        errW1 = sumMatrix(errW1, train.errors.w1);
      }

      if (errW2.length === 0) {
        errW2 = train.errors.w2;
      } else {
        errW2 = sumMatrix(errW2, train.errors.w2);
      }

      errB1 = sumLists(errB1, train.errors.b1);
      errB2 = sumLists(errB2, train.errors.b2);
    }

    // const train1 = feedForward(
    //   await readPixelsFromImage('imgs/1.png'),
    //   [1, 0, 0, 0, 0, 0, 0, 0, 0],
    // );
    // const train2 = feedForward(
    //   await readPixelsFromImage('imgs/2.png'),
    //   [0, 1, 0, 0, 0, 0, 0, 0, 0],
    // );

    // const errW1 = sumMatrix(train1.errors.w1, train2.errors.w1);
    // const errW2 = sumMatrix(train1.errors.w2, train2.errors.w2);
    // const errB1 = sumLists(train1.errors.b1, train2.errors.b1);
    // const errB2 = sumLists(train1.errors.b2, train2.errors.b2);

    // backpropagation
    w2 = w2.map((_w2, x) => _w2.map((_w2i, y) => _w2i - errW2[x][y]));
    b2 = b2.map((_b2, idx) => _b2 - errB2[idx]);

    w1 = w1.map((_w1, x) => _w1.map((_w1i, y) => _w1i - errW1[x][y]));
    b1 = b1.map((_b1, idx) => _b1 - errB1[idx]);

    if (epoch === EPOCHS)
      fs.writeFileSync(TRAIN_FILE_PATH, JSON.stringify({ w1, b1, w2, b2 }));
  }
}

run();
