function sum(arr: number[]): number {
  return arr.reduce((a, b) => a + b, 0);
}

function relu(x: number) {
  return Math.max(0, x);
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

export function neuron(inputs: number[], weights: number[]): number {
  const multiply = inputs.map((input, index) => {
    return weights[index] * input;
  })
  
  const summed = sum(multiply);

  const output = sigmoid(summed);

  return output;
}

// export function NN() {
//   const weights = {
//     input: [[Math.random()], [Math.random()], [Math.random()]],
//     hidden: [
//       [Math.random(), Math.random(), Math.random()],
//       [Math.random(), Math.random(), Math.random()],
//       [Math.random(), Math.random(), Math.random()],
//     ],
//     output: [
//       [Math.random(), Math.random(), Math.random()],
//       [Math.random(), Math.random(), Math.random()],
//       [Math.random(), Math.random(), Math.random()],
//     ]
//   }
  
//   return {
//     weights,
//     exec: (inputs: number[]) => {
//       const neuronI1 = neuron([inputs[0]], weights.input[0]);
//       const neuronI2 = neuron([inputs[1]], weights.input[1]);
//       const neuronI3 = neuron([inputs[2]], weights.input[2]);

//       const neuronH1 = neuron([neuronI1, neuronI2, neuronI3], weights.hidden[0]);
//       const neuronH2 = neuron([neuronI1, neuronI2, neuronI3], weights.hidden[1]);
//       const neuronH3 = neuron([neuronI1, neuronI2, neuronI3], weights.hidden[2]);

//       const neuronO1 = neuron([neuronH1, neuronH2, neuronH3], weights.output[0]);
//       const neuronO2 = neuron([neuronH1, neuronH2, neuronH3], weights.output[1]);
//       const neuronO3 = neuron([neuronH1, neuronH2, neuronH3], weights.output[2]);

//       return setHighestToOne([neuronO1, neuronO2, neuronO3]);
//     }
//   }
// }
