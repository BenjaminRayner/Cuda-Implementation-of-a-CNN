// Very minimal skeleton for the kernel

#include <stdio.h>

// Get thread's work item (cell) and dot product with block (neuron) convolution matrix
extern "C" __global__ void cnn_stage1(double input_data[100][100], double filters[10][5][5], double weights[10][4000], double output_data[10][4000])
{
    // 3D (neuron, vcell, hcell)
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int neuron = idx / 400;         // 10 neurons
    int hcell = idx % 20;           // 20 cells per column
    int vcell = (idx / 20) % 20;    // 20 cells per row

    if (idx >= 4000) return;

    // Convolution layer
    // For thread's specific cell, iterate over cell's pixels & calculate dot product with neuron's convolution matrix
    double cell_output = 0;
    for (int vpixel = 0; vpixel < 5; ++vpixel) {
        for (int hpixel = 0; hpixel < 5; ++hpixel) {
            cell_output += input_data[vcell*5 + vpixel][hcell*5 + hpixel] * filters[neuron][vpixel][hpixel];
        }
    }

    // ReLU layer
    if (cell_output < 0) {
        cell_output = 0;
    }

    // Output layer (stage 1)
    // Do part of the weighted dot product for each neuron since we're here
    for (neuron = 0; neuron < 10; ++neuron) {
        output_data[neuron][idx] = cell_output * weights[neuron][idx];
    }
}

// Divide & Conquer for summing each neuron's dot product (Output layer stage 2)
extern "C" __global__ void cnn_stage2(double input_data[10][4000], double output_data[10][200])
{
    // 2D (neuron, segment)
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int neuron = idx / 200;        // 10 neurons
    int segment = idx % 200;       // 200 segments

    if (idx >= 2000) return;

    // Sum each 20 cell segment
    for (int i = 0; i < 20; ++i) {
        output_data[neuron][segment] += input_data[neuron][segment*20 + i];
    }
}
// (Output layer stage 3)
extern "C" __global__ void cnn_stage3(double input_data[10][200], double output_data[10][10])
{
    // 2D (neuron, segment)
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int neuron = idx / 10;       // 10 neurons
    int segment = idx % 10;      // 10 segments

    if (idx >= 100) return;

    // Sum each 20 cell segment
    for (int i = 0; i < 20; ++i) {
        output_data[neuron][segment] += input_data[neuron][segment*20 + i];
    }
}