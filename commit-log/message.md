## Pull Request Title
Added GPU implementation for the CNN

## Summary
This pull request implements the CNN on a NVIDIA GPU using CUDA to speed up the `compute()` function relative to the CPU implementation.

## Tech Details

### init()
Takes in a convolution matrix and a vector of weights for each neuron. The GPU device is initialized and kernels are loaded. GPU memory is then allocated for the inputs and for any buffers we will need. The buffers will be used for the intermediate results we get between each kernel. All pointers to the GPU memory are stored in the `CudaContext` struct along with other metadata.

### compute()
Takes in the input image to the CNN and copies it to GPU memory. Image is then processed through each of the three kernel stages:

- cnn_stage1: Takes in input image, convolution matricies, and weight vectors. Each 5x5 cell in the image will be a work-item for a total of 400 work-items. Each neuron needs to process the image so we will actually have 10*400 work-items. Every work-item will be assigned a specific neuron and cell. The cell will be dot producted with the neuron's specific convolution matrix (convolution layer). The output of the dot product will be checked if below 0. If it is, set to 0 (ReLU layer). Finally the output of the ReLU layer will be multiplied by an element in the neuron's weight vector (output layer). Once all work-items are done, the final output is 10 neurons, each of which have 4000 weighted values that need to be summed to complete the dot product.
    
- cnn_stage2: Takes in the output of stage 1. This stage will continue the output layer by using the divide-and-conquer strategy to parallelize the summation of every neuron's values. To do this, we will have 2000 work-items, each of which will sum 20-values. Each work-item will be assigned a neuron and a specific segment of the nueron's values to sum. The result after completing all summations will be 10 neurons, each with 200 values.

- cnn_stage3: Takes in the output of stage 2. This stage is a repeat of stage two and will sum segments of each neuron's values in parallel. We will have 100 work-items, each of which will sum 20 values. The result after completing all summations will be 10 neurons, each with 10 values.

We will wait for all kernels to finish and copy the result from stage3 to the host memory. The final stage of the neuron value summation is done on the host as the work is too small to send to the GPU (too much overhead). Once each neuron is summed, we will return a vector of 10 values, 1 value per neuron.

## Testing for Correctness
Correctness was tested by running on eceubuntu servers and making sure output of CPU and GPU implementation were the same. Outputs were compared with `compare.py` script. This is sufficient since the CPU implemenation is known to be correct. Multiple runs were conducted to reduce chance of missing undeterministic behaviour.

## Testing for Performance
Performance was tested by using the `std::time` library provided in `main.rs`. The server used during testing was ecetesla1. For the CPU implementation 25873 micro seconds of work was done compared to 8560 micro seconds on the GPU implemenation. This shows that the GPU was able to effectively parallelize the workload resulting in a roughly 3x speedup.

I also experimented with the block size for the GPU kernels. I tested with every multiple of 32 up to 1024 to see what was fastest. The runtime didn't seem to change that much but the lower block sizes (32/64) where marginally faster.