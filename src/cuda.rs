// This is the skeleton for the CUDA implementation

use crate::cnn::*;
use rustacuda::launch;
use rustacuda::memory::DeviceBox;
use rustacuda::prelude::*;
use std::error::Error;
use std::ffi::CString;

// Fields need to be ordered this way so the DeviceBoxes are
// dropped before the Context. Otherwise the drop will panic.

pub struct CudaContext {
    conv_layer: DeviceBox<ConvLayer>,
    output_layer: DeviceBox<OutputLayer>,
    input: DeviceBox<InputMatrix>,
    buffer1: DeviceBuffer<f64>,
    buffer2: DeviceBuffer<f64>,
    buffer3: DeviceBuffer<f64>,
    buffer4: [f64; 100],
    module: Module,
    stream: Stream,
    _context: Context
}

impl CudaContext {

    pub fn init(cnn: &Cnn) -> Result<Self, Box<dyn Error>> {
        rustacuda::init(CudaFlags::empty())?;
        let device = Device::get_device(0)?;
        let _context = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;
        let ptx = CString::new(include_str!("../kernel/kernel.ptx"))?;

        let conv_layer = DeviceBox::new(&cnn.conv_layer)?;
        let output_layer = DeviceBox::new(&cnn.output_layer)?;

        // Getting all device memory allocated now so we don't waste time in compute()
        let input = DeviceBox::new(&InputMatrix([[0.0; INPUT_DIM]; INPUT_DIM]))?;
        let buffer1 = DeviceBuffer::from_slice(&[0.0; 40000])?;
        let buffer2 = DeviceBuffer::from_slice(&[0.0; 2000])?;
        let buffer3 = DeviceBuffer::from_slice(&[0.0; 100])?;
        let buffer4 = [0.0; 100];

        let module = Module::load_from_string(&ptx)?;
        let stream = Stream::new(StreamFlags::DEFAULT, None)?;
        
        Ok(CudaContext {conv_layer, output_layer, module, stream, _context,
                        input, buffer1, buffer2, buffer3,buffer4})
    }

    pub fn compute(&mut self, input: &InputMatrix) -> Result<OutputVec, Box<dyn Error>> {
        let module = &self.module;
        let stream = &self.stream;
        self.input.copy_from(input)?;

        // lets say each cell is a work item. Therefore we have 20x20 work items * 10 neurons.
        // Outputs result of dot product of weight (Unsummed)
        unsafe {
            let result = launch!(module.cnn_stage1<<<4000/64 + 1, 64, 0, stream>>>(
                self.input.as_device_ptr(),
                self.conv_layer.as_device_ptr(),
                self.output_layer.as_device_ptr(),
                self.buffer1.as_device_ptr()
            ));
            result?;
        }

        // Need 2000 threads each of which will sum 20 work-items (Each neuron has 4000 items)
        // Outputs partial sum of dot product of weight
        unsafe {
            let result = launch!(module.cnn_stage2<<<2000/64 + 1, 64, 0, stream>>>(
                self.buffer1.as_device_ptr(),
                self.buffer2.as_device_ptr()
            ));
            result?;
        }

        // Need 100 threads each of which will sum 20 work-items (Each neuron has 200 items)
        // Outputs partial sum of dot product of weight
        unsafe {
            let result = launch!(module.cnn_stage3<<<100/64 + 1, 64, 0, stream>>>(
                self.buffer2.as_device_ptr(),
                self.buffer3.as_device_ptr()
            ));
            result?;
        }

        // Get data from GPU
        stream.synchronize()?;
        self.buffer3.copy_to(&mut self.buffer4)?;

        // Final stage of dot product sum. Too small to run on GPU
        let mut output = OutputVec([0.0; OUT_LAYER_SIZE]);
        for (i, val) in self.buffer4.iter().enumerate() {
            output.0[i/10] += *val;
        }

        Ok(output)
    }

    // Reset buffers for next round
    pub fn reset(&mut self) -> Result<(), Box<dyn Error>> {
        self.buffer2.copy_from(&[0.0; 2000])?;
        self.buffer3.copy_from(&[0.0; 100])?;
        Ok(())
    }
}
