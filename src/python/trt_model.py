import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import os

class TRTPointPredictor:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
            
        self.context = self.engine.create_execution_context()
        self.input_name = "input"
        self.output_name = "output"
        self.stream = cuda.Stream()
        
        # Buffer cache
        self.d_input = None
        self.d_output = None
        self.max_batch_size = 0
        
        # Meta info
        shape = self.engine.get_tensor_shape(self.input_name)
        self.input_dim = shape[-1]
        self.context_size = self.input_dim // 3

    def __call__(self, input_tensor):
        """
        Support both torch.Tensor (on CPU) and np.ndarray.
        Returns torch.Tensor to match PyTorch model interface.
        """
        import torch
        if isinstance(input_tensor, torch.Tensor):
            input_data = input_tensor.detach().cpu().numpy()
            is_torch = True
        else:
            input_data = input_tensor
            is_torch = False
            
        output_data = self.predict(input_data)
        
        if is_torch:
            return torch.from_numpy(output_data)
        return output_data

    def predict(self, input_data):
        batch_size = input_data.shape[0]
        input_dim = input_data.shape[1]
        
        self.context.set_input_shape(self.input_name, (batch_size, input_dim))
        
        h_input = np.ascontiguousarray(input_data.astype(np.float32))
        
        # Reallocate if batch size grows
        if batch_size > self.max_batch_size:
            if self.d_input: self.d_input.free()
            if self.d_output: self.d_output.free()
            self.d_input = cuda.mem_alloc(h_input.nbytes)
            self.d_output = cuda.mem_alloc(batch_size * 3 * 4)
            self.max_batch_size = batch_size
        
        self.context.set_tensor_address(self.input_name, int(self.d_input))
        self.context.set_tensor_address(self.output_name, int(self.d_output))
        
        cuda.memcpy_htod_async(self.d_input, h_input, self.stream)
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        
        h_output = np.empty((batch_size, 3), dtype=np.float32)
        cuda.memcpy_dtoh_async(h_output, self.d_output, self.stream)
        self.stream.synchronize()
        
        return h_output

    def __del__(self):
        if hasattr(self, 'd_input') and self.d_input: self.d_input.free()
        if hasattr(self, 'd_output') and self.d_output: self.d_output.free()

if __name__ == "__main__":
    engine_path = "models/onnx/mlp_c3_h256.engine"
    if not os.path.exists(engine_path):
        print(f"Engine not found at {engine_path}")
    else:
        predictor = TRTPointPredictor(engine_path)
        test_input = np.random.randn(10, 9).astype(np.float32)
        output = predictor.predict(test_input)
        print("Test Inference Successful!")
        print("Output shape:", output.shape)
        print("Output sample:", output[0])
