import torch
import json
import time
import numpy as np
import onnx
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from test_dataset import *

from onnxruntime.quantization import quantize_dynamic, QuantType

# Paths
onnx_path = "/home/nv/px4_policy_deploy/src/px4ctrl/scripts/models/seq_fusion.onnx"
model_path = "/home/nv/px4_policy_deploy/src/px4ctrl/scripts/models/seq_fusion.pt"
data_path = "/home/nv/px4_policy_deploy/custom_dataset.npz"


# Step 1: Convert PyTorch model to ONNX
# model = torch.jit.load(model_path).to("cuda")
# dummy_pixel_sample = torch.randn(1, *[8, 256, 256], dtype=torch.float32).to('cuda')
# dummy_state_sample = torch.randn(1, *[16], dtype=torch.float32).to('cuda')
# torch.onnx.export(
#     model,
#     (dummy_pixel_sample, dummy_state_sample),
#     onnx_path,
#     export_params=True,
#     opset_version=11,
#     input_names=['pixel', 'state'],
#     output_names=['output'],
#     dynamic_axes=None,
# )
# print("ONNX model exported to:", onnx_path)

# quantized_model_path = "/home/nv/px4_policy_deploy/src/px4ctrl/scripts/models/seq_fusion_int8.onnx"
# quantize_dynamic(
#     model_input=onnx_path,  # Input ONNX model
#     model_output=quantized_model_path,  # Output quantized ONNX model
#     weight_type=QuantType.QUInt8  # Quantize weights to int8
# )


# Step 2: Build TensorRT engine
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

import tensorrt as trt

def build_engine(onnx_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
    # Create the builder, network, and config
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))  # EXPLICIT_BATCH is required for ONNX models
    config = builder.create_builder_config()  # Use the builder config for settings like workspace size
    config.max_workspace_size = 1 << 30  # 1GB workspace size

    # Parse the ONNX model
    parser = trt.OnnxParser(network, TRT_LOGGER)
    print(f"Loading ONNX file from path {onnx_path}...")
    with open(onnx_path, "rb") as model:
        if not parser.parse(model.read()):
            print("Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    print("Successfully parsed ONNX file.")
    
    # Build the engine
    print("Building the TensorRT engine. This may take some time...")
    engine = builder.build_engine(network, config)
    if engine is None:
        print("Failed to build the engine.")
        return None
    print("Engine successfully created.")
    return engine

engine = build_engine(onnx_path)
if engine is None:
    raise RuntimeError("Failed to build TensorRT engine")

# Step 3: TensorRT runtime and inference setup
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append to appropriate list
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append((host_mem, device_mem))
        else:
            outputs.append((host_mem, device_mem))
    return inputs, outputs, bindings, stream

context = engine.create_execution_context()
inputs, outputs, bindings, stream = allocate_buffers(engine)

def infer(context, bindings, inputs, outputs, stream):
    # Transfer input data to device
    for inp in inputs:
        cuda.memcpy_htod_async(inp[1], inp[0], stream)
    # Run inference
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back to host
    for out in outputs:
        cuda.memcpy_dtoh_async(out[0], out[1], stream)
    # Synchronize the stream
    stream.synchronize()
    return [out[0] for out in outputs]

# Load data
data = load_and_test_dataset(data_path, batch_size=1)

# Step 4: Warmup
def warmup(context, bindings, inputs, outputs, stream, duration=5):
    print("Starting warmup...")
    start_time = time.perf_counter()
    count = 0
    while time.perf_counter() - start_time < duration:
        # Prepare dummy inputs for warmup
        pixel_sample = np.random.randn(*[1, 8, 256, 256]).astype(np.float32).ravel()
        state_sample = np.random.randn(*[1, 16]).astype(np.float32).ravel()
        np.copyto(inputs[0][0], pixel_sample)
        np.copyto(inputs[1][0], state_sample)
        
        # Run inference
        infer(context, bindings, inputs, outputs, stream)
        count += 1
    
    end_time = time.perf_counter()
    print(f"Warmup completed! Performed {count} inferences in {end_time - start_time:.2f} seconds")

warmup(context, bindings, inputs, outputs, stream)

# Step 5: Main inference loop with statistics
timing_stats = {
    'data_loading': [],
    'gpu_transfer': [],
    'inference': [],
    'cpu_transfer': [],
}

last_time = time.perf_counter()

for step_index, step in enumerate(data):
    while not time.perf_counter() - last_time > 0.015:
        pass
    last_time = time.perf_counter()

    # 1. Load data
    t1 = time.perf_counter()
    pixel = np.array(step['img'], dtype=np.float32).ravel()
    state = np.array(step['state'], dtype=np.float32).ravel()
    t2 = time.perf_counter()
    timing_stats['data_loading'].append(t2 - t1)

    # 2. GPU transfer
    t1 = time.perf_counter()
    np.copyto(inputs[0][0], pixel)
    np.copyto(inputs[1][0], state)
    t2 = time.perf_counter()
    timing_stats['gpu_transfer'].append(t2 - t1)

    # 3. Inference
    t1 = time.perf_counter()
    output = infer(context, bindings, inputs, outputs, stream)[0]
    t2 = time.perf_counter()
    timing_stats['inference'].append(t2 - t1)

    # 4. CPU transfer
    t1 = time.perf_counter()
    print(output)
    action = np.clip(output, -1.0, 1.0)
    t2 = time.perf_counter()
    timing_stats['cpu_transfer'].append(t2 - t1)
    if (t2 - t1) > 0.0001:
        print(f"Step {step_index}: CPU transfer too long time = {1000 * (t2 - t1)}ms")

    # 5. Error computation
    t1 = time.perf_counter()
    action_label = np.array(step['action'], dtype=np.float32)
    error = np.linalg.norm(action - action_label)
    print(f"Step {step_index}: Error = {error}")

# Step 6: Statistics summary
stats_summary = {}
for operation, times in timing_stats.items():
    stats_summary[operation] = {
        'mean': sum(times) / len(times),
        'min': min(times),
        'max': max(times),
        'total': sum(times)
    }

for operation, stats in stats_summary.items():
    print(f"\n{operation}:")
    print(f"  Mean: {stats['mean'] * 1000:.3f} ms")
    print(f"  Min: {stats['min'] * 1000:.3f} ms")
    print(f"  Max: {stats['max'] * 1000:.3f} ms")
    print(f"  Total: {stats['total'] * 1000:.3f} ms")