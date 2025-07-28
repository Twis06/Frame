import torch
import json
import time
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # Automatically initialize PyCUDA

# 设置模型和数据路径
onnx_model_path = "/home/nv/px4_policy_deploy/src/px4ctrl/scripts/models/actor1900000.onnx"
engine_path = "/home/nv/px4_policy_deploy/src/px4ctrl/scripts/models/actor1900000.engine"
data_path = "/home/nv/px4_policy_deploy/src/px4ctrl/scripts/sa_pair.json"

# 加载测试数据
with open(data_path, 'r') as f:
    data = json.load(f)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# **1. Convert ONNX Model to TensorRT Engine**
def build_engine(onnx_model_path, engine_path):
    # Create the builder, network, and config
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))  # EXPLICIT_BATCH is required for ONNX models
    config = builder.create_builder_config()  # Use the builder config for settings like workspace size
    config.max_workspace_size = 1 << 30  # 1GB workspace size

    # Parse the ONNX model
    parser = trt.OnnxParser(network, TRT_LOGGER)
    print(f"Loading ONNX file from path {onnx_model_path}...")
    with open(onnx_model_path, "rb") as model:
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

    # Serialize the engine to a file
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
    print(f"TensorRT engine saved to {engine_path}")


# **2. Load TensorRT Engine**
def load_engine(engine_path):
    print("Loading TensorRT engine...")
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


# **3. Run Inference with TensorRT**
def inference_with_tensorrt(engine, data):
    print("Running inference with TensorRT...")

    # Allocate buffers
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append({'host': host_mem, 'device': device_mem})
        else:
            outputs.append({'host': host_mem, 'device': device_mem})

    # Create execution context
    context = engine.create_execution_context()

    # Initialize hidden state
    hidden_state = torch.zeros((1, 1, 64), dtype=torch.float32)

    # Timing stats
    timing_stats = {
        'data_loading': [],
        # 'data_copy': [],
        # 'gpu_transfer': [],
        # 'gpu_transfer-pixel': [],
        'inference': [],
        'cpu_transfer': [],
    }

    last_time = time.perf_counter()

    slow_step_index = -1
    best_time = 1e-10

    for step_index, step in enumerate(data):
        # 控制推理速率 (每 15ms 进行一次推理)
        # while not time.perf_counter() - last_time > 0.015:
        #     pass
        # last_time = time.perf_counter()

        # 1. 数据加载
        stream.synchronize()
        t1 = time.perf_counter()
        pixel = torch.tensor(step['obs']['img'], dtype=torch.float32).cuda()
        state = torch.tensor(step['obs']['state'], dtype=torch.float32)
        print("pixel shape: ", pixel.shape, " state shape: ", state.shape)
        stream.synchronize()
        t2 = time.perf_counter()
        timing_stats['data_loading'].append(t2 - t1)

        # Copy inputs to device
        stream.synchronize()
        t1 = time.perf_counter()
        # np.copyto(inputs[0]['host'], pixel.ravel())
        np.copyto(inputs[1]['host'], state.ravel())
        np.copyto(inputs[2]['host'], hidden_state.ravel())
        # stream.synchronize()
        # t2 = time.perf_counter()
        # timing_stats['data_copy'].append(t2 - t1)

        # stream.synchronize()
        # t1 = time.perf_counter()
        cuda.memcpy_dtod_async(inputs[0]['device'], pixel.data_ptr(), pixel.element_size() * pixel.nelement(), stream)
        # cuda.memcpy_htod_async(inputs[0]['device'], inputs[0]["host"], stream)
        # stream.synchronize()
        # t2 = time.perf_counter()
        # timing_stats['gpu_transfer-pixel'].append(t2 - t1)

        # stream.synchronize()
        # t1 = time.perf_counter()
        # inputs[0]['device'] = pixel
        cuda.memcpy_htod_async(inputs[1]['device'], inputs[1]['host'], stream)
        cuda.memcpy_htod_async(inputs[2]['device'], inputs[2]['host'], stream)
        # inputs[2]['device'] = hidden_state
        # stream.synchronize()
        # t2 = time.perf_counter()
        # timing_stats['gpu_transfer'].append(t2 - t1)

        # 2. 推理
        # stream.synchronize()
        # t1 = time.perf_counter()
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

        # Copy outputs back to host
        cuda.memcpy_dtoh_async(outputs[0]['host'], outputs[0]['device'], stream)
        cuda.memcpy_dtoh_async(outputs[1]['host'], outputs[1]['device'], stream)

        hidden_state = outputs[0]['host']
        action = outputs[1]['host']

        stream.synchronize()
        t2 = time.perf_counter() 

        timing_stats['inference'].append(t2 - t1)

        if (t2 - t1) > best_time:
            best_time = t2 - t1
            slow_step_index = step_index

        # 3. CPU 操作
        stream.synchronize()
        t1 = time.perf_counter()
        action = torch.tensor(action).clamp(-1.0, 1.0).detach().cpu()
        stream.synchronize()
        t2 = time.perf_counter()
        timing_stats['cpu_transfer'].append(t2 - t1)

        # 打印误差信息
        action_label = torch.tensor(step['action'], dtype=torch.float32)
        error = torch.norm(action - action_label)
        print(f"Step {step_index}: Error = {error}")

    # 统计信息汇总
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
        print(f"  Mean: {stats['mean']*1000:.3f} ms")
        print(f"  Min: {stats['min']*1000:.3f} ms")
        print(f"  Max: {stats['max']*1000:.3f} ms")
        print(f"  Total: {stats['total']*1000:.3f} ms")
    print(best_time, slow_step_index)

def warm(engine, duration=5):
    print("Starting warmup...")
    start_time = time.perf_counter()
    i = 0
    while i < 10:
        inference_with_tensorrt(engine, data)
        print(f"Warmup {i} done!")
        i += 1

    end_time = time.perf_counter()
    print(f"Warmup completed! Performed inferences in {end_time-start_time:.2f} seconds")

# **Main Workflow**
def main():
    # If engine doesn't exist, build it
    import os
    if not os.path.exists(engine_path):
        build_engine(onnx_model_path, engine_path)

    # Load TensorRT engine
    engine = load_engine(engine_path)

    warm(engine)

    # Run inference
    inference_with_tensorrt(engine, data)

if __name__ == '__main__':
    main()