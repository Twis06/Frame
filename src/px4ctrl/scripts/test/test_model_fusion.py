import torch
import json
import time
from test_dataset import *
import numpy as np

model_path = "/home/nv/px4_policy_deploy/src/px4ctrl/scripts/models/seq_fusion.pt"
data_path = "/home/nv/px4_policy_deploy/custom_dataset.npz"

model = torch.jit.load(model_path).to("cuda")

def warmup(model, pixel_sample, state_sample, duration=5):
    print("Starting warmup...")
    start_time = time.perf_counter()
    
    count = 0
    while time.perf_counter() - start_time < duration:
        output = model(pixel_sample, state_sample)
        output = torch.clamp(output, -1.0, 1.0).detach().cpu()
        count += 1
    
    end_time = time.perf_counter()
    print(f"Warmup completed! Performed {count} inferences in {end_time-start_time:.2f} seconds")

data = load_and_test_dataset(data_path, batch_size=1)
first_sample = next(iter(data))
pixel_sample = torch.tensor(first_sample["img"], dtype=torch.float32).to("cuda")
state_sample = torch.tensor(first_sample["state"], dtype=torch.float32).to("cuda")
print(pixel_sample.shape)
print(state_sample)
 
# Perform warmup 
warmup(model, pixel_sample, state_sample)

# Statistics
max_error = 0
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
    pixel = torch.tensor(step['img'], dtype=torch.float32)
    state = torch.tensor(step['state'], dtype=torch.float32)
    t2 = time.perf_counter()
    timing_stats['data_loading'].append(t2 - t1)

    # 2. GPU transfer
    t1 = time.perf_counter()
    pixel = pixel.to('cuda')
    state = state.to('cuda')
    # 
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    timing_stats['gpu_transfer'].append(t2 - t1)

    # 3. inference 
    t1 = time.perf_counter()
    action = model(pixel, state)
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    timing_stats['inference'].append(t2 - t1)

    # 4. CPU transfer
    t1 = time.perf_counter()
    print(action)
    action = torch.clamp(action, -1.0, 1.0).detach().cpu()
    t2 = time.perf_counter()
    timing_stats['cpu_transfer'].append(t2 - t1)
    if (t2 - t1) > 0.0001:
        print(f"Step {step_index}: CPU transfer too long time = {1000 * (t2 - t1)}ms")

    # 5. Error computation
    t1 = time.perf_counter()
    action_label = torch.tensor(step['action'], dtype=torch.float32)
    error = torch.norm(action - action_label)
    print(f"Step {step_index}: Error = {error}")

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
    print(f"  Arg: {stats['mean']*1000:8.3f} ms")
    print(f"  Min: {stats['min']*1000:8.3f} ms")
    print(f"  Max: {stats['max']*1000:8.3f} ms")
    print(f"  Total: {stats['total']*1000:8.3f} ms")
