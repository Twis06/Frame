import torch
import json
import time

model_path = "/home/nv/px4_policy_deploy/src/px4ctrl/scripts/models/actor34200000.pt"
data_path = "/home/nv/px4_policy_deploy/src/px4ctrl/scripts/sa_pair.json"

model = torch.jit.load(model_path)
hidden_state = torch.zeros(1, 1, 256).to('cuda')

with open(data_path, 'r') as f:
    data = json.load(f)

def warmup(model, pixel_sample, state_sample, duration=5):
    print("Starting warmup...")
    start_time = time.perf_counter()
    
    count = 0
    while time.perf_counter() - start_time < duration:
        hidden_state = torch.zeros((1, 1, 256)).to('cuda')
        output, hidden_state = model(pixel_sample, state_sample, hidden_state.detach())
        output = torch.clamp(output, -1.0, 1.0).detach().cpu()
        count += 1
    
    end_time = time.perf_counter()
    print(f"Warmup completed! Performed {count} inferences in {end_time-start_time:.2f} seconds")

first_sample = next(iter(data))
pixel_sample = torch.tensor(first_sample['obs']['img'], dtype=torch.float32).to('cuda')
state_sample = torch.tensor(first_sample['obs']['state'], dtype=torch.float32).to('cuda')

# Perform warmup
warmup(model, pixel_sample, state_sample)

# Statistics
max_error = 0
for step_index, step in enumerate(data):
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
    pixel = torch.tensor(step['obs']['img'], dtype=torch.float32)
    state = torch.tensor(step['obs']['state'], dtype=torch.float32)
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
    action, hidden_state = model(pixel, state, hidden_state.detach())
    # 
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    timing_stats['inference'].append(t2 - t1)

    # 4. CPU transfer
    t1 = time.perf_counter()
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
