import torch
import json
import time
import onnx
import onnxruntime as ort

# 设置模型和数据路径
model_path = "/home/nv/px4_policy_deploy/src/px4ctrl/scripts/models/actor1900000.pt"
onnx_model_path = "/home/nv/px4_policy_deploy/src/px4ctrl/scripts/models/actor1900000.onnx"
data_path = "/home/nv/px4_policy_deploy/src/px4ctrl/scripts/sa_pair.json"

# **1. 导出为 ONNX 模型**
def export_onnx(model, pixel_sample, state_sample, hidden_state, onnx_model_path):
    print("Exporting the PyTorch model to ONNX format...")
    torch.onnx.export(
        model,
        (pixel_sample, state_sample, hidden_state),  # 示例输入
        onnx_model_path,
        export_params=True,  # 保存训练好的参数
        opset_version=11,    # ONNX opset 版本
        do_constant_folding=True,  # 是否执行常量折叠优化
        input_names=['pixel', 'state', 'hidden_state'],  # 输入名称
        output_names=['action', 'hidden_state_new'],         # 输出名称
        dynamic_axes=None,
    )
    print(f"Model exported to {onnx_model_path}")

# **2. 使用 ONNX Runtime 推理**
def inference_with_onnx(onnx_model_path, data):
    print("Running inference with ONNX Runtime...")
    ort_session = ort.InferenceSession(onnx_model_path)

    # 初始化统计信息
    timing_stats = {
        'data_loading': [],
        'inference': [],
        'cpu_transfer': [],
    }

    last_time = time.perf_counter()
    hidden_state = torch.zeros((1, 1, 64)).numpy()

    for step_index, step in enumerate(data):
        # 控制推理速率 (每 15ms 进行一次推理)
        while not time.perf_counter() - last_time > 0.015:
            pass
        last_time = time.perf_counter()

        # 1. 数据加载
        t1 = time.perf_counter()
        pixel = torch.tensor(step['obs']['img'], dtype=torch.float32).numpy()  # 添加批次维度
        state = torch.tensor(step['obs']['state'], dtype=torch.float32).numpy()
        t2 = time.perf_counter()
        timing_stats['data_loading'].append(t2 - t1)

        # 2. 推理
        t1 = time.perf_counter()
        inputs = {
            'pixel': pixel,
            'state': state,
            'hidden_state': hidden_state
        }
        outputs = ort_session.run(None, inputs)
        action = outputs[0]  # 获取推理的动作输出
        hidden_state = outputs[1]  # 更新隐藏状态
        t2 = time.perf_counter()
        timing_stats['inference'].append(t2 - t1)

        # 3. CPU 操作
        t1 = time.perf_counter()
        action = torch.tensor(action).clamp(-1.0, 1.0).detach().cpu()
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



def main():
    # 加载 PyTorch 模型
    model = torch.jit.load(model_path)

    # 加载测试数据
    with open(data_path, 'r') as f:
        data = json.load(f)

    # 获取第一个样本
    # first_sample = next(iter(data))
    # pixel_sample = torch.tensor(first_sample['obs']['img'], dtype=torch.float32).to('cuda')  # 添加批次维度
    # state_sample = torch.tensor(first_sample['obs']['state'], dtype=torch.float32).to('cuda')  # 添加批次维度
    # hidden_state = torch.zeros((1, 1, 256)).to('cuda')

    # # If engine doesn't exist, build it
    # import os
    # if not os.path.exists(onnx_model_path):
    #     # 导出模型
    #     export_onnx(model, pixel_sample, state_sample, hidden_state, onnx_model_path)

    # 执行推理
    inference_with_onnx(onnx_model_path, data)

if __name__ == '__main__':
    main()
