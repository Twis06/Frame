#!/bin/bash
# ONNX到TensorRT转换示例脚本
# 使用优化后的onnx2trt.py工具

echo "🚀 ONNX到TensorRT转换示例"
echo "================================"

# 设置路径
ONNX_MODEL="./models/onnx/ultralight_segmentation_256x320.onnx"
CALIBRATION_DIR="./test/all_test/images"
TEST_IMAGE="./test/all_test/images/test_001.jpg"
OUTPUT_DIR="./models/trt"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo "📁 检查文件路径..."
if [ ! -f "$ONNX_MODEL" ]; then
    echo "⚠️  ONNX模型文件不存在: $ONNX_MODEL"
    echo "请修改脚本中的路径"
fi

if [ ! -d "$CALIBRATION_DIR" ]; then
    echo "⚠️  校准数据目录不存在: $CALIBRATION_DIR"
    echo "请修改脚本中的路径"
fi

echo ""
echo "1️⃣  FP16转换（推荐，性能和精度平衡）"
echo "================================"
python3 onnx2trt.py \
    --onnx "$ONNX_MODEL" \
    --output "$OUTPUT_DIR/model_fp16_optimized.trt" \
    --precision fp16 \
    --max-batch-size 4 \
    --workspace-size 4 \
    --test-image "$TEST_IMAGE" \
    --verbose

echo ""
echo "2️⃣  INT8转换（最高性能，需要校准数据）"
echo "================================"
python3 onnx2trt.py \
    --onnx "$ONNX_MODEL" \
    --output "$OUTPUT_DIR/model_int8_optimized.trt" \
    --precision int8 \
    --max-batch-size 4 \
    --workspace-size 4 \
    --calibration-dir "$CALIBRATION_DIR" \
    --calibration-samples 200 \
    --test-image "$TEST_IMAGE" \
    --verbose

echo ""
echo "3️⃣  FP32转换（最高精度，较慢）"
echo "================================"
python3 onnx2trt.py \
    --onnx "$ONNX_MODEL" \
    --output "$OUTPUT_DIR/model_fp32_optimized.trt" \
    --precision fp32 \
    --max-batch-size 2 \
    --workspace-size 4 \
    --test-image "$TEST_IMAGE" \
    --verbose

echo ""
echo "4️⃣  高性能INT8转换（大批次，更多校准样本）"
echo "================================"
python3 onnx2trt.py \
    --onnx "$ONNX_MODEL" \
    --output "$OUTPUT_DIR/model_int8_high_perf.trt" \
    --precision int8 \
    --max-batch-size 8 \
    --workspace-size 6 \
    --calibration-dir "$CALIBRATION_DIR" \
    --calibration-samples 500 \
    --test-image "$TEST_IMAGE" \
    --verbose

echo ""
echo "✅ 转换完成！"
echo "📊 生成的引擎文件:"
ls -lh "$OUTPUT_DIR"/*.trt 2>/dev/null || echo "未找到生成的引擎文件"

echo ""
echo "💡 使用建议:"
echo "   - FP16: 推荐用于大多数应用，性能和精度平衡"
echo "   - INT8: 最高性能，适合对精度要求不是极高的场景"
echo "   - FP32: 最高精度，适合精度要求极高的场景"
echo "   - 根据实际GPU内存调整batch_size和workspace_size"
echo "   - 更多校准样本通常能获得更好的INT8精度"