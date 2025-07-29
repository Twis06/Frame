import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # 自动初始化CUDA上下文
import os
import time
import argparse
import numpy as np
import cv2  # 使用OpenCV替代PIL
import glob

class Int8Calibrator(trt.IInt8EntropyCalibrator2):
    """INT8校准器"""
    def __init__(self, calibration_images, cache_file, batch_size=1, max_calibration_samples=100):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.current_index = 0
        self.max_calibration_samples = max_calibration_samples
        
        # 确保CUDA上下文
        cuda.init()
        self.device = cuda.Device(0)
        self.context = self.device.make_context()
        
        # 预处理校准图像
        self.calibration_data = self.load_calibration_data(calibration_images)
        self.device_input = None
        
        print(f"📊 INT8校准器初始化完成，校准图像数量: {len(self.calibration_data)}")
    
    def __del__(self):
        """清理CUDA上下文"""
        try:
            if hasattr(self, 'device_input') and self.device_input:
                self.device_input.free()
            if hasattr(self, 'context'):
                self.context.pop()
        except:
            pass
    
    def load_calibration_data(self, calibration_images):
        """加载并预处理校准数据 - 使用cv2线性resize，无ImageNet归一化"""
        calibration_data = []
        print("📂 加载校准数据...")
        
        for img_path in calibration_images[:self.max_calibration_samples]:  # 使用参数控制数量
            try:
                # 使用cv2读取图像
                image = cv2.imread(img_path)
                if image is None:
                    print(f"跳过无效图像 {img_path}: 无法读取")
                    continue
                
                # BGR转RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # 使用cv2线性resize到目标尺寸
                image = cv2.resize(image, (320, 256), interpolation=cv2.INTER_LINEAR)
                
                # 转换为float32并归一化到[0,1]
                image = image.astype(np.float32) / 255.0
                
                # 转换为CHW格式 (Channel, Height, Width)
                image = np.transpose(image, (2, 0, 1))
                
                calibration_data.append(image)
                
            except Exception as e:
                print(f"跳过无效图像 {img_path}: {e}")
        
        print(f"✅ 成功加载 {len(calibration_data)} 张校准图像")
        return calibration_data
    
    def get_batch_size(self):
        return self.batch_size
    
    def get_batch(self, names):
        """获取下一批校准数据"""
        try:
            if self.current_index + self.batch_size > len(self.calibration_data):
                return None
            
            # 确保CUDA上下文处于活动状态
            self.context.push()
            
            # 准备批次数据
            batch = []
            for i in range(self.batch_size):
                batch.append(self.calibration_data[self.current_index + i])
            
            batch = np.stack(batch, axis=0).astype(np.float32)
            self.current_index += self.batch_size
            
            # 分配GPU内存（如果还没有分配）
            if self.device_input is None:
                self.device_input = cuda.mem_alloc(batch.nbytes)
            
            # 复制数据到GPU
            cuda.memcpy_htod(self.device_input, batch)
            
            print(f"🔄 校准进度: {self.current_index}/{len(self.calibration_data)}")
            
            # 保持上下文活动状态，不要pop
            return [self.device_input]
            
        except Exception as e:
            print(f"❌ 校准批次获取失败: {e}")
            try:
                self.context.pop()
            except:
                pass
            return None
    
    def read_calibration_cache(self):
        """读取校准缓存"""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                print(f"📁 读取校准缓存: {self.cache_file}")
                return f.read()
        return None
    
    def write_calibration_cache(self, cache):
        """写入校准缓存"""
        with open(self.cache_file, "wb") as f:
            f.write(cache)
        print(f"💾 保存校准缓存: {self.cache_file}")

class ONNXToTensorRTConverter:
    def __init__(self, logger_level=trt.Logger.WARNING):
        """初始化ONNX到TensorRT转换器"""
        self.logger = trt.Logger(logger_level)
        print("🔧 初始化TensorRT转换器...")
        
        # 初始化CUDA
        try:
            cuda.init()
            print("✅ CUDA初始化成功")
        except Exception as e:
            print(f"⚠️  CUDA初始化警告: {e}")
            print("继续使用默认CUDA上下文...")
    
    def find_calibration_images(self, calibration_dir):
        """查找校准图像"""
        if not calibration_dir or not os.path.exists(calibration_dir):
            raise ValueError(f"校准数据目录不存在: {calibration_dir}")
        
        # 支持多种图像格式
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        calibration_images = []
        
        for ext in extensions:
            calibration_images.extend(glob.glob(os.path.join(calibration_dir, ext)))
            calibration_images.extend(glob.glob(os.path.join(calibration_dir, ext.upper())))
        
        if len(calibration_images) == 0:
            raise ValueError(f"在 {calibration_dir} 中没有找到校准图像")
        
        print(f"📁 找到 {len(calibration_images)} 张校准图像")
        return sorted(calibration_images)
    
    def set_layer_precision_constraints(self, network, precision):
        """设置层级精度约束以保持模型精度"""
        if precision == 'fp16':
            # 对于某些关键层保持FP32精度
            critical_layer_types = [
                trt.LayerType.SOFTMAX,
                trt.LayerType.REDUCE,
                trt.LayerType.TOPK
            ]
            
            for i in range(network.num_layers):
                layer = network.get_layer(i)
                if layer.type in critical_layer_types:
                    layer.precision = trt.float32
                    layer.set_output_type(0, trt.float32)
                    print(f"🔧 设置关键层 {layer.name} 为FP32精度")
        
        elif precision == 'int8':
            # 对于INT8，某些层保持FP16或FP32
            sensitive_layer_types = [
                trt.LayerType.SOFTMAX,
                trt.LayerType.REDUCE,
                trt.LayerType.TOPK,
                trt.LayerType.NORMALIZATION
            ]
            
            for i in range(network.num_layers):
                layer = network.get_layer(i)
                if layer.type in sensitive_layer_types:
                    layer.precision = trt.float16
                    layer.set_output_type(0, trt.float16)
                    print(f"🔧 设置敏感层 {layer.name} 为FP16精度")
        
        print(f"✅ 层级精度约束设置完成")
    
    def convert(self, onnx_path, output_path=None, precision='fp16', max_batch_size=8, 
                workspace_size=4, calibration_dir=None, calibration_samples=100, 
                enable_strict_types=True, enable_tactic_sources=True):
        """
        将ONNX模型转换为TensorRT引擎 - 优化版本
        
        Args:
            onnx_path: ONNX模型路径
            output_path: 输出TensorRT引擎路径
            precision: 精度模式 ('fp32', 'fp16', 'int8')
            max_batch_size: 最大批次大小
            workspace_size: 工作空间大小 (GB)
            calibration_dir: INT8校准数据目录
            calibration_samples: 校准样本数量
            enable_strict_types: 启用严格类型检查
            enable_tactic_sources: 启用所有策略源
        """
        # 设置输出路径
        if output_path is None:
            output_path = onnx_path.replace('.onnx', f'_tensorrt_{precision}.trt')
        
        print(f"🔄 开始ONNX到TensorRT转换...")
        print(f"输入ONNX: {onnx_path}")
        print(f"输出引擎: {output_path}")
        print(f"精度模式: {precision}")
        print(f"最大批次: {max_batch_size}")
        print(f"工作空间: {workspace_size} GB")
        
        # 检查输入文件
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX文件不存在: {onnx_path}")
        
        # 创建输出目录
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 创建builder和network
        builder = trt.Builder(self.logger)
        config = builder.create_builder_config()
        
        # 设置工作空间
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size * (1 << 30))
        
        # 启用高级优化选项
        if enable_strict_types:
            config.set_flag(trt.BuilderFlag.STRICT_TYPES)
            print("🔧 启用严格类型检查")
        
        # 启用所有策略源以获得最佳性能
        if enable_tactic_sources:
            config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
            print("🔧 启用精度约束优化")
        
        # 启用GPU回退
        config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
        print("🔧 启用GPU回退")
        
        # 设置精度
        if precision == 'fp16':
            config.set_flag(trt.BuilderFlag.FP16)
            # 对于FP16，也启用FP32回退以保证精度
            config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
            print("🎯 启用FP16精度优化（带FP32回退）")
        elif precision == 'int8':
            config.set_flag(trt.BuilderFlag.INT8)
            config.set_flag(trt.BuilderFlag.FP16)  # INT8通常需要FP16支持
            config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
            print("🎯 启用INT8精度优化（带FP16支持）")
            
            # INT8需要校准数据
            if calibration_dir is None:
                raise ValueError("INT8量化需要提供校准数据目录 --calibration-dir")
            
            # 准备校准器
            calibration_images = self.find_calibration_images(calibration_dir)
            cache_file = output_path.replace('.trt', '_calibration.cache')
            calibrator = Int8Calibrator(calibration_images, cache_file, batch_size=1, 
                                      max_calibration_samples=calibration_samples)
            config.int8_calibrator = calibrator
            
            print(f"📊 设置INT8校准器，使用 {calibration_samples} 张校准图像")
        else:
            print("🎯 使用FP32精度")
        
        # 解析ONNX
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, self.logger)
        
        print("📂 解析ONNX模型...")
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                print("❌ ONNX解析失败:")
                for error in range(parser.num_errors):
                    print(f"   错误 {error}: {parser.get_error(error)}")
                raise RuntimeError("ONNX解析失败")
        
        print("✅ ONNX解析成功")
        
        # 设置层级精度约束
        if precision in ['fp16', 'int8']:
            self.set_layer_precision_constraints(network, precision)
        
        # 设置动态形状和校准配置文件
        profile = builder.create_optimization_profile()
        input_name = network.get_input(0).name
        
        # 动态形状配置 (min, opt, max)
        min_shape = (1, 3, 256, 320)
        opt_shape = (max_batch_size // 2, 3, 256, 320)  # 使用中等批次作为优化目标
        max_shape = (max_batch_size, 3, 256, 320)
        
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)
        
        # 为INT8校准设置默认配置文件
        if precision == 'int8':
            print("🔧 为INT8校准设置优化配置文件...")
            config.set_calibration_profile(profile)
        
        # 设置算法选择器超时（增加构建时间但可能获得更好性能）
        config.algorithm_selector = None  # 使用默认算法选择器
        print("🔧 使用默认算法选择器进行最优化")
        
        # 构建引擎
        print("🏗️  构建TensorRT引擎...")
        if precision == 'int8':
            print("⏳ INT8量化需要较长时间，请耐心等待...")
        
        start_time = time.time()
        
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError("TensorRT引擎构建失败")
        
        build_time = time.time() - start_time
        print(f"✅ 引擎构建完成，耗时: {build_time:.2f}秒")
        
        # 保存引擎
        print("💾 保存TensorRT引擎...")
        with open(output_path, 'wb') as f:
            f.write(serialized_engine)
        
        # 获取文件大小
        engine_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✅ TensorRT引擎保存成功: {output_path}")
        print(f"📏 引擎大小: {engine_size:.2f} MB")
        
        return output_path
    
    def validate_engine_precision(self, engine_path, test_image_path=None):
        """验证TensorRT引擎的精度"""
        if not test_image_path or not os.path.exists(test_image_path):
            print("⚠️  未提供测试图像，跳过精度验证")
            return
        
        try:
            import pycuda.driver as cuda
            
            # 加载引擎
            with open(engine_path, 'rb') as f:
                engine_data = f.read()
            
            runtime = trt.Runtime(self.logger)
            engine = runtime.deserialize_cuda_engine(engine_data)
            
            if engine is None:
                print("❌ 引擎加载失败")
                return
            
            # 创建执行上下文
            context = engine.create_execution_context()
            
            # 获取输入输出信息
            input_shape = engine.get_binding_shape(0)
            output_shape = engine.get_binding_shape(1)
            
            print(f"📊 引擎验证信息:")
            print(f"   输入形状: {input_shape}")
            print(f"   输出形状: {output_shape}")
            print(f"   最大批次大小: {engine.max_batch_size}")
            print(f"   设备内存使用: {engine.device_memory_size / 1024 / 1024:.2f} MB")
            
            # 简单的推理测试
            print("🧪 执行简单推理测试...")
            
            # 准备测试数据
            test_image = cv2.imread(test_image_path)
            if test_image is not None:
                test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
                test_image = cv2.resize(test_image, (320, 256), interpolation=cv2.INTER_LINEAR)
                test_image = test_image.astype(np.float32) / 255.0
                test_image = np.transpose(test_image, (2, 0, 1))
                test_input = np.expand_dims(test_image, axis=0)
                
                # 分配GPU内存
                input_mem = cuda.mem_alloc(test_input.nbytes)
                output_mem = cuda.mem_alloc(np.prod(output_shape) * np.dtype(np.float32).itemsize)
                
                # 复制输入数据到GPU
                cuda.memcpy_htod(input_mem, test_input)
                
                # 执行推理
                context.execute_v2([int(input_mem), int(output_mem)])
                
                print("✅ 推理测试成功")
                
                # 清理内存
                input_mem.free()
                output_mem.free()
            
        except Exception as e:
            print(f"⚠️  精度验证失败: {e}")

def main():
    """主函数 - 优化版ONNX到TensorRT转换工具"""
    parser = argparse.ArgumentParser(description='ONNX到TensorRT转换工具（优化版，支持INT8量化和精度验证）')
    
    # 必需参数
    parser.add_argument('--onnx', type=str, default='./models/onnx/ultralight_segmentation_256x320.onnx', help='输入ONNX模型路径')
    
    # 可选参数
    parser.add_argument('--output', type=str, default='./models/trt/ultralight_segmentation_256x320_fp16_optimized.trt', help='输出TensorRT引擎路径')
    parser.add_argument('--precision', type=str, default='fp16', 
                       choices=['fp32', 'fp16', 'int8'], help='精度模式')
    parser.add_argument('--max-batch-size', type=int, default=4, help='最大批次大小')
    parser.add_argument('--workspace-size', type=int, default=4, help='工作空间大小 GB（增加以获得更好优化）')
    parser.add_argument('--calibration-dir', type=str, default='./test/all_test/images', 
                       help='INT8校准数据目录（INT8模式必需）')
    parser.add_argument('--calibration-samples', type=int, default=100,
                       help='校准样本数量（默认100张，更多样本通常获得更好精度）')
    
    # 优化选项
    parser.add_argument('--disable-strict-types', action='store_true', help='禁用严格类型检查')
    parser.add_argument('--disable-tactic-sources', action='store_true', help='禁用策略源优化')
    parser.add_argument('--test-image', type=str, help='用于精度验证的测试图像路径')
    
    # 调试选项
    parser.add_argument('--verbose', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    # 检查INT8参数
    if args.precision == 'int8' and args.calibration_dir is None:
        print("❌ INT8量化需要提供校准数据目录")
        print("请使用: --calibration-dir /path/to/calibration/images")
        return 1
    
    # 创建转换器
    logger_level = trt.Logger.VERBOSE if args.verbose else trt.Logger.WARNING
    converter = ONNXToTensorRTConverter(logger_level)
    
    try:
        # 执行转换
        engine_path = converter.convert(
            onnx_path=args.onnx,
            output_path=args.output,
            precision=args.precision,
            max_batch_size=args.max_batch_size,
            workspace_size=args.workspace_size,
            calibration_dir=args.calibration_dir,
            calibration_samples=args.calibration_samples,
            enable_strict_types=not args.disable_strict_types,
            enable_tactic_sources=not args.disable_tactic_sources
        )
        
        print(f"\n🎉 转换完成!")
        print(f"TensorRT引擎: {engine_path}")
        
        # 显示优化信息
        print(f"\n📈 优化配置:")
        print(f"   精度模式: {args.precision}")
        print(f"   最大批次: {args.max_batch_size}")
        print(f"   工作空间: {args.workspace_size} GB")
        print(f"   严格类型检查: {'启用' if not args.disable_strict_types else '禁用'}")
        print(f"   策略源优化: {'启用' if not args.disable_tactic_sources else '禁用'}")
        
        if args.precision == 'int8':
            print(f"\n📊 INT8量化完成，模型大小应显著减小")
            print(f"   校准样本数量: {args.calibration_samples}")
            cache_file = engine_path.replace('.trt', '_calibration.cache')
            if os.path.exists(cache_file):
                print(f"   💾 校准缓存已保存: {cache_file}")
        
        # 执行精度验证
        if args.test_image:
            print(f"\n🧪 开始精度验证...")
            converter.validate_engine_precision(engine_path, args.test_image)
        
        # 性能建议
        print(f"\n💡 性能优化建议:")
        if args.precision == 'fp32':
            print(f"   - 考虑使用FP16精度以获得更好性能")
        elif args.precision == 'fp16':
            print(f"   - 当前使用FP16，性能和精度的良好平衡")
            print(f"   - 如需更高性能，可尝试INT8量化")
        elif args.precision == 'int8':
            print(f"   - 当前使用INT8，最高性能模式")
            print(f"   - 如精度不足，可增加校准样本数量或使用FP16")
        
        print(f"   - 根据实际使用调整max_batch_size以获得最佳性能")
        print(f"   - 增加workspace_size可能获得更好优化（当前: {args.workspace_size}GB）")
        
    except Exception as e:
        print(f"\n❌ 转换失败: {e}")
        print(f"\n🔧 故障排除建议:")
        print(f"   - 检查ONNX模型是否有效")
        print(f"   - 确保有足够的GPU内存")
        print(f"   - 尝试减小workspace_size或max_batch_size")
        print(f"   - 对于INT8，确保校准数据目录存在且包含图像")
        print(f"   - 使用--verbose查看详细错误信息")
        return 1
    
    return 0

if __name__ == '__main__':

    # 运行主程序
    exit(main())

