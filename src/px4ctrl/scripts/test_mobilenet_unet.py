import onnxruntime as ort
import numpy as np
import time
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

# 加载图像
img_path = "/home/nv/px4_policy_deploy_plus/src/px4ctrl/scripts/image_raw.png"

image = cv2.imread(img_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

onnx_model_path = "/home/nv/px4_policy_deploy_plus/src/px4ctrl/scripts/mobilenetv2_unet.onnx"
providers = ["AzureExecutionProvider"]
transform = A.Compose([
    A.Resize(height=256, width=320),
    A.Normalize(),
    ToTensorV2()
])

transformed = transform(image=image_rgb)
transformed = transformed["image"].unsqueeze(0)
input_tensor = transformed.numpy()
# 创建 ONNX 推理会话
# sess = ort.InferenceSession(onnx_model_path, providers=providers)
sess = ort.InferenceSession(onnx_model_path)
for _ in range(40):
    outputs = sess.run(None, {"input": input_tensor})

start = time.time()
# 图像预处理
transform = A.Compose([
    A.Resize(height=256, width=320),
    A.Normalize(),
    ToTensorV2()
])

transformed = transform(image=image_rgb)
transformed = transformed["image"].unsqueeze(0)
input_tensor = transformed.numpy()


# 计时开始

outputs = sess.run(None, {"input": input_tensor})[0][0, 0] 
mask_img = (outputs > 0.5).astype(np.uint8) * 255 
print(mask_img.shape)
end = time.time()
 # 二值化
print(f"Inference time: {end - start:.6f} seconds")
