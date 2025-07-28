import sys
import threading
import os
import termios
import time
import numpy as np
import cv2
import queue
from ctypes import *

import rospy
from nav_msgs.msg import Odometry
from mavros_msgs.msg import AttitudeTarget
from mavros_msgs.msg import RCIn
from sensor_msgs.msg import Imu
from std_msgs.msg import Empty
# import onnxruntime as ort
# import albumentations as A
# from albumentations.pytorch import ToTensorV2

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import torch
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # Automatically initialize PyCUDA

from ctypes import Structure, c_float, byref
from contextlib import contextmanager

sys.path.append("/opt/MVS/Samples/aarch64/Python/MvImport")
from MvCameraControl_class import *
g_bExit = False

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
INTERVAL = 0.016  # nearly 60fps

ACT_MEAN = torch.tensor([13., 0., 0., 0.], dtype=torch.float32, device='cpu')
ACT_STD = torch.tensor([7., 8., 2., 1.], dtype=torch.float32, device='cpu')
# ACT_STD = torch.tensor([7., 6., 2., 1.], dtype=torch.float32, device='cpu')
# ACT_STD = torch.tensor([7., 3., 5., 1.], dtype=torch.float32, device='cpu')

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)



class CamCtrlNode:
    def __init__(self):
        rospy.init_node('cam_ctrl_node')

        # thread management
        self.g_bExit = False
        self.latest_odom = None
        self.latest_imu = None
        self.odom_lock = threading.Lock()
        self.imu_lock = threading.Lock()
        self.latest_img = None
        self.img_ready = threading.Condition()
        
        # load model
        self.if_tensorrt = rospy.get_param('use_tensorrt')
        model_path = rospy.get_param('model_path')
        self.only_photo = rospy.get_param('only_photo', False)
        if not self.only_photo:
            self.load_model(model_path)

        # node states
        self.receive_rc_cmd = False
        self.last_gap_pos = np.array(rospy.get_param('last_gap_position'), dtype=np.float32)
        self.use_mocap_q = np.array(rospy.get_param('use_mocap_q', True), dtype=bool)
        if not self.use_mocap_q:
            self.q_imu = np.array([1., 0., 0., 0.], dtype=np.float32)
            self.roll_comp = rospy.get_param('roll_compensation', 0.0)
            self.pitch_comp = rospy.get_param('pitch_compensation', 0.0)

        # consecutive gaps
        self.num_gap = rospy.get_param('num_gaps', 1)
        self.mask_color = rospy.get_param('mask_color', ["w"])
        assert len(self.mask_color) == self.num_gap, "input mask color length must be equal to the number of gaps"
        
        self.detect_gap_id = 0  # for updating the detected gap

        # RC channel 10
        self.check_inference_mode = 0.0
        self.have_init_last_check_inference_mode = False
        self.last_check_inference_mode = 0.0

        # stop trigger
        self.auto_stop = np.array(rospy.get_param('auto_stop', True), dtype=bool)
        if self.auto_stop:
            self.stop_pub = rospy.Publisher('/attitude_recovery', Empty, queue_size=10)
            self.fsm_start_pub = rospy.Publisher('/px4ctrl/fsm_start', Empty, queue_size=10)
            self.stop_counter = 0

        # inference warmup
        self.warmup_done = False
        self.warmup_total_counts = rospy.get_param('warmup_times', 300.0)
        self.warmup_counter = 0

        # obs variables
        self.img_obs = torch.zeros((1, 256, 320), dtype=torch.float32, device='cuda')
        self.vec_obs = torch.zeros(6, dtype=torch.float32, device='cuda') if not self.if_tensorrt \
            else torch.zeros(6, dtype=torch.float32, device='cpu')
        self.p = np.array([-4., 0., 1.], dtype=np.float32)  # only for switching to hover mode
        self.q = np.array([1., 0., 0., 0.], dtype=np.float32)
        self.euler_cpu = torch.zeros(3, dtype=torch.float32, device='cpu')
        self.euler_gpu = torch.zeros(3, dtype=torch.float32, device='cuda')
        self.act_cpu = torch.zeros(4, dtype=torch.float32, device='cpu')
        self.act_gpu = torch.zeros(4, dtype=torch.float32, device='cuda')

        self.last_cmd = torch.zeros(4, dtype=torch.float32, device='cuda') if not self.if_tensorrt else \
            torch.zeros(4, dtype=torch.float32, device='cpu')
        self.hidden_dim = rospy.get_param('hidden_dim')
        self.hidden_state = torch.zeros((1, 1, self.hidden_dim), dtype=torch.float32, device='cuda') if not self.if_tensorrt else \
            torch.zeros((1, 1, self.hidden_dim), dtype=torch.float32, device="cpu")

        # ROS publishers and subscribers
        self.odom_sub = rospy.Subscriber('/ekf/ekf_odom', Odometry, self.odom_callback, queue_size=1)
        self.imu_sub = rospy.Subscriber('/mavros/imu/data', Imu, self.imu_callback, queue_size=1, tcp_nodelay=True)
        self.ctrl_pub = rospy.Publisher('/hil_node/fcu_ctrl', AttitudeTarget, queue_size=1)
        self.rc_in_sub = rospy.Subscriber('/mavros/rc/in', RCIn, self.rc_cmd_cb, queue_size=1, tcp_nodelay=True)
        self.visualze = rospy.get_param('visualization', False)
        self.publish_image_raw = True

        if self.visualze:
            self.bridge = CvBridge()
            self.image_pub = rospy.Publisher('/camera/image', Image, queue_size=1)
            self.image_debug_pub = rospy.Publisher('/camera/image_debug', Image, queue_size=1)

            self.image_raw_pub = rospy.Publisher('/camera/image_raw', Image, queue_size=1)


        # camera setup
        rgb_config = rospy.get_param("if_rgb", False)
        if rgb_config:
            self.if_rgb = True
        else:
            self.if_rgb = False if self.mask_color == ["w"] else True
        self.cam_config = rospy.get_param('camera_config')
        self.exposure_time = rospy.get_param('exposure_time')

        #self.engine_path ="/home/nv/m_unet_trt/trt_engine/trt_tiny/mobilenetv2_unet_tiny_fp16.trt"
        self.engine_path="/home/nv/m_unet_trt/trt_engine/trt_normal/mobilenetv2_unet_fp16.trt"            #量化加速后的模型替换位置
        self.batch_size = 1
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        # 加载引擎
        self.engine = self._load_engine()
        self.context = self.engine.create_execution_context()
        # 获取输入输出信息
        self.input_info, self.output_info = self._get_io_info()
        # 分配内存
        self._allocate_memory()

        self.index=502
        self.setup_camera()
        self.start_threads()

    def _load_engine(self):
        with open(self.engine_path, 'rb') as f:
            engine_data = f.read()
        engine = self.runtime.deserialize_cuda_engine(engine_data)
        if engine is None:
            raise RuntimeError(f"❌ 无法加载引擎: {self.engine_path}")
        return engine

    def _get_io_info(self):
        input_info = {}
        output_info = {}

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            mode = self.engine.get_tensor_mode(name)

            if mode == trt.TensorIOMode.INPUT:
                input_info = {'name': name, 'shape': shape}
            else:
                output_info = {'name': name, 'shape': shape}

        if input_info['shape'][0] == -1:
            input_shape = list(input_info['shape'])
            input_shape[0] = self.batch_size
            input_info['actual_shape'] = tuple(input_shape)
            self.context.set_input_shape(input_info['name'], input_info['actual_shape'])
        else:
            input_info['actual_shape'] = input_info['shape']

        output_info['actual_shape'] = self.context.get_tensor_shape(output_info['name'])

        return input_info, output_info

    def _allocate_memory(self):
        input_size = int(np.prod(self.input_info['actual_shape']) * np.dtype(np.float32).itemsize)
        output_size = int(np.prod(self.output_info['actual_shape']) * np.dtype(np.float32).itemsize)

        self.d_input = cuda.mem_alloc(input_size)
        self.d_output = cuda.mem_alloc(output_size)
        self.h_output = np.empty(self.output_info['actual_shape'], dtype=np.float32)


    def load_model(self, model_path):
        try:
            if not self.if_tensorrt:
                self.model = torch.jit.load(model_path)
                self.model.to('cuda')
                self.model.eval()
            else:
                rospy.loginfo('[CamCtrl] Loading TensorRT engine...')
                with open(model_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
                    self.engine = runtime.deserialize_cuda_engine(f.read())
                
                self.inputs = []
                self.outputs = []
                self.bindings = []
                self.stream = cuda.Stream()

                for binding in self.engine:
                    size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
                    dtype = trt.nptype(self.engine.get_binding_dtype(binding))
                    host_mem = cuda.pagelocked_empty(size, dtype)
                    device_mem = cuda.mem_alloc(host_mem.nbytes)
                    self.bindings.append(int(device_mem))
                    if self.engine.binding_is_input(binding):
                        self.inputs.append({'host': host_mem, 'device': device_mem})
                    else:
                        self.outputs.append({'host': host_mem, 'device': device_mem})
                self.context = self.engine.create_execution_context()
            rospy.loginfo('\033[32m[CamCtrl] Model loaded successfully!\033[0m')
        except torch.jit.Error as e:
            self.only_photo = True
            rospy.logerr(f'[CamCtrl] Error loading the model: {str(e)}! Only taking photo now.')

    def setup_camera(self):
        # Initialize SDK
        MvCamera.MV_CC_Initialize()

        SDKVersion = MvCamera.MV_CC_GetSDKVersion()
        print ("[CamCtrl] <Camera Setup> Camera setup begins... SDKVersion[0x%x]" % SDKVersion)

        deviceList = MV_CC_DEVICE_INFO_LIST()
        tlayerType = (MV_GIGE_DEVICE | MV_USB_DEVICE | MV_GENTL_CAMERALINK_DEVICE | MV_GENTL_CXP_DEVICE | MV_GENTL_XOF_DEVICE)

        # Enum device
        ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
        if ret != 0:
            print ("\033[31m Enum devices fail! ret[0x%x]\033[0m" % ret)
            sys.exit()

        if deviceList.nDeviceNum == 0:
            print ("\033[31m Find no camera device!\033[0m")
            sys.exit()

        print ("Find %d devices!" % deviceList.nDeviceNum)

        for i in range(0, deviceList.nDeviceNum):
            mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE or mvcc_dev_info.nTLayerType == MV_GENTL_GIGE_DEVICE:
                print ("\ngige device: [%d]" % i)
                strModeName = ""
                for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName:
                    strModeName = strModeName + chr(per)
                print ("device model name: %s" % strModeName)

                nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
                nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
                nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
                nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
                print ("current ip: %d.%d.%d.%d\n" % (nip1, nip2, nip3, nip4))
            elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
                print ("\nu3v device: [%d]" % i)
                strModeName = ""
                for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName:
                    if per == 0:
                        break
                    strModeName = strModeName + chr(per)
                print ("device model name: %s" % strModeName)

                strSerialNumber = ""
                for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
                    if per == 0:
                        break
                    strSerialNumber = strSerialNumber + chr(per)
                print ("user serial number: %s" % strSerialNumber)
            elif mvcc_dev_info.nTLayerType == MV_GENTL_CAMERALINK_DEVICE:
                print ("\nCML device: [%d]" % i)
                strModeName = ""
                for per in mvcc_dev_info.SpecialInfo.stCMLInfo.chModelName:
                    if per == 0:
                        break
                    strModeName = strModeName + chr(per)
                print ("device model name: %s" % strModeName)

                strSerialNumber = ""
                for per in mvcc_dev_info.SpecialInfo.stCMLInfo.chSerialNumber:
                    if per == 0:
                        break
                    strSerialNumber = strSerialNumber + chr(per)
                print ("user serial number: %s" % strSerialNumber)
            elif mvcc_dev_info.nTLayerType == MV_GENTL_XOF_DEVICE:
                print ("\nXoF device: [%d]" % i)
                strModeName = ""
                for per in mvcc_dev_info.SpecialInfo.stXoFInfo.chModelName:
                    if per == 0:
                        break
                    strModeName = strModeName + chr(per)
                print ("device model name: %s" % strModeName)

                strSerialNumber = ""
                for per in mvcc_dev_info.SpecialInfo.stXoFInfo.chSerialNumber:
                    if per == 0:
                        break
                    strSerialNumber = strSerialNumber + chr(per)
                print ("user serial number: %s" % strSerialNumber)
            elif mvcc_dev_info.nTLayerType == MV_GENTL_CXP_DEVICE:
                print ("\nCXP device: [%d]" % i)
                strModeName = ""
                for per in mvcc_dev_info.SpecialInfo.stCXPInfo.chModelName:
                    if per == 0:
                        break
                    strModeName = strModeName + chr(per)
                print ("device model name: %s" % strModeName)

                strSerialNumber = ""
                for per in mvcc_dev_info.SpecialInfo.stCXPInfo.chSerialNumber:
                    if per == 0:
                        break
                    strSerialNumber = strSerialNumber + chr(per)
                print ("user serial number: %s" % strSerialNumber)

        if sys.version >= '3':
            nConnectionNum = 0
        else:
            rospy.loginfo("\033[31m[CamCtrl] <Camera Setup> Please use Python 3.x!\033[0m")
            return
        
        # Creat Camera Object
        self.cam = MvCamera()
        
        # Select device and create handle
        stDeviceList = cast(deviceList.pDeviceInfo[int(nConnectionNum)], POINTER(MV_CC_DEVICE_INFO)).contents
        ret = self.cam.MV_CC_CreateHandle(stDeviceList)
        if ret != 0:
            print ("\033[31m Create handle fail! ret[0x%x]" % ret)
            sys.exit()

        # Open device
        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            print ("\033[31m Open device fail! ret[0x%x]\033[0m" % ret)
            sys.exit()
        
        # Save the current camera config
        # ret = self.cam.MV_CC_FeatureSave('config')

        # Load the default camera config
        ret = self.cam.MV_CC_FeatureLoad(self.cam_config)
        if ret != 0:
            print ("\033[31m Set camera config fail! ret[0x%x]\033[0m" % ret)
            sys.exit()  

        ret = self.cam.MV_CC_SetFloatValue("ExposureTime", self.exposure_time)

        if ret != 0:
            print ("\033[31m Set camera config fail! ret[0x%x]\033[0m" % ret)
            sys.exit()

        # Detection network optimal package size (It only works for the GigE camera)
        if stDeviceList.nTLayerType == MV_GIGE_DEVICE or stDeviceList.nTLayerType == MV_GENTL_GIGE_DEVICE:
            nPacketSize = self.cam.MV_CC_GetOptimalPacketSize()
            if int(nPacketSize) > 0:
                ret = self.cam.MV_CC_SetIntValue("GevSCPSPacketSize",nPacketSize)
                if ret != 0:
                    print ("Warning: Set Packet Size fail! ret[0x%x]" % ret)
            else:
                print ("Warning: Get Packet Size fail! ret[0x%x]" % nPacketSize)

        # Get payload size
        stParam =  MVCC_INTVALUE()
        memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))
        ret = self.cam.MV_CC_GetIntValue("PayloadSize", stParam)
        if ret != 0:
            print ("\033[31m Get payload size fail! ret[0x%x]\033[0m" % ret)
            sys.exit()
        self.nPayloadSize = stParam.nCurValue

        # For Bayer RGB output
        if self.if_rgb:
            ret = self.cam.MV_CC_SetBayerCvtQuality(0)
            if ret != 0:
                print ("set Bayer convert quality fail! ret[0x%x]" % ret)

        # Start grab images
        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            print ("\033[31m Start grabbing fail! ret[0x%x]\033[0m" % ret)
            sys.exit()

        self.data_buf = (c_ubyte * self.nPayloadSize)()

    def camera_thread(self, cam: MvCamera, pData=0, nDataSize=0):
        stOutFrame = MV_FRAME_OUT()
        memset(byref(stOutFrame), 0, sizeof(stOutFrame))

        while not self.g_bExit: 
            ret = cam.MV_CC_SetCommandValue("TriggerSoftware")
            time_trigger = time.perf_counter()
            
            if ret != 0:
                print ("\033[31m[CamCtrl] <Camera Thread> Failed in TriggerSoftware!\033[0m")
                
            ret = cam.MV_CC_GetImageBuffer(stOutFrame, 1000)

            time_get_frame = time.perf_counter()
            get_time = time_get_frame - time_trigger
            # if get_time > 0.008:
            if get_time > 0.01:
                print (f"\033[33mImage get time: {get_time*1000:.2f}ms, which is too long!\033[0m")
            else:
                # print (f"Image get time: {get_time*1000:.2f}ms")
                pass
            if ret == 0:

                # print ("get one frame: Width[%d], Height[%d], PixelType[0x%x], nFrameNum[%d]"  % (stOutFrame.stFrameInfo.nWidth, stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.enPixelType, stOutFrame.stFrameInfo.nFrameNum)) 
                if not self.if_rgb:
                    nSize = stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight
                    img_buff = (c_ubyte * nSize)()
                    memmove(byref(img_buff), stOutFrame.pBufAddr, nSize)

                    img = np.asarray(img_buff) 


                    img = img.reshape(stOutFrame.stFrameInfo.nHeight , stOutFrame.stFrameInfo.nWidth, -1)

                    # save_path = "/home/nv/px4_policy_deploy_plus/src/px4ctrl/scripts/image_raw.png"  # 你可以换成任何想保存的位置
                    # cv2.imwrite(save_path, img)
                    if self.publish_image_raw:
                        rospy.loginfo("publishing image raw")
                        img_msg = self.bridge.cv2_to_imgmsg(img, encoding="mono8")
                        # rospy.loginfo("publishing image raw")
                        # img_msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8") 
                        self.image_raw_pub.publish(img_msg)
                        

                    convert_time = time.perf_counter()
                    # print (f"convert time: {(convert_time - time_get_frame)*1000:.2f}ms")

                    # _, img = cv2.threshold(img, 64, 255, cv2.THRESH_BINARY)
                    img = cv2.resize(img, (320, 256), cv2.INTER_AREA)  # downsampling
                    _, img = cv2.threshold(img, 96, 255, cv2.THRESH_BINARY)
                    # rospy.loginfo("check3")control_
                else:

                    nRGBSize = stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight * 3

                    stConvertParam = MV_CC_PIXEL_CONVERT_PARAM_EX()
                    memset(byref(stConvertParam), 0, sizeof(stConvertParam))
                    stConvertParam.nWidth = stOutFrame.stFrameInfo.nWidth
                    stConvertParam.nHeight = stOutFrame.stFrameInfo.nHeight
                    stConvertParam.pSrcData = stOutFrame.pBufAddr
                    stConvertParam.nSrcDataLen = stOutFrame.stFrameInfo.nFrameLen
                    stConvertParam.enSrcPixelType = stOutFrame.stFrameInfo.enPixelType
                    stConvertParam.enDstPixelType = PixelType_Gvsp_RGB8_Packed 
                    stConvertParam.pDstBuffer = (c_ubyte * nRGBSize)()
                    stConvertParam.nDstBufferSize = nRGBSize

                    ret = cam.MV_CC_ConvertPixelTypeEx(stConvertParam)
                    if ret != 0:
                        print ("\033[31m[CamCtrl] <Camera Thread> Convert pixel fail! ret[0x%x]\033[0m" % ret)
                        sys.exit()

                    cam.MV_CC_FreeImageBuffer(stOutFrame)

                    time_convert = time.perf_counter()
                    read_time = time_convert - time_trigger
                    # if read_time > 0.010:
                    if read_time > 0.013:
                        print (f"\033[33mImage read time: {read_time*1000:.2f}ms, which is too long!\033[0m")
                    else:
                        # print (f"Image read time: {read_time*1000:.2f}ms")
                        pass

                    img_buff = (c_ubyte * stConvertParam.nDstLen)()
                    memmove(byref(img_buff), stConvertParam.pDstBuffer, stConvertParam.nDstLen)

                    img_raw = np.asarray(img_buff)

                    img_raw = img_raw.reshape(stOutFrame.stFrameInfo.nHeight , stOutFrame.stFrameInfo.nWidth, -1)
                    # class MVCC_FLOATVALUE(Structure):
                    #     _fields_ = [
                    #         ("fCurValue", c_float),
                    #         ("fMax", c_float),
                    #         ("fMin", c_float),
                    #         ("fInc", c_float),
                    #     ]
                    # exposure_time = MVCC_FLOATVALUE()

                    # # 获取曝光时间，接口需要传入 "ExposureTime"
                    # ret = cam.MV_CC_GetFloatValue("ExposureTime", byref(exposure_time))
                    # if ret == 0:
                    #     with open("/home/nv/px4_policy_deploy_plus/src/px4ctrl/scripts/imgshape.txt", "w") as f:
                    #         f.write(str(img_raw.shape)+","+f"ExposureTime:{exposure_time.fCurValue:.2f}")
                    # else:
                    #     with open("/home/nv/px4_policy_deploy_plus/src/px4ctrl/scripts/imgshape.txt", "w") as f:
                    #         f.write(str(img_raw.shape)+","+f"ExposureTime:not get")

                    # if self.index<=500:
                    #     img_BGR= cv2.cvtColor(img_raw, cv2.COLOR_RGB2BGR)
                    #     save_path = f"/home/nv/px4_policy_deploy_plus/src/px4ctrl/scripts/imgs/image_{self.index}.png"  # 你可以换成任何想保存的位置
                    #     print(f"save image_{self.index}")
                    #     cv2.imwrite(save_path, img_BGR)
                    #     self.index+=1
                    img_raw = cv2.resize(img_raw, (320, 256), cv2.INTER_AREA)  # downsampling

                    
                
                with self.img_ready:
                    self.latest_img = img_raw
                    self.img_ready.notify()

                end_time = time.perf_counter()

                # process_time = end_time - convert_time
                # print (f"process time: {process_time*1000:.2f}ms")

                total_time = end_time - time_trigger
                if total_time > INTERVAL - 0.000:
                    print (f"Total image preparation time: {total_time*1000:.2f}ms") 

                sleep_time = max(0, INTERVAL - total_time)  

                # print ("sleep time", sleep_time)
                time.sleep(sleep_time) 

                next_trigger = time.perf_counter()
                fps = 1.0 / (next_trigger - time_trigger)
                if fps < 58.0:
                    print (f"Image stream fps: {fps:.2f}Hz\n")
                # print (f"trigger time: {(next_trigger - time_trigger)*1000:.2f}ms\n")
                
                if not self.if_rgb:
                    cam.MV_CC_FreeImageBuffer(stOutFrame)
            else:
                print ("no data[0x%x]" % ret)
    
    def calculate_mask_learning(self, img_raw) -> np.ndarray:
        """
        使用TensorRT模型预测图像掩码，输出为二值图（0或255）

        Args:
            img_raw (np.ndarray): 输入图像，RGB格式，HWC排列 (256, 320, 3)

        Returns:
            np.ndarray: 二值mask图像，uint8类型，shape为 (256, 320)，值为 0 或 255
        """
        # 图像预处理：模型默认通道顺序为CHW，归一化见 preprocess_image
        img_input = img_raw.astype(np.float32) / 255.0
        img_input = (img_input - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        img_input = img_input.transpose(2, 0, 1)  # HWC -> CHW
        img_input = img_input[np.newaxis, ...]  # 增加 batch 维度 (1, C, H, W)
        img_input = np.ascontiguousarray(img_input.astype(np.float32))

        # 推理
        cuda.memcpy_htod(self.d_input, img_input)
        self.context.set_tensor_address(self.input_info['name'], int(self.d_input))
        self.context.set_tensor_address(self.output_info['name'], int(self.d_output))
        self.context.execute_v2([int(self.d_input), int(self.d_output)])
        cuda.memcpy_dtoh(self.h_output, self.d_output)

        # 提取预测掩码 (1, 1, H, W)
        pred_mask = self.h_output[0, 0]
        binary_mask = (pred_mask > 0.5).astype(np.uint8) * 255  # 转换为 uint8 掩码

        return binary_mask


                                
    def calculate_mask_old(self, img_raw) -> np.array:
        r, g, b = img_raw[:, :, 0], img_raw[:, :, 1], img_raw[:, :, 2]
        if self.mask_color[self.detect_gap_id] == "w":
            mask = (r > 65) & (g > 65) & (b > 65)  
        elif self.mask_color[self.detect_gap_id] == "r":
            mask = (r > 20) & (g < 80) & (b < 80)  
        elif self.mask_color[self.detect_gap_id] == "g":
            raise NotImplementedError("Green mask is not implemented yet!")
        elif self.mask_color[self.detect_gap_id] == "b":
            mask = (b > 150) & (r < 80)
        else:
            raise ValueError("Invalid mask color!")
        
        mask = mask.astype(np.uint8) * 255
        return mask
    

    def count_pixels(self, mask):
        """
        统计二值化图像中的白色和黑色像素数量
        参数:
            mask: 二值化图像
        返回:
            white_pixels: 白色像素数量
            black_pixels: 黑色像素数量
        """
        white_pixels = cv2.countNonZero(mask)
        black_pixels = mask.size - white_pixels
        return white_pixels, black_pixels

    def process_small_white_region(self, mask):
        """
        处理白色像素数量小于500的情况
        在主要白色区域中心创建一个黑色平行四边形洞
        """
        # 找到所有轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return mask
        
        # 找到最大的轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 获取最小外接矩形
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        # 计算中心点
        center = np.mean(box, axis=0)
        
        # 创建平行四边形顶点 (缩小为原矩形的60%)
        scale = 0.6
        pts = []
        for pt in box:
            # 每个点向中心点收缩
            vec = center - pt
            new_pt = pt + vec * scale
            pts.append(new_pt)
        
        # 将点转换为整数坐标
        pts = np.array(pts, dtype=np.int32)
        
        # 创建黑色平行四边形洞
        result = mask.copy()
        cv2.fillPoly(result, [pts], color=0)  # 用黑色填充
        
        return result

    def calculate_mask_hsv(self, img_raw) -> np.array:     

        hsv_ranges = [
            (np.array([0, 120, 15]), np.array([7, 160, 35])),     
            (np.array([170, 110, 15]), np.array([180, 160, 35])),
            (np.array([0, 90, 25]), np.array([7, 120, 35])),
        ]
        img_raw = cv2.cvtColor(img_raw, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(img_raw, cv2.COLOR_BGR2HSV)
        combined_mask = np.zeros(img_raw.shape[:2], dtype=np.uint8)

        # Step 1: 基于 HSV 范围生成初始掩码
        for (lower, upper) in hsv_ranges:
            mask = cv2.inRange(hsv, lower, upper)
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        # Step 2: 闭运算处理（填补空洞 + 平滑）
        kernel_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_big)
        # kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        # combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_small)

        # 统计像素
        white_pixels, black_pixels = self.count_pixels(combined_mask)
        # print(f"白色像素数量: {white_pixels}")
        # print(f"黑色像素数量: {black_pixels}")
        # 如果白色像素少于500，进行处理
        if white_pixels < 500:
            combined_mask = self.process_small_white_region(combined_mask)
            # print("已对白色区域添加黑色洞")
            
        return combined_mask
    
    
    def update_gap_id(self, img_raw: np.array, mask) -> np.array:
        if self.detect_gap_id < self.num_gap - 1:
            if np.sum(mask) < 8:
                rospy.loginfo("\033[33m[CamCtrl] Detect the next gap!!!!!\033[0m")
                self.detect_gap_id += 1
                return self.calculate_mask_hsv(img_raw)
            else:
                return mask
        else:
            return mask
    
    def white_mask_gray(self, img):
        _, img = cv2.threshold(img, 96, 255, cv2.THRESH_BINARY)

    def red_mask_rgb(self, img):  # TODO: bug here
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        img = (r > 30) & (g < 30) & (b < 30)  # TODO: check img type

    @contextmanager
    def cuda_thread_context(self):
        ctx = pycuda.autoinit.context
        ctx.push()
        try:
            yield
        finally:
            ctx.pop()
            
    def control_thread(self):
        while not self.g_bExit:
            img = None
            with self.img_ready:
                self.img_ready.wait(timeout=INTERVAL+0.01)
                try:
                    start_time = time.perf_counter()
                    
                    img_raw = self.latest_img
                    
                    # r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
                    # img = (r > 20) & (g < 80) & (b < 80)  # TODO: check img type
                    # img = img.astype(np.uint8) * 255
                    calculate_start_time = time.perf_counter()
                    with self.cuda_thread_context():
                        img = self.calculate_mask_learning(img_raw)
                    #img=self.calculate_mask_hsv(img_raw)
                    calculate_end_time = time.perf_counter()

                    print (f"******************* time: {(calculate_end_time-calculate_start_time)*1000:.2f}ms") 

                    img = self.update_gap_id(img_raw, img)     #useless?

                    # cv2.imwrite('img_%d.jpg' % stOutFrame.stFrameInfo.nFrameNum, img)
                    if self.visualze:  # visualize the mask
                        img_msg = self.bridge.cv2_to_imgmsg(img, encoding="mono8")
                        self.image_pub.publish(img_msg)

                    img = img / 255.0  # normalize to 0-1

                    # cv2.imshow('img_.jpg', img)
                    # cv2.waitKey(1)


                    # rospy.loginfo(f"Retrived image shape: {img.shape}")

                    # if self.only_photo and self.visualze:
                    #     img_normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
                    #     img_normalized = img_normalized.astype(np.uint8)
                    #     img_msg = self.bridge.cv2_to_imgmsg(img_normalized, encoding="mono8")
                    #     self.image_debug_pub.publish(img_msg)

                    get_img_time = time.perf_counter()
                    # rospy.loginfo(f"Retrive image time: {(get_img_time - start_time)*1000:.2f}ms")
                except Exception as e:
                    rospy.loginfo(f"\033[31mError in control thread: {str(e)}\033[0m")
                    
            if img is not None:
                self.latest_img = None
                img = cv2.resize(img, (320, 256), cv2.INTER_AREA) 
                if self.auto_stop:
                    # if np.sum(img) < 12 and self.stop_counter == 0 and self.euler_cpu[1] > np.pi / 9:
                    #     self.stop_counter = 1
                    #     rospy.loginfo("Start to count stop!")
                    # elif np.sum(img) < 12 and self.stop_counter > 0:
                    #     self.stop_counter += 1
                    if np.sum(img) < 8 and self.warmup_done:
                        if not self.stop_counter >= 12:
                            self.stop_counter += 1
                            rospy.loginfo(f"stop counter++ now is {self.stop_counter}")
                    else:
                        self.stop_counter = 0
                    # else:
                        # rospy.loginfo("seeing some bright pixel in stop mode")
                        # if not self.warmup_done:
                        #     rospy.loginfo(f"\033[31mSee nothing when taking off! Where is the gap?\033[0m")
                        # self.stop_counter = 0
                
                self.img_obs = torch.from_numpy(img).float().cuda()
                convert_device_time = time.perf_counter() - get_img_time
                if convert_device_time > 0.004:
                    rospy.loginfo(f"\033[33mConvert device time: {convert_device_time*1000:.2f}ms, which is too long!\033[0m")
                else:
                    # rospy.loginfo(f"Convert device time: {convert_device_time*1000:.2f}ms")
                    pass
                
                if not self.only_photo:  # enabling control
                    self.pixel_based_control()
                    
                    control_time = time.perf_counter() - start_time
                    if control_time < INTERVAL - 0.002:
                        # rospy.loginfo(f"Control process time: {control_time*1000:.2f}ms")
                        pass
                    else:
                        rospy.loginfo(f"\033[33mControl process time: {control_time*1000:.2f}ms, which is too long!\033[0m")
            
            elif self.receive_rc_cmd:
                rospy.loginfo("\033[31mError: No image received yet. Don't enable rc commands!\033[0m")
    
    def __del__(self):
        if hasattr(self, 'img_ready'):
            with self.img_ready:
                self.img_ready.notify_all()

    def pixel_based_control(self):
        if (self.p[0] < self.last_gap_pos[0] + 0.2 and not self.auto_stop) or (self.auto_stop and self.stop_counter < 12): # (self.auto_stop and self.stop_counter < 9):
            if not self.warmup_done or not self.receive_rc_cmd:
                self.inference()
                self.hidden_state = torch.zeros((1, 1, self.hidden_dim), dtype=torch.float32, device='cuda')
                
                if not self.if_tensorrt:
                    self.act_cpu.copy_(self.act_gpu)

                self.warmup_counter += 1

                if self.warmup_counter % 30 == 0 and self.warmup_counter < 300:
                    rospy.loginfo(f"[CamCtrl] Warmup {self.warmup_counter}/{self.warmup_total_counts}!")

                if not self.warmup_done and self.receive_rc_cmd:
                    rospy.logwarn(f"[CamCtrl] The warmup has not been done yet. Please switch back to the RC mode!")
                    self.receive_rc_cmd = False
            else:
                if self.update_state():  # must receive the first odom message
                    self.inference()

                    if not self.if_tensorrt:
                        self.act_cpu.copy_(self.act_gpu)

                    if not self.if_tensorrt:
                        self.last_cmd.copy_(self.act_gpu)
                    else:
                        self.last_cmd.copy_(self.act_cpu)
                    self.publish_control()
                    rospy.loginfo("publishing control msg")
        
        elif self.auto_stop and self.stop_counter >= 12 and self.warmup_done and self.receive_rc_cmd:
            if self.stop_counter == 12:
                rospy.loginfo("\033[32m[px4ctrl] <HIL node> Switch to blind control!\033[0m")
                self.stop_pub.publish(Empty())
                rospy.loginfo("publishing stop msg")
        # else:
            # rospy.loginfo("doing nothing")

        if self.warmup_counter >= self.warmup_total_counts and not self.warmup_done:
            rospy.loginfo(f"\033[32m[CamCtrl] Warmup done!\033[0m")
            self.warmup_done = True

    @torch.no_grad()
    def inference(self):
        if not self.if_tensorrt:
            self.euler_gpu.copy_(self.euler_cpu)
        self.vec_obs = torch.concat([self.euler_gpu[:2], self.last_cmd], dim=0)

        img_obs_batched = self.img_obs.unsqueeze(0)
        vec_obs_batched = self.vec_obs.unsqueeze(0)

        if self.if_tensorrt:
            cuda.memcpy_dtod_async(self.inputs[0]['device'], img_obs_batched.data_ptr(), img_obs_batched.element_size() * img_obs_batched.nelement(), self.stream)
            np.copyto(self.inputs[1]['host'], self.vec_obs.ravel())
            np.copyto(self.inputs[2]['host'], self.hidden_state.ravel())
            cuda.memcpy_htod_async(self.inputs[1]['device'], self.inputs[1]['host'], self.stream)
            cuda.memcpy_htod_async(self.inputs[2]['device'], self.inputs[2]['host'], self.stream)

        if not self.if_tensorrt:
            # rospy.loginfo("vec_obs now: ", self.vec_obs)
            self.act_gpu, self.hidden_state = self.model(img_obs_batched, vec_obs_batched, self.hidden_state.detach()) 
            self.act_gpu = torch.clamp(self.act_gpu, -1.0, 1.0)
            self.act_gpu = self.act_gpu.squeeze(0)
        else:
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
            cuda.memcpy_dtoh_async(self.outputs[1]['host'], self.outputs[1]['device'], self.stream)
            self.hidden_state = self.outputs[0]['host']
            self.act_cpu = torch.tensor(self.outputs[1]['host']).clamp(-1.0, 1.0).detach()  # TODO: check shape
    
    def publish_control(self):
        cmd = AttitudeTarget()
        act = self.denormalize_action(self.act_cpu)
        
        cmd.thrust = act[0].item()
        cmd.body_rate.x = act[1].clamp(-6.0, 6.0).item()
        cmd.body_rate.y = act[2].item()
        cmd.body_rate.z = act[3].item()
        
        self.ctrl_pub.publish(cmd)

    def denormalize_action(self, action: torch.Tensor):
        return action * ACT_STD + ACT_MEAN

    # def start_threads(self):
    #     try:
    #         rospy.loginfo("\033[32m[CamCtrl] Starting threads...\033[0m")
    #         self.cam_thread = threading.Thread(target=self.camera_thread, 
    #                                         args=(self.cam, byref(self.data_buf), self.nPayloadSize))
    #         rospy.loginfo("\033[32m[CamCtrl] Camera thread started successfully!\033[0m")
    #         self.ctrl_thread = threading.Thread(target=self.control_thread)
    #         rospy.loginfo("\033[32m[CamCtrl] Control thread started successfully!\033[0m")
            
    #         self.cam_thread.start()
    #         self.ctrl_thread.start()
    #     except:
    #         rospy.loginfo("\033[31mError: unable to start threads\033[0m")

    def start_threads(self):
        try:
            rospy.loginfo("\033[32m[CamCtrl] Starting threads...\033[0m")
            
            self.cam_thread = threading.Thread(target=self.camera_thread, 
                                        args=(self.cam, byref(self.data_buf), self.nPayloadSize))
            self.ctrl_thread = threading.Thread(target=self.control_thread)
            
            try:
                self.cam_thread.start()
                rospy.loginfo("\033[32m[CamCtrl] Camera thread started successfully!\033[0m")
            except Exception as e:
                rospy.loginfo(f"\033[31mError starting camera thread: {str(e)}\033[0m")
                raise
                
            try:
                self.ctrl_thread.start()
                rospy.loginfo("\033[32m[CamCtrl] Control thread started successfully!\033[0m")
            except Exception as e:
                rospy.loginfo(f"\033[31mError starting control thread: {str(e)}\033[0m")
                raise
                
        except Exception as e:
            rospy.loginfo(f"\033[31mError in start_threads: {str(e)}\033[0m")
            raise

    def odom_callback(self, msg):
        with self.odom_lock:
            self.latest_odom = msg
    
    def imu_callback(self, msg):
        with self.imu_lock:
            self.latest_imu = msg

    def update_state(self):
        with self.odom_lock:
            if not self.auto_stop and self.latest_odom is None:
                rospy.loginfo("\033[31m[CamCtrl] No odometry message received yet!\033[0m")
                return False
        
            with self.imu_lock:
                if not self.use_mocap_q and self.latest_imu is None:
                    rospy.loginfo("\033[31m[CamCtrl] No IMU message received yet!\033[0m")
                    return False
             
                msg = self.latest_odom
                msg_imu = self.latest_imu
                
                if not self.auto_stop:
                    self.p[0] = msg.pose.pose.position.x
                    self.p[1] = msg.pose.pose.position.y
                    self.p[2] = msg.pose.pose.position.z
                
                # if self.use_mocap_q:
                #     q = [msg.pose.pose.orientation.w,
                #         msg.pose.pose.orientation.x,
                #         msg.pose.pose.orientation.y,
                #         msg.pose.pose.orientation.z]
                # else:
                q = [msg_imu.orientation.w,
                    msg_imu.orientation.x,
                    msg_imu.orientation.y,
                    msg_imu.orientation.z]
                
                quatnp2eulertensor(q, self.euler_cpu)
                if not self.use_mocap_q:
                    self.euler_cpu[0] += self.roll_comp
                    self.euler_cpu[1] += self.pitch_comp

                # rospy.loginfo("Update state successfully!")
            
                return True
    
    def rc_cmd_cb(self, msg):  

        # rospy.loginfo("self.auto_stop = ",self.auto_stop)

        if not self.auto_stop:
            self.check_inference_mode = (msg.channels[9] - 1000.0) / 1000.0  # 10 channel
        else: 
            self.check_inference_mode = (msg.channels[4] - 1000.0) / 1000.0  # 5 channel # pn tag
        
        # self.check_inference_mode = (msg.channels[4] - 1000.0) / 1000.0  # 5 channel
        
        if not self.have_init_last_check_inference_mode:
            self.have_init_last_check_inference_mode = True
            self.last_check_inference_mode = self.check_inference_mode
        if self.last_check_inference_mode < 0.75 and self.check_inference_mode > 0.75 and not self.receive_rc_cmd:
            self.receive_rc_cmd = True
            # self.auto_stop = False
            self.stop_counter = 0
            # rospy.loginfo("\033[32m[CamCtrl] Receive RC Channel 5!!!\033[0m")
            rospy.loginfo("\033[32m[CamCtrl] CamCtrl Triggered by channel 5!!!\033[0m")

            
        elif self.check_inference_mode < 0.75:
            if self.receive_rc_cmd == True:
                rospy.loginfo("\033[32m[CamCtrl] CamCtrl Disabled by channel 5!!!\033[0m")
                self.hidden_state = torch.zeros((1, 1, self.hidden_dim), dtype=torch.float32, device='cuda')
                rospy.loginfo("reseting hidden state")
                self.fsm_start_pub.publish(Empty())
                rospy.loginfo("publishing fsm start trigger")


            self.receive_rc_cmd = False


    def press_q_to_exit():
        fd = sys.stdin.fileno()
        old_ttyinfo = termios.tcgetattr(fd)
        new_ttyinfo = old_ttyinfo[:]
        new_ttyinfo[3] &= ~termios.ICANON
        new_ttyinfo[3] &= ~termios.ECHO
        
        termios.tcsetattr(fd, termios.TCSANOW, new_ttyinfo)
        try:
            while True:
                key = os.read(fd, 1)
                if key == b'q':
                    break
        except:
            pass
        finally:
            termios.tcsetattr(fd, termios.TCSANOW, old_ttyinfo)

    def press_any_key_exit():
        fd = sys.stdin.fileno()
        old_ttyinfo = termios.tcgetattr(fd)
        new_ttyinfo = old_ttyinfo[:]
        new_ttyinfo[3] &= ~termios.ICANON
        new_ttyinfo[3] &= ~termios.ECHO
        #sys.stdout.write(msg)
        #sys.stdout.flush()
        termios.tcsetattr(fd, termios.TCSANOW, new_ttyinfo)
        try:
            os.read(fd, 7)
        except:
            pass
        finally:
            termios.tcsetattr(fd, termios.TCSANOW, old_ttyinfo)

def quatnp2eulertensor(quat: np.ndarray, euler: torch.Tensor):
    w = quat[0]
    x = quat[1]
    y = quat[2]
    z = quat[3]
    
    # roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    euler[0] = torch.atan2(torch.tensor(sinr_cosp), torch.tensor(cosr_cosp))
    
    # pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    
    assert -1.0 <= sinp <= 1.0, "invalid input for asin in pitch calculation"
    euler[1] = torch.asin(torch.tensor(sinp))
    
    # yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    euler[2] = torch.atan2(torch.tensor(siny_cosp), torch.tensor(cosy_cosp))

def world2body_velocity_matrix(odom_q: np.ndarray, world_vel: torch.Tensor):
    w, x, y, z = odom_q

    R = torch.tensor([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
    ], dtype=world_vel.dtype, device=world_vel.device)
    
    R_trans = R.T
    return R_trans @ world_vel

def rpy_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    # Roll (X-axis rotation)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    # Pitch (Y-axis rotation)
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    # Yaw (Z-axis rotation)
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation matrix: R = Rz * Ry * Rx
    return Rz @ Ry @ Rx
    
def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [    2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z,     2*y*z - 2*w*x],
        [    2*x*z - 2*w*y,     2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])

if __name__ == "__main__":
    try:
        node = CamCtrlNode()
        rospy.loginfo("\033[32m[CamCtrl] Node started successfully!\033[0m")
        
        rospy.spin()
        
    except KeyboardInterrupt:
        rospy.loginfo("\033[33m[CamCtrl] Detected keyboard interrupt...\033[0m")
    except Exception as e:
        rospy.loginfo(f"\033[31m[CamCtrl] Error occurred: {str(e)}\033[0m")
    finally:
        if 'node' in locals():
            node.g_bExit = True
            
            if hasattr(node, 'cam_thread'):
                node.cam_thread.join(timeout=1.0) 
            if hasattr(node, 'ctrl_thread'):
                node.ctrl_thread.join(timeout=1.0)

            if hasattr(node, 'cam'):
                ret = node.cam.MV_CC_StopGrabbing()
                if ret != 0:
                    rospy.loginfo("\033[31mStop grabbing fail! ret[0x%x]\033[0m" % ret)
                
                ret = node.cam.MV_CC_CloseDevice()
                if ret != 0:
                    rospy.loginfo("\033[31mClose device fail! ret[0x%x]\033[0m" % ret)
                
                ret = node.cam.MV_CC_DestroyHandle()
                if ret != 0:
                    rospy.loginfo("\033[31mDestroy handle fail! ret[0x%x]\033[0m" % ret)
            
            MvCamera.MV_CC_Finalize()
            
        rospy.loginfo("\033[32m[CamCtrl] Node shutdown completed.\033[0m")
