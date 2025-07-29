# -- coding: utf-8 --

import sys
import threading
import os
import termios
import time
import numpy as np
import cv2

from ctypes import *

# # ROS
# import rospy
# from std_msgs.msg import String
# from std_msgs.msg import UInt32

sys.path.append("/opt/MVS/Samples/64/Python/MvImport")
from MvCameraControl_class import *

g_bExit = False
interval = 1/60

# # ROS
# def img_time_callback(msg):
#     latest_img_time = rospy.Time.now()
#     print("Cam Trigger Time[%f]" % (latest_img_time.to_sec()))

# 为线程定义一个函数
def work_thread(cam=0, pData=0, nDataSize=0):
    stOutFrame = MV_FRAME_OUT()
    memset(byref(stOutFrame), 0, sizeof(stOutFrame))

    while True:
        ret = cam.MV_CC_SetCommandValue("TriggerSoftware")
        time_trigger = time.time()
        
        if ret != 0:
            print ("failed in TriggerSoftware")
            
        ret = cam.MV_CC_GetImageBuffer(stOutFrame, 1000)

        if ret == 0:
            print ("get one frame: Width[%d], Height[%d], PixelType[0x%x], nFrameNum[%d]"  % (stOutFrame.stFrameInfo.nWidth, stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.enPixelType, stOutFrame.stFrameInfo.nFrameNum)) 

            nSize = stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight
            img_buff = (c_ubyte * nSize)()
            memmove(byref(img_buff), stOutFrame.pBufAddr, nSize)

            img = np.asarray(img_buff) 
            img = img.reshape(stOutFrame.stFrameInfo.nHeight , stOutFrame.stFrameInfo.nWidth, -1)

            time_getframe = time.time()
            print ("get time", time_getframe - time_trigger)
            
            # cv2.imwrite('img_', img)
            # print(img)

            _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
            img = cv2.resize(img, (320, 256), cv2.INTER_AREA)
            _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

            # cv2.imwrite('img_%d.jpg' % stOutFrame.stFrameInfo.nFrameNum, img)

            img = img / 255.0

            # process_time = time.time() - time_getframe
            # print ("process time", process_time)

            # cv2.imshow('img_.jpg', img)
            # cv2.waitKey(1)
            end_time = time.time()

            process_time = end_time - time_getframe
            print ("process time", process_time)

            total_time = end_time - time_trigger
            print ("total time", total_time)

            sleep_time = max(0, interval - total_time)  # 确保不会出现负值

            # print ("sleep time", sleep_time)
            time.sleep(sleep_time)  # 等待剩余时间

            next_trigger = time.time()
            print ("trigger time", next_trigger - time_trigger)

            cam.MV_CC_FreeImageBuffer(stOutFrame)
        else:
            print ("no data[0x%x]" % ret)

        if g_bExit == True:
            break
        

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
    
if __name__ == "__main__":
    
    # # ROS
    # rospy.init_node('camera_capture', anonymous=True)

    # ch:初始化SDK | en: initialize SDK
    MvCamera.MV_CC_Initialize()

    SDKVersion = MvCamera.MV_CC_GetSDKVersion()
    print ("SDKVersion[0x%x]" % SDKVersion)

    deviceList = MV_CC_DEVICE_INFO_LIST()
    tlayerType = (MV_GIGE_DEVICE | MV_USB_DEVICE | MV_GENTL_CAMERALINK_DEVICE | MV_GENTL_CXP_DEVICE | MV_GENTL_XOF_DEVICE)
    
    # ch:枚举设备 | en:Enum device
    ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
    if ret != 0:
        print ("enum devices fail! ret[0x%x]" % ret)
        sys.exit()

    if deviceList.nDeviceNum == 0:
        print ("find no device!")
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
        nConnectionNum = input("please input the number of the device to connect:")
    else:
        nConnectionNum = raw_input("please input the number of the device to connect:")

    if int(nConnectionNum) >= deviceList.nDeviceNum:
        print ("intput error!")
        sys.exit()

    # rospy.Subscriber("/fast_sync/stereo/img_time", UInt32, img_time_callback)

    # ch:创建相机实例 | en:Creat Camera Object
    cam = MvCamera()
    
    # ch:选择设备并创建句柄| en:Select device and create handle
    stDeviceList = cast(deviceList.pDeviceInfo[int(nConnectionNum)], POINTER(MV_CC_DEVICE_INFO)).contents

    ret = cam.MV_CC_CreateHandle(stDeviceList)
    if ret != 0:
        print ("create handle fail! ret[0x%x]" % ret)
        sys.exit()

    # ch:打开设备 | en:Open device
    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if ret != 0:
        print ("open device fail! ret[0x%x]" % ret)
        sys.exit()
    
    ret = cam.MV_CC_FeatureSave('config')

    ret = cam.MV_CC_FeatureLoad('camera_config')
    if ret != 0:
        print ("set camera config fail! ret[0x%x]" % ret)
        sys.exit()
    
    # ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
    if stDeviceList.nTLayerType == MV_GIGE_DEVICE or stDeviceList.nTLayerType == MV_GENTL_GIGE_DEVICE:
        nPacketSize = cam.MV_CC_GetOptimalPacketSize()
        if int(nPacketSize) > 0:
            ret = cam.MV_CC_SetIntValue("GevSCPSPacketSize",nPacketSize)
            if ret != 0:
                print ("Warning: Set Packet Size fail! ret[0x%x]" % ret)
        else:
            print ("Warning: Get Packet Size fail! ret[0x%x]" % nPacketSize)

    # ret = cam.MV_CC_FeatureSave('config')
    #cam.MV_CC_FeatureLoad('camera_config')
    
    # ch:获取数据包大小 | en:Get payload size
    stParam =  MVCC_INTVALUE()
    memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))
    
    ret = cam.MV_CC_GetIntValue("PayloadSize", stParam)
    if ret != 0:
        print ("get payload size fail! ret[0x%x]" % ret)
        sys.exit()
    nPayloadSize = stParam.nCurValue

    # ret = cam.MV_CC_SetBayerCvtQuality(1)
    # if ret != 0:
    #     print ("set Bayer convert quality fail! ret[0x%x]" % ret)
    
    # ch:开始取流 | en:Start grab image
    ret = cam.MV_CC_StartGrabbing()
    if ret != 0:
        print ("start grabbing fail! ret[0x%x]" % ret)
        sys.exit()

    data_buf = (c_ubyte * nPayloadSize)()

    try:
        hThreadHandle = threading.Thread(target=work_thread, args=(cam, byref(data_buf), nPayloadSize))
        hThreadHandle.start()
    except:
        print ("error: unable to start thread")
        
    print ("press a key to stop grabbing.")
    press_any_key_exit()

    g_bExit = True
    hThreadHandle.join()

    # ch:停止取流 | en:Stop grab image
    ret = cam.MV_CC_StopGrabbing()
    if ret != 0:
        print ("stop grabbing fail! ret[0x%x]" % ret)
        del data_buf
        sys.exit()

    # ch:关闭设备 | Close device
    ret = cam.MV_CC_CloseDevice()
    if ret != 0:
        print ("close deivce fail! ret[0x%x]" % ret)
        del data_buf
        sys.exit()

    # ch:销毁句柄 | Destroy handle
    ret = cam.MV_CC_DestroyHandle()
    if ret != 0:
        print ("destroy handle fail! ret[0x%x]" % ret)
        del data_buf
        sys.exit()

    del data_buf

    # ch:反初始化SDK | en: finalize SDK
    MvCamera.MV_CC_Finalize()
