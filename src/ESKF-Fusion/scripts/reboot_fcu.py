import rospy
from mavros_msgs.srv import CommandLong

rospy.init_node('reboot_fc')

# 等待服务端准备好
rospy.wait_for_service('/mavros/cmd/command')

try:
    cmd_service = rospy.ServiceProxy('/mavros/cmd/command', CommandLong)
    # command = 246 是 MAV_CMD_PREFLIGHT_REBOOT_SHUTDOWN
    # param1 = 1.0 表示重启飞控
    resp = cmd_service(
        broadcast=False,
        command=246,
        confirmation=0,
        param1=1.0,  # reboot autopilot
        param2=0.0,
        param3=0.0,
        param4=0.0,
        param5=0.0,
        param6=0.0,
        param7=0.0
    )
    print(f"Command result: {resp.success}")

except rospy.ServiceException as e:
    print(f"Service call failed: {e}")
