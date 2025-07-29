import rospy
from nav_msgs.msg import Odometry
from rospy import Time

class HILNode:
    def __init__(self):
        rospy.init_node('HIL_node')
        self.odom_sub = rospy.Subscriber('/ekf/ekf_odom', Odometry, self.odom_cb_ofb, queue_size=1, tcp_nodelay=True)
        
        # 初始化receive_odom和last_odom_time
        self.receive_odom = False
        self.last_odom_time = None
        self.last_receive_odom_time = None

    def odom_cb_ofb(self, msg):  # offboard inference callback
        if self.last_odom_time is None:
            self.last_odom_time = msg.header.stamp  # 初始化第一次接收的时间
            self.last_receive_odom_time = rospy.Time.now()
            return

        # 计算当前时间与上一次时间的间隔
        else:
            time_diff = (msg.header.stamp - self.last_odom_time).to_sec()
            time_receive_diff = (rospy.Time.now() - self.last_receive_odom_time).to_sec()

        # if time_diff > 0.010:
        print(f"[px4ctrl] <NIL node> "f"The last odom is sent {1000 * time_diff:.4f}ms ago.")
        print(f"[px4ctrl] <NIL node> "f"The last odom is received {1000 * time_receive_diff:.4f}ms ago.")
        
        # 更新最后一次接收到消息的时间
        self.last_odom_time = msg.header.stamp
        self.last_receive_odom_time = rospy.Time.now()

if __name__ == '__main__':
    hil_node = HILNode()
    rospy.spin()
