"""
Copyright 2024 ZheJiang University and Shanghai Jiao Tong University
"""

import argparse
import cv2
import os
import numpy as np
import rosbag
import rospy
from sensor_msgs.msg import Image, Imu
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import time

image_size = [640, 480]

def check_topic_in_bag(bag_file, target_topic):
    try:
        with rosbag.Bag(bag_file, "r") as bag:
            # get all topcid name
            topics = bag.get_type_and_topic_info().topics
            if target_topic in topics:
                print(f"Topic '{target_topic}' exists in the bag.")
                return True
            else:
                print(f"Topic '{target_topic}' does not exist in the bag.")
                return False
    except Exception as e:
        print(f"Error opening bag file: {e}")
        return False


def main(bag_file, odom_topic, imu_topic, skip_time_second):
    rospy.init_node("rosbag_player")
    odom_pub = rospy.Publisher(odom_topic, Odometry, queue_size=100)
    imu_pub = rospy.Publisher(imu_topic, Imu, queue_size=100)
    bridge = CvBridge()

    rospy.logwarn(f"Skip bag start time {skip_time_second}")
    img = np.zeros((200, 600, 3), dtype=np.uint8)
    cv2.putText(img, "Press any key to publish next IMU", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    time.sleep(1)
    first_pose_get = False
    # open rosbag file
    start_time = None
    with rosbag.Bag(bag_file, "r") as bag:
        for topic, msg, t in bag.read_messages(topics=[odom_topic, imu_topic]):
            if start_time is None:
                start_time = t  # record start time
            # skip the first few seconds
            if (t - start_time).to_sec() < skip_time_second:
                continue
            if topic == odom_topic:
                first_pose_get = True
                # print current image time
                rospy.loginfo(f"Publishing ODOM at time {t.to_sec()}")

                odom_pub.publish(msg)

            elif topic == imu_topic:
                if first_pose_get == False:
                    continue
                # print current image time
                rospy.logwarn(f"Publishing IMU at time {t.to_sec()}")
                rospy.loginfo(f"IMU data: {msg.linear_acceleration.x}, {msg.linear_acceleration.y}, {msg.linear_acceleration.z}")

              
                cv2.imshow("Image", img)
                key = cv2.waitKey(0) & 0xFF
                if key == 27:  # ESC = 27
                    rospy.loginfo("ESC pressed. Exiting...")
                    break
                # Publish IMU data to ROS
                imu_pub.publish(msg)
                
            
            # control the playback speed
            rospy.sleep(0.001)

    # destroy all OpenCV windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Play ROS bag and publish images and IMU data."
    )
    parser.add_argument("bag_file", type=str, help="Path to the input ROS bag file.")
    parser.add_argument(
        "odom_topic",
        type=str,
        nargs="?",
        default="/Odometry",
        help="Left image topic name (default: /camera/left/image_raw).",
    )
    parser.add_argument(
        "imu_topic",
        type=str,
        nargs="?",
        default="/livox/imu",
        help="IMU topic name (default: /imu/data).",
    )
    parser.add_argument(
        "skip_start_second",
        type=float,
        nargs="?",
        default=0.0,
        help="Skip bag start time in seconds (default: 0.0).",
    )

    args = parser.parse_args()

    bag_file_path = args.bag_file
    if os.path.exists(bag_file_path):
        print(f"{bag_file_path} exists.")
    else:
        print(f"{bag_file_path} does not exist.")
        exit(0)

    if not check_topic_in_bag(
        args.bag_file, args.odom_topic
    ) or not check_topic_in_bag(args.bag_file, args.imu_topic):
        exit(0)
    
    main(args.bag_file, args.odom_topic, args.imu_topic, args.skip_start_second)
