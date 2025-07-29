#pragma once

#include <estimator/ekf.h>
#include <mavros_msgs/OpticalFlowRad.h>
#include <common/timer_clock.h>
#include <queue>
#include <thread>

namespace estimator {

class EKFInterface {
public:
  static constexpr double kFlowQualityMinValue = 40.0;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  EKFInterface(ros::NodeHandle &nh);
  ~EKFInterface() = default;

  void readParams();

  void initSubscribe();

  void imuCallback(const sensor_msgs::ImuConstPtr& imu_msg);

  void poseCallback(const nav_msgs::OdometryConstPtr& odom_msg);

  void velCallback(const geometry_msgs::Vector3StampedConstPtr& vel_msg);

  void opticalFlowCallback(const mavros_msgs::OpticalFlowRadConstPtr& optical_flow_msg);

  bool getNewestIMU(double newest_time, common::IMU &imu);

  void setEKFInitPose(Eigen::Quaterniond q, Eigen::Vector3d p);

private:
  ros::NodeHandle nh_;
  ros::Subscriber sub_imu_;
  ros::Subscriber sub_odom_;
  ros::Subscriber sub_vel_;
  ros::Subscriber sub_optical_flow_;

  std::queue<common::IMU> imu_queue_;
  std::mutex imu_queue_mutex_;

  common::Visualizer visualizer_;
  estimator::EKFFusion ekf_fusion_;
  estimator::EKFFusion::Options options_;

  Eigen::Vector2d compensate_optical_flow_gyro_ = Eigen::Vector2d::Zero();
  double last_height_ = -1;
  common::IMU newest_imu_;
};
} // namespace estimator