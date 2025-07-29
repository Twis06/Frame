#pragma once

#include <common/eigen_types.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <geometry_msgs/TwistStamped.h>
#include <fstream>

namespace common {
class Visualizer {
public:
  Visualizer();

  ~Visualizer() = default;

  void publishGinsOdom(double time, const Sophus::SE3d &pose, const Vec3d &vel,
                       const Eigen::MatrixXd &cov = Eigen::Matrix3d::Zero());
  void publishGinsPath(double time, const Sophus::SE3d &pose);
  void publishOpticalFlowSpeed(double time, const Vec3d &vel_b);

private:
  ros::Publisher pub_fusion_odom_;
  ros::Publisher pub_fusion_path_;
  ros::Publisher pub_optical_flow_speed_;
  nav_msgs::Path gins_path_;
  ros::NodeHandle nh_;

  std::string traj_path_;
  std::ofstream save_traj_;
};

}  // namespace common