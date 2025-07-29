#include <common/visualizer.h>
#include <ros/package.h>
#include <common/param_helper.h>
#include <common/logging.h>

namespace common {

Visualizer::Visualizer() : nh_("~") {
  pub_fusion_odom_ = nh_.advertise<nav_msgs::Odometry>("fusion_odom", 10);
  pub_fusion_path_ = nh_.advertise<nav_msgs::Path>("fusion_path", 10);
  pub_optical_flow_speed_ = nh_.advertise<geometry_msgs::TwistStamped>("optical_flow_cal_speed", 10);

  // Init traj saver
  const std::string path = ros::package::getPath("eskf_fusion") + "/../../output/";
  const std::string traj_name = common::param<std::string>(nh_, "traj_name", "ekf-fusion.txt");
  traj_path_ = path + traj_name;
  FAR_INFO_STREAM("traj save path: " << traj_path_);
  save_traj_.open(traj_path_);
}

void Visualizer::publishGinsOdom(double time, const Sophus::SE3d &pose,
                                 const Vec3d &vel, const Eigen::MatrixXd &cov) {
  if (pub_fusion_odom_.getNumSubscribers() == 0) {
    return;
  }

  Eigen::Vector3d pos_b = /*pose.so3().inverse() **/ pose.translation();

  nav_msgs::Odometry odom;
  odom.header.stamp = ros::Time(time);
  odom.header.frame_id = "world";
  odom.pose.pose.position.x = pos_b.x();
  odom.pose.pose.position.y = pos_b.y();
  odom.pose.pose.position.z = pos_b.z();
  odom.pose.pose.orientation.w = pose.so3().unit_quaternion().w();
  odom.pose.pose.orientation.x = pose.so3().unit_quaternion().x();
  odom.pose.pose.orientation.y = pose.so3().unit_quaternion().y();
  odom.pose.pose.orientation.z = pose.so3().unit_quaternion().z();

  Eigen::Vector3d vel_b = pose.so3().inverse() * vel;
  Eigen::Vector3d vel_w = vel;
  
  odom.twist.twist.linear.x = vel_w.x();
  odom.twist.twist.linear.y = vel_w.y();
  odom.twist.twist.linear.z = vel_w.z();

  odom.twist.twist.angular.x = vel_b.x();
  odom.twist.twist.angular.y = vel_b.y();
  odom.twist.twist.angular.z = vel_b.z();

  pub_fusion_odom_.publish(odom);
  // if (cov.size() > 0) {
  //   Eigen::MatrixXd cov_6x6 = cov.block<6, 6>(0, 0);
  //   Eigen::MatrixXd cov_3x3 = cov.block<3, 3>(0, 0);
  //   Eigen::MatrixXd cov_3x3_inv = cov_3x3.inverse();
  //   odom.pose.covariance[0] = cov_6x6(0, 0);
  //   odom.pose.covariance[1] = cov_6x6(0, 1);
  //   odom.pose.covariance[2] = cov_6x6(0, 2);
  //   odom.pose.covariance[3] = cov_6x6(1, 0);
  //   odom.pose.covariance[4] = cov_6x6(1, 1);
  //   odom.pose.covariance[5] = cov_6x6(1, 2);
  // }

  Eigen::Quaterniond q = pose.unit_quaternion();
  Eigen::Vector3d p = pose.translation();
  save_traj_ << std::fixed << time << " " << p[0] << " " << p[1]
              << " " << p[2] << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
}

void Visualizer::publishGinsPath(double time, const Sophus::SE3d &pose) {
  if (pub_fusion_path_.getNumSubscribers() == 0) {
    return;
  }
  geometry_msgs::PoseStamped pose_stamped;
  pose_stamped.header.stamp = ros::Time(time);
  pose_stamped.header.frame_id = "world";
  pose_stamped.pose.position.x = pose.translation().x();
  pose_stamped.pose.position.y = pose.translation().y();
  pose_stamped.pose.position.z = pose.translation().z();
  pose_stamped.pose.orientation.w = pose.so3().unit_quaternion().w();
  pose_stamped.pose.orientation.x = pose.so3().unit_quaternion().x();
  pose_stamped.pose.orientation.y = pose.so3().unit_quaternion().y();
  pose_stamped.pose.orientation.z = pose.so3().unit_quaternion().z();

  gins_path_.header.stamp = ros::Time(time);
  gins_path_.header.frame_id = "world";
  gins_path_.poses.push_back(pose_stamped);
  pub_fusion_path_.publish(gins_path_);
}

void Visualizer::publishOpticalFlowSpeed(double time, const Vec3d &vel_b) {
  if (pub_optical_flow_speed_.getNumSubscribers() == 0) {
    return;
  }
  geometry_msgs::TwistStamped twist;
  twist.header.stamp = ros::Time(time);
  twist.header.frame_id = "world";
  twist.twist.linear.x = vel_b.x();
  twist.twist.linear.y = vel_b.y();
  twist.twist.linear.z = vel_b.z();

  pub_optical_flow_speed_.publish(twist);
}

}  // namespace common
