#include <common/io_utils.h>
#include <common/visualizer.h>
#include <estimator/se3_fusion.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>
#include <fstream>
#include <iomanip>
#include <boost/bind.hpp>
#include <common/param_helper.h>
#include <common/rotation_utils.h>

void imuCallaback(const sensor_msgs::ImuConstPtr& imu_msg, common::Visualizer &visualizer, estimator::SE3Fusion &se3_fusion) { 
  common::IMU imu;
  imu.timestamp_ = imu_msg->header.stamp.toSec();
  imu.gyro_ = Vec3d(imu_msg->angular_velocity.x, imu_msg->angular_velocity.y, imu_msg->angular_velocity.z);
  Eigen::Vector3d acc = Vec3d(imu_msg->linear_acceleration.x, imu_msg->linear_acceleration.y, imu_msg->linear_acceleration.z);
  imu.acce_ = acc; 
  bool status = se3_fusion.imuCallback(imu);
  if (!status) {
    return;
  }

  auto state = se3_fusion.GetNominalState();
  visualizer.publishGinsOdom(state.timestamp_, state.GetSE3(), state.v_);
  visualizer.publishGinsPath(state.timestamp_, state.GetSE3());
  // std::cout << "normal state " << state.timestamp_ << " " << state.GetSE3().translation().transpose() << std::endl;
}

void poseCallback(const nav_msgs::OdometryConstPtr& odom, common::Visualizer &visualizer, estimator::SE3Fusion &se3_fusion) {
  // Convert the pose to nav state
  common::NavStated nav_state;
  nav_state.timestamp_ = odom->header.stamp.toSec();
  nav_state.R_ = SO3(Eigen::Quaterniond(odom->pose.pose.orientation.w, odom->pose.pose.orientation.x,
    odom->pose.pose.orientation.y, odom->pose.pose.orientation.z));
  nav_state.p_ = Vec3d(odom->pose.pose.position.x, odom->pose.pose.position.y, odom->pose.pose.position.z);
  nav_state.v_ = Vec3d(odom->twist.twist.linear.x, odom->twist.twist.linear.y, odom->twist.twist.linear.z);

  // static bool first_gnss_set = false;
  // static Eigen::Vector3d origin = Eigen::Vector3d::Zero();
  // if (!first_gnss_set) {
  //   origin = nav_state.p_;
  //   first_gnss_set = true;
  //   std::cout << "Orign " << origin.transpose() << std::endl;
  // }
  // nav_state.p_ -= origin;

  auto state0 = se3_fusion.GetNominalState();
  bool status = se3_fusion.poseCallback(nav_state);
  if (!status) {
    return;
  }

  auto state1 = se3_fusion.GetNominalState();
  // visualizer.publishGinsPath(state.timestamp_, state.GetSE3()); 
  static int index = 0;
  
  Eigen::Vector3d pose_diff = state1.GetSE3().translation() - state0.GetSE3().translation();
  std::cout << "[Update] update pose " << pose_diff.norm() 
            << ", ypr:" << common::R2ypr(state1.R_.matrix()).transpose() 
            << ", ba: " << state1.ba_.transpose() 
            << ", bg: " << state1.bg_.transpose() << std::endl;
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "lio_fusion_node");
  google::InitGoogleLogging(argv[0]);
  FLAGS_stderrthreshold = google::INFO;
  FLAGS_colorlogtostderr = true;
  google::ParseCommandLineFlags(&argc, &argv, true);

  ros::NodeHandle nh("~");
  common::Visualizer visualizer;
  estimator::SE3Fusion se3_fusion;

  estimator::ESKFD::Options options;
  options.lvio_pos_noise_   = common::param<double>(nh, "lvio_pos_noise", 0.1);                           // Position noise variance
  options.lvio_ang_noise_   = common::param<double>(nh, "lvio_angle_noise", 1) * common::math::kDEG2RAD;  // Angular noise variance
  options.update_bias_acce_ = common::param<bool>(  nh, "update_bias_acc", true);
  options.update_bias_gyro_ = common::param<bool>(  nh, "update_bias_gyr", true);
  options.acce_var_         = common::param<double>(nh, "acce_var", 0.2);         // Accelerometer noise variance
  options.gyro_var_         = common::param<double>(nh, "gyro_var", 0.05);        // Gyroscope noise variance
  options.bias_acce_var_    = common::param<double>(nh, "bias_acce_var", 0.001);  // Accelerometer bias noise variance
  options.bias_gyro_var_    = common::param<double>(nh, "bias_gyro_var", 0.001);  // Gyroscope bias noise variance

  Eigen::Vector3d init_ba(-0.133999, 0.146775, 0.0109571);        // Initial accelerometer bias
  Eigen::Vector3d init_bg(0.00169356, 2.92211e-05, 0.000259947);  // Initial accelerometer bias

  init_ba.setZero();
  init_bg.setZero();

  Eigen::Vector3d g(0, 0, 0);
  g.z() = -common::param<double>(nh, "g_norm", 9.81);  // Gravity vector
  se3_fusion.getESKF().SetInitialConditions(options, init_bg, init_ba, g);

  std::string odom_topic = common::param<std::string>(nh, "odom_topic", "/imu");
  std::string imu_topic  = common::param<std::string>(nh, "imu_topic", "/bodyodom");  
  
  ros::Subscriber sub_imu = nh.subscribe<sensor_msgs::Imu>(
      imu_topic, 1000, boost::bind(imuCallaback, _1, boost::ref(visualizer), boost::ref(se3_fusion)),
      nullptr, ros::TransportHints().tcpNoDelay());

  ros::Subscriber sub_pose = nh.subscribe<nav_msgs::Odometry>(
      odom_topic, 1000, boost::bind(&poseCallback, _1, boost::ref(visualizer), boost::ref(se3_fusion)),
      nullptr, ros::TransportHints().tcpNoDelay());

  ros::spin();
  return 0;
}