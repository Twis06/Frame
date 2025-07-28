#include <estimator/ekf_interface.h>

namespace estimator {

EKFInterface::EKFInterface(ros::NodeHandle &nh) : nh_(nh) {
  readParams();
  initSubscribe();

  printFZ();
  LOG(INFO) << ros::this_node::getName() << " Started, waiting for data...";
}

void EKFInterface::readParams() {
  options_.acce_var_          = common::param<double>(nh_, "imu_acc_var", 0.2);
  options_.gyro_var_          = common::param<double>(nh_, "imu_gyro_var", 0.02);
  options_.lvio_pos_noise_    = common::param<double>(nh_, "odom_pos_noise", 0.02);
  options_.lvio_ang_noise_    = common::param<double>(nh_, "odom_angle_noise", 0.01);
  options_.g_scale_           = common::param<double>(nh_, "g_scale", 9.81);
  options_.vel_xy_noise_      = common::param<double>(nh_, "optical_flow_vel_xy_noise", 0.1);
  options_.vel_z_noise_       = common::param<double>(nh_, "optical_flow_vel_z_noise", 0.1);
  options_.pos_priori_cov     = common::param<double>(nh_, "pos_priori_cov",  1);
  options_.vel_priori_cov     = common::param<double>(nh_, "vel_priori_cov",  1);
  options_.rot_priori_cov     = common::param<double>(nh_, "rot_priori_cov",  1);
  options_.bg_bias_priori_cov = common::param<double>(nh_, "bg_bias_priori_cov",  1);
  options_.ba_bias_priori_cov = common::param<double>(nh_, "ba_bias_priori_cov",  1);

  ekf_fusion_.setInitOptions(options_);
}

void EKFInterface::initSubscribe() {
  std::string odom_topic = common::param<std::string>(nh_, "odom_topic", "/imu");
  std::string imu_topic  = common::param<std::string>(nh_, "imu_topic", "/bodyodom");  
  std::string vel_topic  = common::param<std::string>(nh_, "vel_topic", "/vel_w");
  std::string flow_topic = common::param<std::string>(nh_, "optical_flow_topic", "/optical_flow_rad");
  
  sub_imu_ = nh_.subscribe<sensor_msgs::Imu>(
      imu_topic, 1000, &EKFInterface::imuCallback, 
      this, ros::TransportHints().tcpNoDelay());

  sub_odom_ = nh_.subscribe<nav_msgs::Odometry>(
      odom_topic, 1000, &EKFInterface::poseCallback, 
      this, ros::TransportHints().tcpNoDelay());

  sub_vel_ = nh_.subscribe<geometry_msgs::Vector3Stamped>(
      vel_topic, 1000, &EKFInterface::velCallback, 
      this, ros::TransportHints().tcpNoDelay());

  sub_optical_flow_ = nh_.subscribe<mavros_msgs::OpticalFlowRad>(
      flow_topic, 1000, &EKFInterface::opticalFlowCallback, 
      this, ros::TransportHints().tcpNoDelay());
  
  LOG(INFO) << "Subscriber  IMU topic: " << imu_topic;
  LOG(INFO) << "Subscriber Pose topic: " << odom_topic;
  LOG(INFO) << "Subscriber  Vel topic: " << vel_topic;
  LOG(INFO) << "Subscriber Flow topic: " << flow_topic;
}

void EKFInterface::imuCallback(const sensor_msgs::ImuConstPtr& imu_msg) {
  common::IMU imu;
  imu.timestamp_ = imu_msg->header.stamp.toSec();
  imu.gyro_ = Vec3d(imu_msg->angular_velocity.x, imu_msg->angular_velocity.y, imu_msg->angular_velocity.z);
  imu.acce_ = Vec3d(imu_msg->linear_acceleration.x, imu_msg->linear_acceleration.y, imu_msg->linear_acceleration.z);

  Eigen::Quaterniond q(imu_msg->orientation.w, imu_msg->orientation.x, imu_msg->orientation.y, imu_msg->orientation.z);
  imu.q_ = q.normalized();
  newest_imu_ = imu;
  // push IMU data to queue
  // {
  //   std::lock_guard<std::mutex> lock(imu_queue_mutex_);
  //   imu_queue_.push(imu);
  // }

  bool status = ekf_fusion_.predict(imu);
  if (!status) {
    return;
  }

  // auto state = ekf_fusion_.getNominalState();
  // visualizer_.publishGinsOdom(state.timestamp_, state.GetSE3(), state.v_);
  // visualizer_.publishGinsPath(state.timestamp_, state.GetSE3());
}

void EKFInterface::poseCallback(const nav_msgs::OdometryConstPtr& odom) {
  // Convert the pose to nav state
  common::NavStated nav_state;
  nav_state.timestamp_ = odom->header.stamp.toSec();
  nav_state.R_ = SO3(Eigen::Quaterniond(odom->pose.pose.orientation.w, odom->pose.pose.orientation.x,
                                        odom->pose.pose.orientation.y, odom->pose.pose.orientation.z));
  nav_state.p_ = Vec3d(odom->pose.pose.position.x, odom->pose.pose.position.y, odom->pose.pose.position.z);
  nav_state.v_ = Vec3d(odom->twist.twist.linear.x, odom->twist.twist.linear.y, odom->twist.twist.linear.z);

  // static bool first_pose_set = false;
  // if (!first_pose_set) {
  //   ekf_fusion_.observeSE3(nav_state);
  //   first_pose_set = true;
  //   return;
  // }
  // return;

  bool status = ekf_fusion_.observeSE3(nav_state);
  if (!status) {
    return;
  }
}

void EKFInterface::velCallback(const geometry_msgs::Vector3StampedConstPtr& vel_w) {
  // Convert the pose to nav state
  Eigen::Vector3d vel_world(vel_w->vector.x, vel_w->vector.y, vel_w->vector.z);
  bool status = ekf_fusion_.observeSpeedWorld(vel_world);
  if (!status) {
    return;
  }
}

void EKFInterface::opticalFlowCallback(const mavros_msgs::OpticalFlowRadConstPtr& optical_flow_msg) {
  // TODO(wh) check optical flow FPS
  // TODO(wh) wait imu time


  Eigen::Vector2d vel_b_xy = Eigen::Vector2d::Zero();
  double vel_w_z = 0.0;

  double optical_flow_time = optical_flow_msg->header.stamp.toSec();
  double integrated_x = optical_flow_msg->integrated_x;
  double integrated_y = optical_flow_msg->integrated_y;
  double distance = optical_flow_msg->distance;
  int quality = static_cast<int>(optical_flow_msg->quality);
  double gyro_x = optical_flow_msg->integrated_xgyro;
  double gyro_y = optical_flow_msg->integrated_ygyro;
  double dt = optical_flow_msg->integration_time_us * 0.000001;
  double fps = 1.0 / dt;

  // if(distance == -1)
  if(fabs(distance+1) < 1e-4)
  {
    return;
  }

  common::IMU imu = newest_imu_;
  // TimerClock timer;
  // while (!getNewestIMU(optical_flow_time, imu)) {
  //   std::this_thread::sleep_for(std::chrono::milliseconds(1));
  //   LOG(WARNING) << "Waiting for IMU data...";
  //   if (timer.end() > 10.) {
  //     LOG(ERROR) << "IMU data is too old, abandon this observation";
  //     return;
  //   }
  // }

  if (!ekf_fusion_.getFirstPoseSet()) {
    setEKFInitPose(imu.q_, Eigen::Vector3d(0, 0, 0));
  }

  if (quality > kFlowQualityMinValue) {
    double fx = (integrated_x - compensate_optical_flow_gyro_.x()) * fps;
    double fy = (integrated_y - compensate_optical_flow_gyro_.y()) * fps;
    double speed_x = distance * fx;
    double speed_y = distance * fy;
    vel_b_xy = Eigen::Vector2d(-speed_y, speed_x);
    ekf_fusion_.observeSpeedBodyXY(vel_b_xy);
  } else {
    LOG(WARNING) << "Optical flow quality is too low: " << quality << ", Abandon this observation";
    return;
  }

  double height = 0.0;
  if (optical_flow_msg->distance > 0.) {
    height = distance * imu.q_.toRotationMatrix()(2, 2);

    if (last_height_ > 0.0) {
      double z_vel_w = (height - last_height_) * fps;
      vel_w_z = z_vel_w;  // optical flow height velocity
      ekf_fusion_.observeSpeedWorld(vel_w_z);
    }
  }

  last_height_ = height;

  compensate_optical_flow_gyro_.x() = gyro_x;
  compensate_optical_flow_gyro_.y() = gyro_y;

  auto state = ekf_fusion_.getNominalState();
  visualizer_.publishGinsOdom(state.timestamp_, state.GetSE3(), state.v_);
  visualizer_.publishGinsPath(state.timestamp_, state.GetSE3());
}

bool EKFInterface::getNewestIMU(double newest_time, common::IMU &imu) {
  std::lock_guard<std::mutex> lock(imu_queue_mutex_);

  while (imu_queue_.size() >= 1) {
    common::IMU imu_temp = imu_queue_.front();
    if (imu_temp.timestamp_ < newest_time) {
      imu_queue_.pop();
    } else {
      break;
    }
  }

  if (imu_queue_.empty()) {
    return false;
  }
  
  imu = imu_queue_.front();
  imu_queue_.pop();

  if (std::abs(imu.timestamp_ - newest_time) > 0.05) {
    LOG(WARNING) << "IMU data is too old: " << imu.timestamp_ - newest_time;
    return false;
  }

  return true;
}

void EKFInterface::setEKFInitPose(Eigen::Quaterniond q, Eigen::Vector3d p) {
  Eigen::Quaterniond Q_init;
  {
    Eigen::Matrix3d R0 = q.toRotationMatrix();
    double yaw = common::RotationUtility::R2ypr(R0).x();
    R0 = common::RotationUtility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    Q_init = R0;
    Q_init.normalize();
  }
  ekf_fusion_.setInitPose(SE3(Q_init, p));
  LOG(INFO) << GREEN << "Mavros Init Rotation(ypr): "
            << common::RotationUtility::R2ypr(q.toRotationMatrix()).transpose() << TAIL;
  LOG(INFO) << GREEN << "Set Init Rotation(ypr): "
            << common::RotationUtility::R2ypr(Q_init.toRotationMatrix()).transpose() << TAIL;
  return;
}

}  // namespace estimator