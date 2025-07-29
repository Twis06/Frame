#include <estimator/ekf_interface.h>

sensor_msgs::Imu newest_imu;
void imuCallaback(const sensor_msgs::ImuConstPtr& imu_msg, common::Visualizer &visualizer, estimator::EKFFusion &se3_fusion) { 
  common::IMU imu;
  imu.timestamp_ = imu_msg->header.stamp.toSec();
  imu.gyro_ = Vec3d(imu_msg->angular_velocity.x, imu_msg->angular_velocity.y, imu_msg->angular_velocity.z);
  imu.acce_ = Vec3d(imu_msg->linear_acceleration.x, imu_msg->linear_acceleration.y, imu_msg->linear_acceleration.z);
  newest_imu = *imu_msg;
  bool status = se3_fusion.predict(imu);
  if (!status) {
    return;
  }

  auto state = se3_fusion.getNominalState();
  // visualizer.publishGinsOdom(state.timestamp_, state.GetSE3(), state.v_);
  // visualizer.publishGinsPath(state.timestamp_, state.GetSE3());
}

void poseCallback(const nav_msgs::OdometryConstPtr& odom, common::Visualizer &visualizer, estimator::EKFFusion &se3_fusion) {
  // Convert the pose to nav state
  common::NavStated nav_state;
  nav_state.timestamp_ = odom->header.stamp.toSec();
  nav_state.R_ = SO3(Eigen::Quaterniond(odom->pose.pose.orientation.w, odom->pose.pose.orientation.x,
                                        odom->pose.pose.orientation.y, odom->pose.pose.orientation.z));
  nav_state.p_ = Vec3d(odom->pose.pose.position.x, odom->pose.pose.position.y, odom->pose.pose.position.z);
  nav_state.v_ = Vec3d(odom->twist.twist.linear.x, odom->twist.twist.linear.y, odom->twist.twist.linear.z);

  // static bool first_pose_set = false;
  // if (!first_pose_set) {
  //   se3_fusion.observeSE3(nav_state);
  //   first_pose_set = true;
  //   return;
  // }
  // return;

  bool status = se3_fusion.observeSE3(nav_state);
  if (!status) {
    return;
  }
}

void velCallback(const geometry_msgs::Vector3StampedConstPtr& vel_w, common::Visualizer &visualizer, estimator::EKFFusion &se3_fusion) {
  // Convert the pose to nav state
  Eigen::Vector3d vel_world(vel_w->vector.x, vel_w->vector.y, vel_w->vector.z);
  bool status = se3_fusion.observeSpeedWorld(vel_world);
  if (!status) {
    return;
  }
}

Eigen::Vector2d last_optical_flow_gyro = Eigen::Vector2d::Zero();

void opticalFlowCallback(const mavros_msgs::OpticalFlowRadConstPtr& optical_flow_msg, common::Visualizer &visualizer, estimator::EKFFusion &se3_fusion) {
  double kFlowQualityMinValue = 40.0;
  Eigen::Vector3d vel_b = Eigen::Vector3d::Zero();

  double integrated_x = optical_flow_msg->integrated_x;
  double integrated_y = optical_flow_msg->integrated_y;
  double distance = optical_flow_msg->distance;
  int quality = static_cast<int>(optical_flow_msg->quality);
  double gyro_x = optical_flow_msg->integrated_xgyro;
  double gyro_y = optical_flow_msg->integrated_ygyro;
  double dt = optical_flow_msg->integration_time_us * 0.000001; 

  if (quality > kFlowQualityMinValue) {
    double fx = (integrated_x - last_optical_flow_gyro.x()) * 50.; // optical flow ang vel. rad/s 
    double fy = (integrated_y - last_optical_flow_gyro.y()) * 50.; // optical flow ang vel. rad/s
    double speed_x = distance * fx;
    double speed_y = distance * fy;
    vel_b = Eigen::Vector3d(-speed_y, speed_x, 0.0);
  }

  static double last_height = -1;
  Eigen::Quaterniond q(newest_imu.orientation.w, newest_imu.orientation.x, newest_imu.orientation.y, newest_imu.orientation.z);
  double height = 0.0;
  if (optical_flow_msg->distance > 0.) {
    height = distance * q.toRotationMatrix()(2, 2);

    if (last_height > 0.0) {
      double z_vel_w = (height - last_height) * 50;
      // vel_b.z() = q.toRotationMatrix().transpose()(2, 2) * z_vel_body; // optical flow height velocity
      vel_b.z() = z_vel_w; // optical flow height velocity
    }
  }
  last_height = height;
  visualizer.publishOpticalFlowSpeed(optical_flow_msg->header.stamp.toSec(), vel_b);

  if (!se3_fusion.getFirstPoseSet()) {
    Eigen::Quaterniond Q_init;
    {
      Eigen::Matrix3d R0 = q.toRotationMatrix();
      double yaw  = common::RotationUtility::R2ypr(R0).x();
      R0 = common::RotationUtility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
      Q_init = R0;
      Q_init.normalize();
    }
    se3_fusion.setInitPose(SE3(Q_init, Vec3d(0, 0, 0)));
    LOG(INFO) << GREEN << "Mavros Init Rotation(ypr): " << common::RotationUtility::R2ypr(q.toRotationMatrix()).transpose() << TAIL;
    LOG(INFO) << GREEN << "Set Init Rotation(ypr): " << common::RotationUtility::R2ypr(Q_init.toRotationMatrix()).transpose() << TAIL;
    return;
  }

  se3_fusion.observeSpeedBodyXY(vel_b.head<2>());
  se3_fusion.observeSpeedWorld(vel_b.z());
  auto state = se3_fusion.getNominalState();
  visualizer.publishGinsOdom(state.timestamp_, state.GetSE3(), state.v_);
  visualizer.publishGinsPath(state.timestamp_, state.GetSE3());

  last_optical_flow_gyro.x() = gyro_x;
  last_optical_flow_gyro.y() = gyro_y;
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "ekf_fusion_node");
  google::InitGoogleLogging(argv[0]);
  FLAGS_stderrthreshold = google::INFO;
  FLAGS_colorlogtostderr = true;
  google::ParseCommandLineFlags(&argc, &argv, true);

  ros::NodeHandle nh("~");
  common::Visualizer visualizer;
  estimator::EKFFusion ekf_fusion;

  estimator::EKFFusion::Options options;
  options.lvio_pos_noise_   = common::param<double>(nh, "odom_pos_noise", 0.1);                              // Position noise variance
  options.lvio_ang_noise_   = common::param<double>(nh, "odom_angle_noise", 1);// * common::math::kDEG2RAD;  // Angular noise variance
  options.update_bias_acce_ = common::param<bool>(  nh, "update_bias_acc", true);
  options.update_bias_gyro_ = common::param<bool>(  nh, "update_bias_gyr", true);
  options.acce_var_         = common::param<double>(nh, "imu_acc_var", 0.2);         // Accelerometer noise variance
  options.gyro_var_         = common::param<double>(nh, "imu_gyro_var", 0.05);        // Gyroscope noise variance
  options.bias_acce_var_    = common::param<double>(nh, "imu_acc_bias_var", 0.001);  // Accelerometer bias noise variance
  options.bias_gyro_var_    = common::param<double>(nh, "imu_gyro_bias_var", 0.001);  // Gyroscope bias noise variance
  options.g_scale_          = common::param<double>(nh, "g_scale", 9.81);         // Gravity scale

  options.vel_xy_noise_     = common::param<double>(nh, "optical_flow_vel_xy_noise", 0.1);     // Velocity noise variance
  options.vel_z_noise_      = common::param<double>(nh, "optical_flow_vel_z_noise", 0.1);      // Velocity noise variance
  // options.R_odom_body_ << 0.965926, 0, 0.258819, 0, 1, 0, -0.258819, 0, 0.965926;

  options.pos_priori_cov = common::param<double>(nh, "pos_priori_cov",  1);
  options.vel_priori_cov = common::param<double>(nh, "vel_priori_cov",  1);
  options.rot_priori_cov = common::param<double>(nh, "rot_priori_cov",  1);
  options.bg_bias_priori_cov = common::param<double>(nh, "bg_bias_priori_cov",  1);
  options.ba_bias_priori_cov = common::param<double>(nh, "ba_bias_priori_cov",  1);


  Eigen::Vector3d init_ba;        // Initial accelerometer bias
  Eigen::Vector3d init_bg;        // Initial accelerometer bias

  init_ba.setZero();
  init_bg.setZero();

  std::string odom_topic = common::param<std::string>(nh, "odom_topic", "/imu");
  std::string imu_topic  = common::param<std::string>(nh, "imu_topic", "/bodyodom");  
  std::string vel_topic  = common::param<std::string>(nh, "vel_topic", "/vel_w");
  std::string flow_topic = common::param<std::string>(nh, "optical_flow_topic", "/optical_flow_rad");
  
  ekf_fusion.setInitOptions(options);
 
  ros::Subscriber sub_imu = nh.subscribe<sensor_msgs::Imu>(
      imu_topic, 1000, boost::bind(imuCallaback, _1, boost::ref(visualizer), boost::ref(ekf_fusion)),
      nullptr, ros::TransportHints().tcpNoDelay());

  ros::Subscriber sub_pose = nh.subscribe<nav_msgs::Odometry>(
      odom_topic, 1000, boost::bind(&poseCallback, _1, boost::ref(visualizer), boost::ref(ekf_fusion)),
      nullptr, ros::TransportHints().tcpNoDelay());

  ros::Subscriber sub_velw = nh.subscribe<geometry_msgs::Vector3Stamped>(
      vel_topic, 1000, boost::bind(&velCallback, _1, boost::ref(visualizer), boost::ref(ekf_fusion)),
      nullptr, ros::TransportHints().tcpNoDelay());
  
  ros::Subscriber sub_optf = nh.subscribe<mavros_msgs::OpticalFlowRad>(
      flow_topic, 1000, boost::bind(&opticalFlowCallback, _1, boost::ref(visualizer), boost::ref(ekf_fusion)),
      nullptr, ros::TransportHints().tcpNoDelay());

  printFZ();
  LOG(INFO) << "Subscriber  IMU topic: " << imu_topic;
  LOG(INFO) << "Subscriber Pose topic: " << odom_topic;
  LOG(INFO) << "Subscriber  Vel topic: " << vel_topic;
  LOG(INFO) << "Subscriber Flow topic: " << flow_topic;
  LOG(INFO) << ros::this_node::getName() << " Started, waiting for data...";
  ros::spin();
  return 0;
}