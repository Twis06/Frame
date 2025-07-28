#pragma once

#include <common/io_utils.h>
#include <common/math_utils.h>
#include <common/nav_state.h>
#include <common/param_helper.h>
#include <common/rotation_utils.h>
#include <common/visualizer.h>
#include <common/logging.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <sensor_msgs/Imu.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <boost/bind.hpp>
#include <fstream>
#include <iomanip>

namespace estimator {

class EKFFusion
{
public:
  static constexpr int kStateSize = 16;       // p,q,v,bg,ba
  static constexpr int kErrorStateSize = 15;
  static constexpr int kInputSize = 6;        // gyro, acce
  static constexpr int kPoseMeasureSize = 7;  // pos, q
  static constexpr int kVelMeasureSize = 3;   // vel
  static constexpr double kGravity = 9.81;    // gravity

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  struct Options {
    Options() = default;

    /// IMU measurement and zero-offset parameters
    double imu_dt_ = 0.01;  // IMU measurement interval
    // NOTE: The noise items of the IMU are all discrete-time and do not need to be multiplied by dt again.
    //  The IMU noise can be specified by the initializer
    double gyro_var_ = 1e-5;       // Gyroscope measures the standard deviation
    double acce_var_ = 1e-2;       // Add the measurement of standard deviation
    double bias_gyro_var_ = 1e-6;  // The zero deviation of the gyroscope travels the standard deviation
    double bias_acce_var_ = 1e-4;  // Add the standard deviation of the zero-biased walk

    /// Odometer parameters
    double odom_var_ = 0.5;
    double odom_span_ = 0.1;        // Odometer measurement interval
    double wheel_radius_ = 0.155;   // Wheel radius
    double circle_pulse_ = 1024.0;  // The number of pulses per revolution of the encoder

    /// RTK observation parameters
    double gnss_pos_noise_ = 0.1;                           // GNSS location noise
    double gnss_height_noise_ = 0.1;                        // GNSS high noise
    double gnss_ang_noise_ = 1.0 * common::math::kDEG2RAD;  // GNSS rotation noise

    /// Other configurations
    bool update_bias_gyro_ = true;  // Whether to update the gyroscope bias
    bool update_bias_acce_ = true;  // Whether to update the additional bias

    /// vio/lio pose noise
    double lvio_pos_noise_ = 0.05;
    double lvio_ang_noise_ = 0.5 * common::math::kDEG2RAD;  // Lvio rotation noise

    double g_scale_ = 1.0;  // gravity scale
    double vel_xy_noise_ = 0.1;  // velocity noise
    double vel_z_noise_ = 0.1;   // velocity noise
    Eigen::Matrix3d R_odom_body_ = Eigen::Matrix3d::Identity(); 

    double pos_priori_cov;
    double vel_priori_cov;
    double rot_priori_cov;
    double bg_bias_priori_cov;
    double ba_bias_priori_cov;  
  };

  Options& getOptions() { return options_; }
  const Options& getOptions() const { return options_; }

  void setInitOptions(const Options& options);

  void buildNoise();

  bool predict(const common::IMU &imu);

  bool observeSE3(const common::NavStated &pose);

  bool observeSpeedWorld(const Eigen::Vector3d &vel_w);

  bool observeSpeedWorld(const Eigen::Vector2d &vel_w_xy);

  bool observeSpeedBodyXY(const Eigen::Vector2d &vel_b_xy);

  bool observeSpeedWorld(const double &vel_w_z);

  bool updateState(const Eigen::VectorXd &dx);

  common::NavStated getNominalState() const {
    return x_state_;
  }

  void setInitPose(const SE3 &pose) {
    if (first_pose_set_)
      return;
    
    x_state_.R_ = pose.so3();
    x_state_.p_ = pose.translation();
    x_state_.q_ = pose.unit_quaternion();
    first_pose_set_ = true;
  }

  bool getFirstPoseSet() const {
    return first_pose_set_;
  }

private:
  Eigen::MatrixXd Qt;      // process noise matrix
  Eigen::MatrixXd Rt;      // pos/rot observation noise matrix
  Eigen::MatrixXd Rt_vel;  // vel observation noise matrix
  Eigen::MatrixXd Pt;      // covariance matrix

  common::NavStated x_state_;  // state vector

	Options options_;
  bool first_pose_set_ = false;     // is first pose options
  bool first_imu_flag_ = true;
  double current_time_ = 0.0;  // current timestamp
  double last_time_ = 0.0;     // last timestamp
};

}