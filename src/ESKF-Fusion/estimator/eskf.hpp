#pragma once

#include <glog/logging.h>

#include <iomanip>

#include <common/eigen_types.h>
#include <common/gnss.h>
#include <common/imu.h>
#include <common/math_utils.h>
#include <common/nav_state.h>
#include <common/odom.h>

namespace estimator {

/*
The Error State Kalman filter
The readings for observing GNSS can be specified, and GNSS should be converted to the vehicle coordinate system in advance
This ESKF uses 18-dimensional ESKF. The scalar type can be specified by S, and the default is double
Variable order: p, v, R, bg, ba, gravity
* @tparam S The precision of the state variable, taking float or double
*/
template <typename S = double>
class ESKF {
 public:
  
  using SO3 = Sophus::SO3<S>;                     // Rotate variable type
  using VecT = Eigen::Matrix<S, 3, 1>;            // Vector type
  using Vec18T = Eigen::Matrix<S, 18, 1>;         // 18-dimensional vector type
  using Mat3T = Eigen::Matrix<S, 3, 3>;           // 3x3 matrix type
  using MotionNoiseT = Eigen::Matrix<S, 18, 18>;  // Types of motion noise
  using OdomNoiseT = Eigen::Matrix<S, 3, 3>;      // Odometer noise type
  using GnssNoiseT = Eigen::Matrix<S, 6, 6>;      // GNSS noise type
  using Mat18T = Eigen::Matrix<S, 18, 18>;        // 18-dimensional variance type
  using NavStateT = common::NavState<S>;          // Overall nominal state variable type

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

    double lvio_pos_noise_ = 0.05;
    double lvio_ang_noise_ = 0.5 * common::math::kDEG2RAD;  // Lvio rotation noise
  };

  /// @brief The initial zero is biased to zero
  ESKF(Options option = Options()) : options_(option) { BuildNoise(option); }

  /// @brief Set the initial conditions
  /// @param options Noise item configuration
  /// @param init_bg Initial zero-offset gyroscope
  /// @param init_ba initial zero partial addition
  /// @param gravity
  void SetInitialConditions(Options options, const VecT& init_bg,
                            const VecT& init_ba,
                            const VecT& gravity = VecT(0, 0, -9.8)) {
    BuildNoise(options);
    options_ = options;
    bg_ = init_bg;
    ba_ = init_ba;
    g_ = gravity;
    cov_ = Mat18T::Identity() * 1e-4;
  }

  Options GetOptions() const { return options_; }

  /// Use IMU recursion
  bool Predict(const common::IMU& imu);

  /// Observe using a wheel speedometer
  bool ObserveWheelSpeed(const common::Odom& odom);

  /// Observation using GPS
  bool ObserveGps(const common::GNSS& gnss);

  /// @brief Observe using SE3
  /// @param pose Observe the pose
  /// @param trans_noise translation noise
  /// @param ang_noise angular noise
  bool ObserveSE3(const SE3& pose, double trans_noise = 0.1,
                  double ang_noise = 1.0 * common::math::kDEG2RAD);

  /// @brief Obtain the full state
  NavStateT GetNominalState() const {
    return NavStateT(current_time_, R_, p_, v_, bg_, ba_);
  }

  /// @brief abtain the SE3 status
  SE3 GetNominalSE3() const { return SE3(R_, p_); }

  /// @brief Set status X
  void SetX(const common::NavStated& x, const Vec3d& grav) {
    current_time_ = x.timestamp_;
    R_ = x.R_;
    p_ = x.p_;
    v_ = x.v_;
    bg_ = x.bg_;
    ba_ = x.ba_;
    g_ = grav;
  }

  void SetX(const SE3& x, double time) {
    current_time_ = time;
    R_ = x.so3();
    p_ = x.translation();
  }

  /// @brief Set covariance
  void SetCov(const Mat18T& cov) { cov_ = cov; }

  /// @brief Obtain gravity
  Vec3d GetGravity() const { return g_; }

 private:
  void BuildNoise(const Options& options) {
    double ev = options.acce_var_;
    double et = options.gyro_var_;
    double eg = options.bias_gyro_var_;
    double ea = options.bias_acce_var_;

    double ev2 = ev;  // * ev;
    double et2 = et;  // * et;
    double eg2 = eg;  // * eg;
    double ea2 = ea;  // * ea;

    // Set process noise
    Q_.diagonal() << 0, 0, 0, ev2, ev2, ev2, et2, et2, et2, eg2, eg2, eg2, ea2,
        ea2, ea2, 0, 0, 0;

    // Set the odometer noise
    double o2 = options_.odom_var_ * options_.odom_var_;
    odom_noise_.diagonal() << o2, o2, o2;

    // Set the GNSS status
    double gp2 = options.gnss_pos_noise_ * options.gnss_pos_noise_;
    double gh2 = options.gnss_height_noise_ * options.gnss_height_noise_;
    double ga2 = options.gnss_ang_noise_ * options.gnss_ang_noise_;
    gnss_noise_.diagonal() << gp2, gp2, gh2, ga2, ga2, ga2;
  }

  /// Update the nominal state variable and reset the error state
  void UpdateAndReset() {
    p_ += dx_.template block<3, 1>(0, 0);
    v_ += dx_.template block<3, 1>(3, 0);
    R_ = R_ * SO3::exp(dx_.template block<3, 1>(6, 0));

    if (options_.update_bias_gyro_) {
      bg_ += dx_.template block<3, 1>(9, 0);
    }

    if (options_.update_bias_acce_) {
      ba_ += dx_.template block<3, 1>(12, 0);
    }

    g_ += dx_.template block<3, 1>(15, 0);

    ProjectCov();
    dx_.setZero();
  }

  /// 对P阵进行投影，参考式(3.63)
  void ProjectCov() {
    Mat18T J = Mat18T::Identity();
    J.template block<3, 3>(6, 6) =
        Mat3T::Identity() - 0.5 * SO3::hat(dx_.template block<3, 1>(6, 0));
    cov_ = J * cov_ * J.transpose();
  }

  /// member variable
  double current_time_ = 0.0;  // current timestamp

  /// norminal state
  VecT p_ = VecT::Zero();
  VecT v_ = VecT::Zero();
  SO3 R_;
  VecT bg_ = VecT::Zero();
  VecT ba_ = VecT::Zero();
  VecT g_{0, 0, -9.8};

  /// error state
  Vec18T dx_ = Vec18T::Zero();

  /// covariance
  Mat18T cov_ = Mat18T::Identity();

  /// noise matrix
  MotionNoiseT Q_ = MotionNoiseT::Zero();
  OdomNoiseT odom_noise_ = OdomNoiseT::Zero();
  GnssNoiseT gnss_noise_ = GnssNoiseT::Zero();

  /// gnss flag
  bool first_gnss_ = true;  // is first gnss options

  /// EKSF options
  Options options_;
};

using ESKFD = ESKF<double>;
using ESKFF = ESKF<float>;

template <typename S>
bool ESKF<S>::Predict(const common::IMU& imu) {
  assert(imu.timestamp_ >= current_time_);

  double dt = imu.timestamp_ - current_time_;
  if (dt > (5 * options_.imu_dt_) || dt < 0) {
    // 时间间隔不对，可能是第一个IMU数据，没有历史信息
    LOG(INFO) << "skip this imu because dt_ = " << dt;
    current_time_ = imu.timestamp_;
    return false;
  }

  // nominal state 递推
  VecT new_p = p_ + v_ * dt + 0.5 * (R_ * (imu.acce_ - ba_)) * dt * dt +
               0.5 * g_ * dt * dt;
  VecT new_v = v_ + R_ * (imu.acce_ - ba_) * dt + g_ * dt;
  SO3 new_R = R_ * SO3::exp((imu.gyro_ - bg_) * dt);

  R_ = new_R;
  v_ = new_v;
  p_ = new_p;
  // 其余状态维度不变

  // error state 递推
  // 计算运动过程雅可比矩阵 F，见(3.47)
  // F实际上是稀疏矩阵，也可以不用矩阵形式进行相乘而是写成散装形式，这里为了教学方便，使用矩阵形式
  Mat18T F = Mat18T::Identity();                          // 主对角线
  F.template block<3, 3>(0, 3) = Mat3T::Identity() * dt;  // p 对 v
  F.template block<3, 3>(3, 6) =
      -R_.matrix() * SO3::hat(imu.acce_ - ba_) * dt;       // v对theta
  F.template block<3, 3>(3, 12) = -R_.matrix() * dt;       // v 对 ba
  F.template block<3, 3>(3, 15) = Mat3T::Identity() * dt;  // v 对 g
  F.template block<3, 3>(6, 6) =
      SO3::exp(-(imu.gyro_ - bg_) * dt).matrix();          // theta 对 theta
  F.template block<3, 3>(6, 9) = -Mat3T::Identity() * dt;  // theta 对 bg

  // mean and cov prediction
  dx_ =
      F *
      dx_;  // 这行其实没必要算，dx_在重置之后应该为零，因此这步可以跳过，但F需要参与Cov部分计算，所以保留
  cov_ = F * cov_.eval() * F.transpose() + Q_;
  current_time_ = imu.timestamp_;
  return true;
}

template <typename S>
bool ESKF<S>::ObserveWheelSpeed(const common::Odom& odom) {
  assert(odom.timestamp_ >= current_time_);
  // odom 修正以及雅可比
  // 使用三维的轮速观测，H为3x18，大部分为零
  Eigen::Matrix<S, 3, 18> H = Eigen::Matrix<S, 3, 18>::Zero();
  H.template block<3, 3>(0, 3) = Mat3T::Identity();

  // 卡尔曼增益
  Eigen::Matrix<S, 18, 3> K =
      cov_ * H.transpose() * (H * cov_ * H.transpose() + odom_noise_).inverse();

  // velocity obs
  double velo_l = options_.wheel_radius_ * odom.left_pulse_ /
                  options_.circle_pulse_ * 2 * M_PI / options_.odom_span_;
  double velo_r = options_.wheel_radius_ * odom.right_pulse_ /
                  options_.circle_pulse_ * 2 * M_PI / options_.odom_span_;
  double average_vel = 0.5 * (velo_l + velo_r);

  VecT vel_odom(average_vel, 0.0, 0.0);
  VecT vel_world = R_ * vel_odom;

  dx_ = K * (vel_world - v_);

  // update cov
  cov_ = (Mat18T::Identity() - K * H) * cov_;

  UpdateAndReset();
  return true;
}

template <typename S>
bool ESKF<S>::ObserveGps(const common::GNSS& gnss) {
  /// GNSS 观测的修正
  assert(gnss.unix_time_ >= current_time_);

  if (first_gnss_) {
    R_ = gnss.utm_pose_.so3();
    p_ = gnss.utm_pose_.translation();
    first_gnss_ = false;
    current_time_ = gnss.unix_time_;
    return true;
  }

  assert(gnss.heading_valid_);
  ObserveSE3(gnss.utm_pose_, options_.gnss_pos_noise_,
             options_.gnss_ang_noise_);
  current_time_ = gnss.unix_time_;

  return true;
}

template <typename S>
bool ESKF<S>::ObserveSE3(const SE3& pose, double trans_noise,
                         double ang_noise) {
  /// 既有旋转，也有平移
  /// 观测状态变量中的p, R，H为6x18，其余为零
  Eigen::Matrix<S, 6, 18> H = Eigen::Matrix<S, 6, 18>::Zero();
  H.template block<3, 3>(0, 0) = Mat3T::Identity();  // P部分
  H.template block<3, 3>(3, 6) = Mat3T::Identity();  // R部分（3.66)

  // 卡尔曼增益和更新过程
  Vec6d noise_vec;
  noise_vec << trans_noise, trans_noise, trans_noise, ang_noise, ang_noise,
      ang_noise;

  Mat6d V = noise_vec.asDiagonal();
  Eigen::Matrix<S, 18, 6> K =
      cov_ * H.transpose() * (H * cov_ * H.transpose() + V).inverse();

  // 更新x和cov
  Vec6d innov = Vec6d::Zero();
  innov.template head<3>() = (pose.translation() - p_);  // 平移部分
  innov.template tail<3>() =
      (R_.inverse() * pose.so3()).log();  // 旋转部分(3.67)

  dx_ = K * innov;
  cov_ = (Mat18T::Identity() - K * H) * cov_;

  UpdateAndReset();
  return true;
}

}  // namespace estimator