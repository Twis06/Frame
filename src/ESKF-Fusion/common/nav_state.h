#pragma once

#include <sophus/so3.hpp>

#include "common/eigen_types.h"

namespace common {

/*
* State variables for navigation
* @tparam T scalar type
*/
template <typename T>
struct NavState {
  using Vec3 = Eigen::Matrix<T, 3, 1>;
  using SO3 = Sophus::SO3<T>;

  NavState() = default;

  // from time, R, p, v, bg, ba
  explicit NavState(double time, const SO3& R = SO3(),
                    const Vec3& t = Vec3::Zero(), const Vec3& v = Vec3::Zero(),
                    const Vec3& bg = Vec3::Zero(),
                    const Vec3& ba = Vec3::Zero())
      : timestamp_(time), R_(R), p_(t), v_(v), bg_(bg), ba_(ba) {}

  // from pose and vel
  NavState(double time, const SE3& pose, const Vec3& vel = Vec3::Zero())
      : timestamp_(time), R_(pose.so3()), p_(pose.translation()), v_(vel) {}

  /// Convert to Sophus
  Sophus::SE3<T> GetSE3() const { return SE3(R_, p_); }

  friend std::ostream& operator<<(std::ostream& os, const NavState<T>& s) {
    os << "p: " << s.p_.transpose() << ", v: " << s.v_.transpose()
      //  << ", q: " << s.R_.unit_quaternion().coeffs().transpose()
       << ", angle: " << s.ang_.transpose()
       << ", bg: " << s.bg_.transpose() << ", ba: " << s.ba_.transpose();
    return os;
  }

  double timestamp_ = 0;    // Timestamp
  SO3 R_;                   // Rotation
  Eigen::Quaternion<T> q_;  // Quaternion
  Vec3 p_ = Vec3::Zero();   // Translation
  Vec3 v_ = Vec3::Zero();   // Speed
  Vec3 bg_ = Vec3::Zero();  // Gyro bias
  Vec3 ba_ = Vec3::Zero();  // Acc bias
  Vec3 ang_ = Vec3::Zero(); // Euler angle
};

using NavStated = NavState<double>;
using NavStatef = NavState<float>;

}  // namespace common
