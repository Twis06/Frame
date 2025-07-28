#pragma once

#include <common/eigen_types.h>

#include <estimator/eskf.hpp>

namespace estimator {

class SE3Fusion {
public:
  SE3Fusion();

  ~SE3Fusion() = default;

  bool imuCallback(const common::IMU &imu);
  bool poseCallback(const common::NavStated &nav_state);
  common::NavStated GetNominalState() const { return eskf_.GetNominalState(); }

  ESKFD &getESKF() { return eskf_; }

private:
  ESKFD eskf_;
  bool first_pose_received_ = false;
};

}  // namespace estimator
