#include <estimator/se3_fusion.h>
#include <common/rotation_utils.h>

namespace estimator {

SE3Fusion::SE3Fusion() {

}

bool SE3Fusion::imuCallback(const common::IMU &imu) {
  if (!first_pose_received_) {
    return false;
  }

  // Call the ESKF predict function
  return eskf_.Predict(imu);
}

bool SE3Fusion::poseCallback(const common::NavStated &nav_state) {
  SE3 pose = nav_state.GetSE3();
  
  if (!first_pose_received_) {
    first_pose_received_ = true;

    eskf_.SetX(pose, nav_state.timestamp_);
    Eigen::Vector3d ypr = common::R2ypr(pose.so3().matrix());
    LOG(INFO) << "First pose position: " << pose.translation().transpose() << ", First pose ypr: " << ypr.transpose();
    return false;
  }

  // Call the ESKF observe function
  eskf_.ObserveSE3(pose, eskf_.GetOptions().lvio_pos_noise_, eskf_.GetOptions().lvio_ang_noise_);
  return true;
}

}  // namespace estimator