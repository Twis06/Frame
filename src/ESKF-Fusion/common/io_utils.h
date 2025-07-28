#pragma once

#include <common/eigen_types.h>
#include <common/gnss.h>
#include <common/imu.h>
#include <common/odom.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>

#include <fstream>
#include <functional>
#include <utility>

namespace common {

class TxtIO {
 public:
  TxtIO(const std::string &file_path) : fin(file_path) {}

  /// Define the callback function
  using IMUProcessFuncType = std::function<void(const IMU &)>;
  using OdomProcessFuncType = std::function<void(const Odom &)>;
  using GNSSProcessFuncType = std::function<void(const GNSS &)>;

  TxtIO &SetIMUProcessFunc(IMUProcessFuncType imu_proc) {
    imu_proc_ = std::move(imu_proc);
    return *this;
  }

  TxtIO &SetOdomProcessFunc(OdomProcessFuncType odom_proc) {
    odom_proc_ = std::move(odom_proc);
    return *this;
  }

  TxtIO &SetGNSSProcessFunc(GNSSProcessFuncType gnss_proc) {
    gnss_proc_ = std::move(gnss_proc);
    return *this;
  }

  // Traverse the file content and call the callback function
  void Go();

 private:
  std::ifstream fin;
  IMUProcessFuncType imu_proc_;
  OdomProcessFuncType odom_proc_;
  GNSSProcessFuncType gnss_proc_;
};
}  // namespace common
