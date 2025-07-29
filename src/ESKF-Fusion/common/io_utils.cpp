#include <common/io_utils.h>
#include <glog/logging.h>

namespace common {

void TxtIO::Go() {
  if (!fin) {
    LOG(FATAL) << "Could Not Find File";
    return;
  }

  while (!fin.eof()) {
    std::string line;
    std::getline(fin, line);
    if (line.empty()) {
      continue;
    }

    if (line[0] == '#') {
      // Those beginning with # are comments
      continue;
    }

    // load data from line
    std::stringstream ss;
    ss << line;
    std::string data_type;
    ss >> data_type;

    if (data_type == "IMU" && imu_proc_) {
      double time, gx, gy, gz, ax, ay, az;
      ss >> time >> gx >> gy >> gz >> ax >> ay >> az;
      // imu_proc_(IMU(time, Vec3d(gx, gy, gz) * math::kDEG2RAD, Vec3d(ax, ay,
      // az)));
      imu_proc_(IMU(time, Vec3d(gx, gy, gz), Vec3d(ax, ay, az)));
    } else if (data_type == "ODOM" && odom_proc_) {
      double time, wl, wr;
      ss >> time >> wl >> wr;
      odom_proc_(Odom(time, wl, wr));
    } else if (data_type == "GNSS" && gnss_proc_) {
      double time, lat, lon, alt, heading;
      bool heading_valid;
      ss >> time >> lat >> lon >> alt >> heading >> heading_valid;
      gnss_proc_(GNSS(time, 4, Vec3d(lat, lon, alt), heading, heading_valid));
    }
  }

  LOG(INFO) << "=================Done=================";
}
}  // namespace common