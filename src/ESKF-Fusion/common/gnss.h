#pragma once

#include <common/eigen_types.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/NavSatFix.h>

namespace common {

/// GNSS status bit information
/// It is usually provided by GNSS vendors. Here, the status bit provided by Qianxun is used
enum class GpsStatusType {
  GNSS_FLOAT_SOLUTION = 5,         // Floating-point solution (between cm and dm)
  GNSS_FIXED_SOLUTION = 4,         // Fixed solution (cm level)
  GNSS_PSEUDO_SOLUTION = 2,        // Pseudo-range difference decomposition (decimeter level)
  GNSS_SINGLE_POINT_SOLUTION = 1,  // Single-point solution (10m level)
  GNSS_NOT_EXIST = 0,              // There is no GPS signal
  GNSS_OTHER = -1,                 // Others
};

/// UTM coordinate
struct UTMCoordinate {
  UTMCoordinate() = default;
  explicit UTMCoordinate(int zone, const Vec2d& xy = Vec2d::Zero(),
                         bool north = true)
      : zone_(zone), xy_(xy), north_(north) {}

  int zone_ = 0;              // utm region
  Vec2d xy_ = Vec2d::Zero();  // utm xy
  double z_ = 0;              // z height(Directly from gps)
  bool north_ = true;         // Whether it is in the Northern Hemisphere
};

/// A GNSS reading structure
struct GNSS {
  GNSS() = default;
  GNSS(double unix_time, int status, const Vec3d& lat_lon_alt, double heading,
       bool heading_valid)
      : unix_time_(unix_time),
        lat_lon_alt_(lat_lon_alt),
        heading_(heading),
        heading_valid_(heading_valid) {
    status_ = GpsStatusType(status);
  }

  /// Convert from ros' NavSatFix
  /// NOTE This only has position information but no orientation information.
  ///  Please convert the UTM coordinates from the code of 'utm_convert'
  GNSS(sensor_msgs::NavSatFix::Ptr msg) {
    unix_time_ = msg->header.stamp.toSec();
    // Status bit
    if (int(msg->status.status) >= int(sensor_msgs::NavSatStatus::STATUS_FIX)) {
      status_ = GpsStatusType::GNSS_FIXED_SOLUTION;
    } else {
      status_ = GpsStatusType::GNSS_OTHER;
    }
    // Longitude and latitude
    lat_lon_alt_ << msg->latitude, msg->longitude, msg->altitude;
  }

  double unix_time_ = 0;                                  // unix system time
  GpsStatusType status_ = GpsStatusType::GNSS_NOT_EXIST;  // GNSS status bit
  Vec3d lat_lon_alt_ = Vec3d::Zero();                     // Longitude, latitude and altitude. The units of the first two are degrees
  double heading_ = 0.0;                                  // The azimuth Angle read by the dual antennas, with the unit being degrees
  bool heading_valid_ = false;                            // Whether the azimuth is valid
  UTMCoordinate utm_;                                     // UTM coordinates (including regions and the like)
  bool utm_valid_ = false;                                // Has the UTM coordinate been calculated? (If the longitude and latitude give incorrect values, it is also false here.)
  SE3 utm_pose_;                                          // 6DoF Pose for post-processing
};

}  // namespace common

using GNSSPtr = std::shared_ptr<common::GNSS>;
