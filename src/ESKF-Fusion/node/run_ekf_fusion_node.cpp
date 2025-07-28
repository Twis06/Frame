#include <estimator/ekf_interface.h>

int main(int argc, char *argv[])
{
  ros::init(argc, argv, "ekf_fusion_node");
  google::InitGoogleLogging(argv[0]);
  FLAGS_stderrthreshold = google::INFO;
  FLAGS_colorlogtostderr = true;
  ros::NodeHandle nh("~");

  estimator::EKFInterface ekf_fusion(nh);

  ros::spin();
  google::ShutdownGoogleLogging();
  return 0;
}
