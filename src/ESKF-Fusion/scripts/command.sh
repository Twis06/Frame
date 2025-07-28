alias opt_hz="rosrun mavros mavcmd long 511 106 20000 0 0 0 0 0"
alias imu_hz="rosrun mavros mavcmd long 511 105 5000 0 0 0 0 0 & sleep 1;
rosrun mavros mavcmd long 511 31 5000 0 0 0 0 0 & sleep 1;
rosrun mavros mavcmd long 511 32 10000 0 0 0 0 0 & sleep 1;"
alias record="rosbag record --tcpnodelay /mavros/local_position/odom /ekf_fusion/fusion_odom /drone/odom /ekf_quat/ekf_odom /mavros/px4flow/raw/optical_flow_rad /mavros/imu/data /ekf_interface_fusion_node/fusion_odom"
