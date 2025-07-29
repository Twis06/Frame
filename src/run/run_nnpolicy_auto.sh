# source ./devel/setup.bash
sudo chmod 777 /dev/ttyTHS0
sudo chmod 777 /dev/ttyACM0

roslaunch mavros px4.launch & sleep 4;
rosrun mavros mavcmd long 511 31 5000 0 0 0 0 0 & sleep 1;   # ATTITUDE_QUATERNION
rosrun mavros mavcmd long 511 105 5000 0 0 0 0 0 & sleep 1;  # HIGHRES_IMU
rosrun mavros mavcmd long 511 106 20000 0 0 0 0 0 & sleep 1;  # HIGHRES_IMU

# rosrun mavros mavcmd long 511 83 5000 0 0 0 0 0 & sleep 1;   # ATTITUDE_TARGET
# rosrun mavros mavcmd long 511 147 5000 0 0 0 0 0 & sleep 1;  # BATTERY_STATUS

# roslaunch vrpn_client_ros sample.launch server:=10.1.1.198 &sleep 2;
# roslaunch ekf_quat nokov.launch & sleep 2;

# roslaunch traj_server traj_server.launch & sleep 1;
roslaunch px4ctrl run_ctrl_nnpolicy.launch & sleep 2;
sudo ./renice.sh;
# roslaunch waypoint_trajectory_generator traj_gen.launch 

wait;
