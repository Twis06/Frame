sudo nvpmodel -m 8 & sleep 1;
chmod +x ./scrips/max_cpu_freq.sh;
chmod +x ./scrips/max_emc_freq.sh;
chmod +x ./scrips/max_gpu_freq.sh;

sudo ./scrips/max_cpu_freq.sh;
sudo ./scrips/max_emc_freq.sh;
sudo ./scrips/max_gpu_freq.sh;

source ../devel/setup.bash
sudo chmod 777 /dev/ttyTHS*

roslaunch realsense2_camera rs_camera.launch & sleep 5;


wait;
