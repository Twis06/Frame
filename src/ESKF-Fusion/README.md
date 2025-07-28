# :helicopter: ESKF-Fusion

## 🛠️ Build and Run

```shell
mkdir -r ekf-fusion-ws/src
cd ekf-fusion-ws/src
git clone https://github.com/weihaoysgs/ESKF-Fusion.git
cd ..
catkin_make -j
```

launch 🏃 the ekf fusion node, the output odom is `50HZ`, and topic name is `/ekf_interface_fusion_node/fusion_odom`
```shell
source devel/setup.zsh
roslaunch eskf_fusion ekf_fusion_node.launch
```

