cmake_minimum_required(VERSION 2.8.3)
project(px4ctrl)

set(CMAKE_VERBOSE_MAKEFILE "false")
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  quadrotor_msgs
  geometry_msgs
  sensor_msgs
  uav_utils
  mavros
)

execute_process(
  COMMAND whoami
  OUTPUT_VARIABLE USERNAME
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(USERNAME STREQUAL "nv")
  set(CMAKE_CXX_FLAGS "-std=c++14")
  set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-11.4")

  set(CUDA_SEPARABLE_COMPILATION ON)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -use_fast_math;)
  set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

  set(CUDA_NVCC_FLAGS -gencode arch=compute_87,code=sm_87;)
  list(APPEND CMAKE_PREFIX_PATH /home/nv/.local/lib/python3.8/site-packages/torch)
  set(Torch_DIR /home/nv/.local/lib/python3.8/site-packages/torch)

  find_package(Torch REQUIRED)

elseif(USERNAME STREQUAL "intel")
  set(CMAKE_CXX_FLAGS "-std=c++17")
  find_package(Torch REQUIRED PATHS "/home/intel/Apps/libtorch-cxx11-abi-shared-with-deps-2.1.0+cu121/libtorch")

elseif(USERNAME STREQUAL "moderngangster")
  set(CMAKE_CXX_FLAGS "-std=c++17")
  set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.1")
  find_package(Torch REQUIRED PATHS "/home/moderngangster/apps/libtorch-cxx11-abi-shared-with-deps-2.1.0+cu121/libtorch")
elseif(USERNAME STREQUAL "xugt")
  set(CMAKE_CXX_FLAGS "-std=c++17")
  set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-11.4")
  set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.4/bin/nvcc")
  find_package(Torch REQUIRED PATHS "/home/xugt/Apps/libtorch-cxx11-abi-shared-with-deps-2.1.0+cu121/libtorch")
elseif(USERNAME STREQUAL "pn")
  set(CMAKE_CXX_FLAGS "-std=c++17")
  set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-11.4")
  set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.4/bin/nvcc")
  # find_package(Torch REQUIRED PATHS "/home/xugt/Apps/libtorch-cxx11-abi-shared-with-deps-2.1.0+cu121/libtorch")
else()
  message(FATAL_ERROR "Error: Username '${USERNAME}' is not supported. Only 'nv' and 'intel' users are allowed now.")
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
find_package(Eigen3 REQUIRED) 

catkin_package(
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN_INCLUDE_DIRS}
  ${TORCH_INCLUDE_DIRS}
  include/${PROJECT_NAME}
  include
)

# add_executable(px4ctrl_node 
#   src/px4ctrl_node.cpp
#   src/PX4CtrlFSM.cpp
#   src/PX4CtrlParam.cpp
#   src/controller.cpp
#   src/nnpolicy.cpp
#   src/input.cpp
# )

# add_executable(px4ctrl_test_node 
#   src/px4ctrl_test.cpp
#   src/PX4CtrlFSM.cpp
#   src/PX4CtrlParam.cpp
#   src/controller.cpp
#   src/nnpolicy.cpp
#   src/input.cpp
# )

add_executable(px4ctrl_node 
  src/px4ctrl_node.cpp
  src/PX4CtrlFSM_optflow.cpp
  src/PX4CtrlParam.cpp
  src/controller.cpp
  src/nnpolicy.cpp
  src/input.cpp
)

add_executable(px4ctrl_test_node 
  src/px4ctrl_test.cpp
  src/PX4CtrlFSM_optflow.cpp
  src/PX4CtrlParam.cpp
  src/controller.cpp
  src/nnpolicy.cpp
  src/input.cpp
)

add_dependencies(px4ctrl_node quadrotor_msgs)

add_dependencies(px4ctrl_test_node quadrotor_msgs)


target_link_libraries(px4ctrl_node
  ${catkin_LIBRARIES}
  ${TORCH_LIBRARIES}
)

target_link_libraries(px4ctrl_test_node
  ${catkin_LIBRARIES}
  ${TORCH_LIBRARIES}
)

catkin_install_python(PROGRAMS 
  thrust_calibrate_scrips/thrust_calibrate.py
  scripts/HIL_node.py
  scripts/virtual_HIL_node.py
  scripts/camera_control.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
