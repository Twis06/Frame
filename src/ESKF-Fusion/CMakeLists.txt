cmake_minimum_required(VERSION 3.0.2)
project(eskf_fusion)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++17")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

add_definitions("-DCATKIN_ENABLE_TESTING=0")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  rosbag
)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

catkin_package()

add_subdirectory(${PROJECT_SOURCE_DIR}/thirdparty/backward-cpp)

include_directories(
  ${PROJECT_SOURCE_DIR}
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/thirdparty/sophus
  ${PROJECT_SOURCE_DIR}/thirdparty/
)

set(THIRD_PARTY_LIBS
    glog
    gflags
    ${OpenCV_LIBS}
    ${catkin_LIBRARIES}
)

add_library(ESKF_Fusion
  common/utm_convert.cpp
  common/io_utils.cpp
  common/visualizer.cpp
  estimator/static_imu_init.cpp
  estimator/se3_fusion.cpp
  estimator/ekf.cpp
  estimator/ekf_interface.cpp

  ${PROJECT_SOURCE_DIR}/thirdparty/utm_convert/tranmerc.cc
  ${PROJECT_SOURCE_DIR}/thirdparty/utm_convert/utm.cc
)

target_link_libraries(ESKF_Fusion
  ${THIRD_PARTY_LIBS}
)

add_subdirectory(test)
add_subdirectory(node)
