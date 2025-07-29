/*************************************************************/
/* Acknowledgement: @wdww */
/*************************************************************/

#ifndef __NNPOLICY_H
#define __NNPOLICY_H

#include <mavros_msgs/AttitudeTarget.h>
#include <quadrotor_msgs/Px4ctrlDebug.h>
#include <queue>

#include "input.h"
#include <torch/torch.h>
#include <torch/script.h>
#include <Eigen/Dense>
#include "controller.h"

// #include <opencv2/opencv.hpp>

// struct Controller_Output_t
// {
// 	// Orientation of the body frame with respect to the world frame
// 	Eigen::Quaterniond q;

// 	// Body rates in body frame
// 	Eigen::Vector3d bodyrates; // [rad/s]

// 	// Collective mass normalized thrust
// 	double thrust; // m/s^2
// };

class NNPolicy
{
public:
	Parameter_t &param;

	torch::jit::Module model;
	torch::Device device;
	bool is_initialized;

	quadrotor_msgs::Px4ctrlDebug debug; //debug

	// Thrust-accel mapping params
	double thr_scale_compensate;
	const double rho2 = 0.998; // do not change
	double thr2acc;
	double P;

	std::queue<std::pair<ros::Time, double>> timed_thrust;

	// Pre-allocate tensors
	torch::Tensor state_gpu = torch::zeros({1, 13}, torch::TensorOptions().device(device).dtype(torch::kFloat32));
    torch::Tensor action_gpu = torch::zeros({1, 4}, torch::TensorOptions().device(device).dtype(torch::kFloat32));
    torch::Tensor state_cpu = torch::zeros({1, 13}, torch::TensorOptions().pinned_memory(true).dtype(torch::kFloat32));
    torch::Tensor action_cpu = torch::zeros({1, 4}, torch::TensorOptions().pinned_memory(true).dtype(torch::kFloat32));
    Eigen::Matrix<float, 13, 1> state_eigen = Eigen::Matrix<float, 13, 1>::Zero();
    Eigen::Matrix<float, 4, 1> action_eigen = Eigen::Matrix<float, 4, 1>::Zero();

	// last action: normalized thrust, bodyrates
	Eigen::Vector4f last_cmd = Eigen::Vector4f::Zero();  // TODO: hover cmd
	bool tanh_norm = true;  //! the utilization of tanh should be consistent with the training setup

	NNPolicy(Parameter_t &); 

	bool initialize(const std::string &model_path)
	{
		try {
			model = torch::jit::load(model_path);
			model.to(device);
			model.eval();
			
			is_initialized = true;
			ROS_INFO("\033[32m[px4ctrl] Model loaded successfully!\033[32m");
		} catch (const c10::Error &e) {
			ROS_ERROR("[px4ctrl] Error loading the model: %s", e.what());
		}
		return is_initialized;
	}

	void warmup_step(bool print);

	quadrotor_msgs::Px4ctrlDebug state_based_control(
		const Odom_Data_t &odom,
		Controller_Output_t &u,
		double voltage);

	quadrotor_msgs::Px4ctrlDebug pixel_based_HIL_control(
		const Odom_Data_t &odom,
		Controller_Output_t &u_hil,
		Controller_Output_t &u,
		double voltage);

	quadrotor_msgs::Px4ctrlDebug attitude_recovery_control(
		const Imu_Data_t &imu,
		Controller_Output_t &u,
		double voltage);

	torch::Tensor inference(const torch::Tensor state);
	
	// torch::Tensor pixel_based_control(
	// 	const cv::Mat &image,
	// 	const Odom_Data_t &odom,
	// 	Controller_Output_t &u,
	// 	double voltage);

	Eigen::Vector3d computeFeedBackControlBodyrates(
		const Eigen::Quaterniond &des_q,
		const Eigen::Quaterniond &est_q,
		const Parameter_t &param);

	double computeDesiredCollectiveThrustSignal(
		const double &des_acc_norm,
		const Parameter_t &param,
		double voltage);

	double AccurateThrustAccMapping(
		const double des_acc_z,
		double voltage,
		const Parameter_t &param) const;

	Eigen::Vector4f normalize_cmd(Controller_Output_t cmd)
	{
		Eigen::Vector4f norm_cmd;
		norm_cmd(0) = (cmd.thrust - act_mean(0)) / act_std(0);
		norm_cmd(1) = (cmd.bodyrates(0) - act_mean(1)) / act_std(1);
		norm_cmd(2) = (cmd.bodyrates(1) - act_mean(2)) / act_std(2);
		norm_cmd(3) = (cmd.bodyrates(2) - act_mean(3)) / act_std(3);
		return norm_cmd;
	}

	void denormalize_cmd(Eigen::Vector4f &norm_cmd)
	{
		norm_cmd(0) = norm_cmd(0) * act_std(0) + act_mean(0);
		norm_cmd(1) = norm_cmd(1) * act_std(1) + act_mean(1);
		norm_cmd(2) = norm_cmd(2) * act_std(2) + act_mean(2);
		norm_cmd(3) = norm_cmd(3) * act_std(3) + act_mean(3);
	}

	bool estimateThrustModel(
		const Eigen::Vector3d &est_v,
		const double voltage,
		const Parameter_t &param);
		
	void resetThrustMapping(void);

private:
	Eigen::Vector4f act_mean = Eigen::Vector4f(13.0, 0.0, 0.0, 0.0);
	Eigen::Vector4f act_std = Eigen::Vector4f(7.0, 6.0, 2.0, 1.0);
};

using namespace Eigen;

inline Vector3d quat2euler(const Quaterniond &q)
{
    Vector3d euler;  // roll, pitch, yaw
    euler(0) = atan2(2 * (q.w() * q.x() + q.y() * q.z()), 1 - 2 * (q.x() * q.x() + q.y() * q.y()));
    euler(1) = asin(2 * (q.w() * q.y() - q.z() * q.x()));
    euler(2) = atan2(2 * (q.w() * q.z() + q.x() * q.y()), 1 - 2 * (q.y() * q.y() + q.z() * q.z()));
    return euler;
}

#endif