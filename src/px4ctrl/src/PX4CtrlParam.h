#ifndef __PX4CTRLPARAM_H
#define __PX4CTRLPARAM_H

#include <ros/ros.h>
#include <Eigen/Dense>

class Parameter_t
{
public:
	std::string mode; // "classic" or "neural"
	std::string model_path;
	
	// post-traversal modes 
	int post_hover_mode;  
	double switch_hover_x;
	Eigen::Vector3d post_hover_pos;

	struct Gain
	{
		double Kp0, Kp1, Kp2;
		double Kv0, Kv1, Kv2;
		double Kvi0, Kvi1, Kvi2;
		double Kvd0, Kvd1, Kvd2;
		double KAngR, KAngP, KAngY;
	};

	struct RotorDrag
	{
		double x, y, z;
		double k_thrust_horz;
	};

	struct MsgTimeout
	{
		double odom;
		double rc;
		double cmd;
		double imu;
		double bat;
		double hil_cmd;
	};

	struct ThrustMapping
	{
		bool print_val;
		double K1;
		double K2;
		double K3;
		bool accurate_thrust_model;
		double hover_percentage;
		double accurate_thrust_scale;

	};

	struct RCReverse
	{
		bool roll;
		bool pitch;
		bool yaw;
		bool throttle;
	};

	struct AutoTakeoffLand
	{
		bool enable;
		bool enable_auto_arm;
		bool no_RC;
		double height;
		double speed;
	};

	Gain gain;
	RotorDrag rt_drag;
	MsgTimeout msg_timeout;
	RCReverse rc_reverse;
	ThrustMapping thr_map;
	AutoTakeoffLand takeoff_land;

	int pose_solver;
	int policy_modality;
	double mass;
	double gra;
	double max_angle;

	double ctrl_freq_max;
	double control_interval;
	double spin_interval;

	double max_manual_vel;
	double low_voltage;

	bool use_bodyrate_ctrl;
	Parameter_t();
	void config_from_ros_handle(const ros::NodeHandle &nh);
	void config_full_thrust(double hov);
	int spin_mode;


private:
	template <typename TName, typename TVal>
	void read_essential_param(const ros::NodeHandle &nh, const TName &name, TVal &val)
	{
		if (nh.getParam(name, val))
		{
			// pass
		}
		else
		{
			ROS_ERROR_STREAM("Read param: " << name << " failed.");
			ROS_BREAK();
		}
	};
	bool read_optional_param(const ros::NodeHandle &nh, const std::string &name, double &val)
	{
		if (nh.getParam(name, val))
		{
			return true;
		}
		else
		{
			return false;
		}
	}
};

#endif