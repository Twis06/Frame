#include "PX4CtrlFSM.h"
#include <uav_utils/converters.h>
using namespace std;
using namespace uav_utils;

// PX4CtrlFSM::PX4CtrlFSM(Parameter_t &param_, Controller &controller_) : param(param_), controller(controller_)
// {
// 	state = MANUAL_CTRL;
// 	hover_pose.setZero();
// }

PX4CtrlFSM::PX4CtrlFSM(Parameter_t &param_, Controller &controller_, NNPolicy &nnpolicy_) : param(param_), controller(controller_), policy(nnpolicy_)
{
	state = MANUAL_CTRL;
	hover_pose.setZero();
}

/*
		Finite State Machine

		  system start
				|
				|
				v
	----- > MANUAL_CTRL <-----------------
	|         ^   |    \                 |
	|         |   |     \                |
	|         |   |      > AUTO_TAKEOFF  |
	|         |   |        /             |
	|         |   |       /              |
	|         |   |      /               |
	|         |   v     /                |
	|       AUTO_HOVER <                 |
	|         ^   |  \  \                |
	|         |   |   \  \               |
	|         |	  |    > AUTO_LAND -------
	|         |   |
	|         |   v
	-------- CMD_CTRL

*/

void PX4CtrlFSM::process()
{

	ros::Time now_time = ros::Time::now();
	Controller_Output_t u;
	Desired_State_t des(odom_data);
	bool rotor_low_speed_during_land = false;

	// STEP1: state machine runs
	// 变量在CMD后赋值
	// static bool optical_flow_state_once = true;
	// static bool time_sys_state_once = true;
	// static bool pub_auto_hover_state_once = true;
	// static bool hover_print_once_ = true;
	// static bool print_auto_info_once = true;
	// static bool pub_cmd_state_once = true;
	// static bool cmd_print_once_ = true;
	// static bool pub_aoto_takeoff_state_once = true;
	// static bool pub_land_state_once = true;

	switch (state)
	{
	case MANUAL_CTRL:
	{
		if (rc_data.enter_hover_mode) // Try to jump to AUTO_HOVER
		{
			// if (!odom_is_received(now_time))
			// {
			// 	ROS_ERROR("[px4ctrl] Reject AUTO_HOVER(L2). No odom!");
			// 	break;
			// }
			// if (cmd_is_received(now_time))
			// {
			// 	ROS_ERROR("[px4ctrl] Reject AUTO_HOVER(L2). You are sending commands before toggling into AUTO_HOVER, which is not allowed. Stop sending commands now!");
			// 	break;
			// }
			// if (odom_data.v.norm() > 3.0)
			// {
			// 	ROS_ERROR("[px4ctrl] Reject AUTO_HOVER(L2). Odom_Vel=%fm/s, which seems that the locolization module goes wrong!", odom_data.v.norm());
			// 	break;
			// }

			state = CMD_CTRL;  
			controller.resetThrustMapping();
			policy.resetThrustMapping();
			// set_hov_with_odom();
			toggle_offboard_mode(true);
			ROS_INFO("\033[32m[px4ctrl] MANUAL_CTRL(L1) --> CMD_CTRL(L3)\033[32m");
			// ROS_INFO("\033[32m Current mode is: %s \033[32m", state_data.current_state.mode.c_str());
			// ROS_INFO("\033[32m Before mode is: %s \033[32m", state_data.state_before_offboard.mode.c_str());

		}
		else if (param.takeoff_land.enable && takeoff_land_data.triggered && takeoff_land_data.takeoff_land_cmd == quadrotor_msgs::TakeoffLand::TAKEOFF) // Try to jump to AUTO_TAKEOFF
		{
			// kdkd debug info
			//  static bool auto_take_off_state_once = true;

			if (!odom_is_received(now_time))
			{
				// //kdkd debug info
				// auto_take_off_state_once = true;
				// px4_node_state.debug_info = "[px4ctrl] Reject AUTO_TAKEOFF. No odom!";
				ROS_ERROR("[px4ctrl] Reject AUTO_TAKEOFF. No odom!");
				break;
			}
			if (cmd_is_received(now_time))
			{
				// kdkd debug info
				//  auto_take_off_state_once = true;
				//  px4_node_state.debug_info = "[px4ctrl] Reject AUTO_TAKEOFF. You are sending commands before toggling into AUTO_TAKEOFF, which is not allowed. Stop sending commands now!";
				ROS_ERROR("[px4ctrl] Reject AUTO_TAKEOFF. You are sending commands before toggling into AUTO_TAKEOFF, which is not allowed. Stop sending commands now!");
				break;
			}
			if (odom_data.v.norm() > 0.1)
			{
				// auto_take_off_state_once = true;
				// px4_node_state.debug_info = "[px4ctrl] Reject AUTO_TAKEOFF.non-static takeoff is not allowed!Odom_Vel is too large";
				ROS_ERROR("[px4ctrl] Reject AUTO_TAKEOFF. Odom_Vel=%fm/s, non-static takeoff is not allowed!", odom_data.v.norm());
				break;
			}
			if (!get_landed())
			{
				// auto_take_off_state_once = true;
				// px4_node_state.debug_info = "[px4ctrl] Reject AUTO_TAKEOFF. land detector says that the drone is not landed now!";
				ROS_ERROR("[px4ctrl] Reject AUTO_TAKEOFF. land detector says that the drone is not landed now!");
				break;
			}

			// kdkd time sync
			//  if (local_time_data.not_synchronized)
			//  {
			//  	//kdkd state
			//  	// static bool time_sys_state_once = true;
			//  	if (time_sys_state_once)
			//  	{
			//  		px4_node_state.current_node_state = 11;
			//  		px4_state_to_drone_node_pub.publish(px4_node_state);
			//  		time_sys_state_once = false;
			//  	}
			//  	px4_node_state.debug_info = "[px4ctrl] Time Not Synchronization!!!,swarm can not takeoff!!!";
			//  	std::cout<<"local_time_data.not_synchronized:"<<local_time_data.not_synchronized<<std::endl;
			//  	ROS_ERROR("Time Not Synchronization!!!,swarm can not takeoff!!!");
			//  	break;
			//  }
			//-------------

			if (rc_is_received(now_time)) // Check this only if RC is connected.
			{
				if (!rc_data.is_hover_mode || !rc_data.is_command_mode || !rc_data.check_centered())
				{
					ROS_ERROR("[px4ctrl] Reject AUTO_TAKEOFF. If you have your RC connected, keep its switches at \"auto hover\" and \"command control\" states, and all sticks at the center, then takeoff again.");
					while (ros::ok())
					{
						ros::Duration(0.01).sleep();
						ros::spinOnce();
						if (rc_data.is_hover_mode && rc_data.is_command_mode && rc_data.check_centered())
						{
							ROS_INFO("\033[32m[px4ctrl] OK, you can takeoff again.\033[32m");
							break;
						}
					}
					break;
				}
			}

			state = AUTO_TAKEOFF;
			controller.resetThrustMapping();
			policy.resetThrustMapping();
			set_start_pose_for_takeoff_land(odom_data);
			toggle_offboard_mode(true);				  // toggle on offboard before arm
			for (int i = 0; i < 10 && ros::ok(); ++i) // wait for 0.1 seconds to allow mode change by FMU // mark
			{
				ros::Duration(0.01).sleep();
				ros::spinOnce();
			}
			if (param.takeoff_land.enable_auto_arm)
			{
				toggle_arm_disarm(true);
			}
			takeoff_land.toggle_takeoff_land_time = now_time;

			ROS_INFO("\033[32m[px4ctrl] MANUAL_CTRL(L1) --> AUTO_TAKEOFF\033[32m");
			// kdkd debug_info
			//  if (auto_take_off_state_once)
			//  {
			//  	px4_node_state.current_node_state = 8;
			//  	px4_state_to_drone_node_pub.publish(px4_node_state);
			//  	auto_take_off_state_once = false;
			//  }
		}

		if (rc_data.toggle_reboot) // Try to reboot. EKF2 based PX4 FCU requires reboot when its state estimator goes wrong.
		{
			if (state_data.current_state.armed)
			{
				ROS_ERROR("[px4ctrl] Reject reboot! Disarm the drone first!");
				break;
			}
			reboot_FCU();
		}

		break;
	}

	case AUTO_HOVER:
	{
		// kdkd debug info
		//  if (pub_auto_hover_state_once)
		//  {
		//  	px4_node_state.current_node_state = 7;
		//  	px4_state_to_drone_node_pub.publish(px4_node_state);
		//  	pub_auto_hover_state_once = false;
		//  }
		if (!rc_data.is_hover_mode || !odom_is_received(now_time))
		{
			state = MANUAL_CTRL;
			toggle_offboard_mode(false);

			ROS_WARN("[px4ctrl] AUTO_HOVER(L2) --> MANUAL_CTRL(L1)");
		}
		else if ((rc_data.is_command_mode && param.mode == "classic" && cmd_is_received(now_time) && !done) 
		|| (rc_data.is_command_mode && param.mode == "neural" && rc_data.is_check_inference_mode && !done))
		{
			if (state_data.current_state.mode == "OFFBOARD")
			{
				if (param.mode == "classic")
				{
					state = CMD_CTRL;
					des = get_cmd_des();
					ROS_INFO("\033[32m[px4ctrl] AUTO_HOVER(L2) --> CMD_CTRL(L3)\033[32m");
				}
				else
				{
					is_nn_warmup = true;
					if (is_nn_warmup)
					{
						state = CMD_CTRL;
						ROS_INFO("\033[32m[px4ctrl_nnpolicy] AUTO_HOVER(L2) --> CMD_CTRL(L3)\033[32m");
					}
					set_hov_with_rc();
					des = get_hover_des(); // PID for hovering
				}
			}
		}
		else if (takeoff_land_data.triggered && takeoff_land_data.takeoff_land_cmd == quadrotor_msgs::TakeoffLand::LAND)
		{

			state = AUTO_LAND;
			set_start_pose_for_takeoff_land(odom_data);

			ROS_INFO("\033[32m[px4ctrl] AUTO_HOVER(L2) --> AUTO_LAND\033[32m");
		}
		else
		{
			if (param.mode == "neural")
			{
				policy.warmup_step(false); // keep inference warmup
				//! [NOTE] note that the inference time cannot be too long, otherwise the drone will be out of control without PID control
			}
			
			set_hov_with_rc();
			des = get_hover_des();
			if ((rc_data.enter_command_mode) ||
				(takeoff_land.delay_trigger.first && now_time > takeoff_land.delay_trigger.second))
			{
				takeoff_land.delay_trigger.first = false;
				publish_trigger(odom_data.msg);
				ROS_INFO("\033[32m[px4ctrl] TRIGGER sent, allow user command.\033[32m");
			}

			// cout << "des.p=" << des.p.transpose() << endl;
		}

		break;
	}

	case CMD_CTRL:
	{
		if (param.mode == "classic")
		{
			// kdkd debug info
			//  if (pub_cmd_state_once)
			//  {
			//  	px4_node_state.current_node_state = 5;
			//  	px4_state_to_drone_node_pub.publish(px4_node_state);
			//  	pub_cmd_state_once = false;
			//  }
			if (!rc_data.is_hover_mode || !odom_is_received(now_time))
			{
				state = MANUAL_CTRL;
				toggle_offboard_mode(false);

				ROS_WARN("[px4ctrl] From CMD_CTRL(L3) to MANUAL_CTRL(L1)!");
			}
			else if (!rc_data.is_command_mode || !cmd_is_received(now_time))
			{
				state = AUTO_HOVER;
				set_hov_with_odom();
				des = get_hover_des();
				ROS_INFO("[px4ctrl] From CMD_CTRL(L3) to AUTO_HOVER(L2)! case1");
			}
			else
			{
				des = get_cmd_des();
			}

			if (takeoff_land_data.triggered && takeoff_land_data.takeoff_land_cmd == quadrotor_msgs::TakeoffLand::LAND)
			{
				ROS_ERROR("[px4ctrl] Reject AUTO_LAND, which must be triggered in AUTO_HOVER. \
						Stop sending control commands for longer than %fs to let px4ctrl return to AUTO_HOVER first.",
						  param.msg_timeout.cmd);
			}

		}
		else if (param.mode == "neural")
		{
			// TODO: check stable inference time and output (set some flag)
			// if (!rc_data.is_hover_mode || !odom_is_received(now_time))
			if (!rc_data.is_hover_mode)
			{
				state = MANUAL_CTRL;
				toggle_offboard_mode(false);
				ROS_WARN("[px4ctrl] From CMD_CTRL(L3) to MANUAL_CTRL(L1)!");
			}
			// else if (param.policy_modality == 2 )  //&& !hil_cmd_is_received(now_time)
			// {
			// 	state = AUTO_HOVER;
			// 	set_hov_with_odom();
			// 	des = get_hover_des();

			// 	ROS_WARN("[px4ctrl] From CMD_CTRL(L3) to AUTO_HOVER(L2) due to no timely command from the HIL node!");
			// }
			else if (!rc_data.is_command_mode)
			{
				state = AUTO_HOVER;
				set_hov_with_odom();
				des = get_hover_des();
				ROS_INFO("\033[32m[px4ctrl] From CMD_CTRL(L3) to AUTO_HOVER(L2)! case2 check your channel 6\033[32m");
			}
			else  // run the policy
			{
				if (param.post_hover_mode == 0) // directly applying mocap for hovering
				{  
					if (odom_data.p(0) > param.switch_hover_x) 
					{
						state = AUTO_HOVER;
						set_hov_with_odom_for_cross(); // 
						des = get_hover_des();
						ROS_INFO("\033[32m[px4ctrl_nnpolicy] From CMD_CTRL(L3) to AUTO_HOVER(L2) due to cross!!! case3\033[32m");
					}
				}
				else if (param.post_hover_mode == 1) // use px4 attitude control then mocap for hovering
				{  	
					if (if_switch2hover.triggered)
					{
						Vector3d ypr = quaternion_to_ypr(odom_data.q);
						// std::cout<< "cross pitch : " << ypr(1)*180/3.14 << std::endl;
						// std::cout<< "cross roll : " << ypr(2)*180/3.14 << std::endl;

						if (abs(ypr(1)) > 5.0 / 180.0 * M_PI || abs(ypr(2)) > 10.0 / 180.0 * M_PI)  // the case use px4 attitude control
						{
							state = ATTI_RECOVER;
							ROS_INFO("\033[32m[px4ctrl_nnpolicy] From CMD_CTRL(L3) to ATTI_RECOVER due to cross!!!\033[32m");
						}
						else  // the case use mocap for hovering
						{	state = MANUAL_CTRL;
							toggle_offboard_mode(false);
							ROS_INFO("\033[32m[px4ctrl_nnpolicy] From CMD_CTRL(L3) to AUTO_HOVER(L2) due to attitude is ok!!!\033[32m");
							// ROS_INFO("\033[32m Current mode is: %s \033[32m", state_data.current_state.mode.c_str());
							// ROS_INFO("\033[32m Before mode is: %s \033[32m", state_data.state_before_offboard.mode.c_str());
						}
					}
					else
					{
						// ROS_INFO("in else case");
					}
					
				}
			}

			if (takeoff_land_data.triggered && takeoff_land_data.takeoff_land_cmd == quadrotor_msgs::TakeoffLand::LAND)
			{
				ROS_ERROR("[px4ctrl] Reject AUTO_LAND, which must be triggered in AUTO_HOVER. \
						Stop sending control commands for longer than %fs to let px4ctrl return to AUTO_HOVER first.",
						  param.msg_timeout.cmd);
			}
		}
		else
		{
			ROS_ERROR("[px4ctrl_nnpolicy] wrong mode set! Please check the config file!");
		}

		break;
	}

	case ATTI_RECOVER:
	{
		Vector3d ypr = quaternion_to_ypr(imu_data.q);
		std::cout << "ypr: " << ypr.transpose() << std::endl;
		// if (!rc_data.is_hover_mode || !odom_is_received(now_time))
		// {
		// 	state = MANUAL_CTRL;
		// 	toggle_offboard_mode(false);

		// 	ROS_WARN("[px4ctrl] ATTI_RECOVER(L2) --> MANUAL_CTRL(L1)");
		// }
		// else if (!rc_data.is_command_mode)
		// {
		// 	state = AUTO_HOVER;
		// 	set_hov_with_odom();
		// 	des = get_hover_des();
		// 	ROS_INFO("[px4ctrl] From ATTI_RECOVER (not command_mode) to AUTO_HOVER(L2)!");
		// }
		// else if ((abs(ypr(1)) < 5.0 / 180.0 * M_PI && abs(ypr(2)) < 10.0 / 180.0 * M_PI))
		if ((abs(ypr(1)) < 5.0 / 180.0 * M_PI && abs(ypr(2)) < 10.0 / 180.0 * M_PI))
		{			
			state = MANUAL_CTRL;
			toggle_offboard_mode(false);
			ROS_WARN("[px4ctrl] From ATTI_RECOVER(L3) to MANUAL_CTRL(L1)!");
			// ROS_INFO("\033[32m Current mode is: %s \033[32m", state_data.current_state.mode.c_str());
			// state = AUTO_HOVER;
			// set_hov_with_odom_for_cross(); // 
			// des = get_hover_des();
			// // std::cout<< "\033[32m[px4ctrl_nnpolicy] attitude is ok: \033[32m"<< ypr.transpose() << std::endl;
			// ROS_INFO("\033[32m[px4ctrl_nnpolicy] From ATTI_RECOVER (attitude ok!) to AUTO_HOVER(L2) with stable attitude\033[0m");
		}
		break;
	}

	case AUTO_TAKEOFF:
	{
		// kdkd state
		//  if (pub_aoto_takeoff_state_once)
		//  {
		//  	px4_node_state.current_node_state = 4;
		//  	px4_state_to_drone_node_pub.publish(px4_node_state);
		//  	pub_aoto_takeoff_state_once = false;
		//  }
		if ((now_time - takeoff_land.toggle_takeoff_land_time).toSec() < AutoTakeoffLand_t::MOTORS_SPEEDUP_TIME) // Wait for several seconds to warn prople.
		{
			des = get_rotor_speed_up_des(now_time);
		}
		else if (odom_data.p(2) >= (takeoff_land.start_pose(2) + param.takeoff_land.height)) // reach the desired height
		{
			state = AUTO_HOVER;
			set_hov_with_odom();
			ROS_INFO("\033[32m[px4ctrl] AUTO_TAKEOFF --> AUTO_HOVER(L2)\033[32m");

			takeoff_land.delay_trigger.first = true;
			takeoff_land.delay_trigger.second = now_time + ros::Duration(AutoTakeoffLand_t::DELAY_TRIGGER_TIME);
		}
		else
		{
			des = get_takeoff_land_des(param.takeoff_land.speed);
		}

		break;
	}

	case AUTO_LAND:
	{
		// kdkd state
		//  if (pub_land_state_once)
		//  {
		//  	px4_node_state.current_node_state = 6;
		//  	px4_state_to_drone_node_pub.publish(px4_node_state);
		//  	pub_land_state_once = false;
		//  }
		if (!rc_data.is_hover_mode || !odom_is_received(now_time))
		{
			state = MANUAL_CTRL;
			toggle_offboard_mode(false);

			ROS_WARN("[px4ctrl] From AUTO_LAND to MANUAL_CTRL(L1)!");
		}
		else if (!rc_data.is_command_mode)
		{
			state = AUTO_HOVER;
			set_hov_with_odom();
			des = get_hover_des();
			ROS_INFO("[px4ctrl] From AUTO_LAND to AUTO_HOVER(L2)!");
		}
		else if (!get_landed())
		{
			des = get_takeoff_land_des(-param.takeoff_land.speed);
		}
		else
		{
			rotor_low_speed_during_land = true;

			static bool print_once_flag = true;
			if (print_once_flag)
			{
				ROS_INFO("\033[32m[px4ctrl] Wait for abount 10s to let the drone arm.\033[32m");
				print_once_flag = false;
			}

			if (extended_state_data.current_extended_state.landed_state == mavros_msgs::ExtendedState::LANDED_STATE_ON_GROUND) // PX4 allows disarm after this
			{
				static double last_trial_time = 0; // Avoid too frequent calls
				if (now_time.toSec() - last_trial_time > 1.0)
				{
					if (toggle_arm_disarm(false)) // disarm
					{
						print_once_flag = true;
						state = MANUAL_CTRL;
						toggle_offboard_mode(false); // toggle off offboard after disarm
						ROS_INFO("\033[32m[px4ctrl] AUTO_LAND --> MANUAL_CTRL(L1)\033[32m");
					}

					last_trial_time = now_time.toSec();
				}
			}
		}

		break;
	}

	default:
		break;
	}

	// STEP2: estimate thrust model
	if (state == AUTO_HOVER || state == CMD_CTRL || state == MANUAL_CTRL)
	{
		if (param.mode == "classic")
		{
			controller.estimateThrustModel(imu_data.a, bat_data.volt, param);
		}
		else if(param.mode == "neural")
		{
			if (state == CMD_CTRL)
			{
				policy.estimateThrustModel(imu_data.a, bat_data.volt, param);
				controller.thr_scale_compensate = policy.thr_scale_compensate;  // heritage the scale from the other controller
				
				controller.timed_thrust = std::queue<std::pair<ros::Time, double>>();
				std::queue<std::pair<ros::Time, double>> temp_queue_copy = policy.timed_thrust;
				while (!temp_queue_copy.empty())
				{
					controller.timed_thrust.push(temp_queue_copy.front());
					temp_queue_copy.pop();
    			}
			}
			else
			{
				controller.estimateThrustModel(imu_data.a, bat_data.volt, param);
				policy.thr_scale_compensate = controller.thr_scale_compensate;  // heritage the scale from the other controller

				policy.timed_thrust = std::queue<std::pair<ros::Time, double>>();
				std::queue<std::pair<ros::Time, double>> temp_queue_copy = controller.timed_thrust;
				while (!temp_queue_copy.empty())
				{
					policy.timed_thrust.push(temp_queue_copy.front());
					temp_queue_copy.pop();
    			}
			}
		}
	}

	// STEP3: solve and update new control commands
	if (rotor_low_speed_during_land) // used at the start of auto takeoff
	{
		motors_idling(imu_data, u);
	}
	else
	{
		if (param.mode == "classic")
			switch (param.pose_solver)
			{
			case 0:
				debug_msg = controller.update_alg0(des, odom_data, imu_data, u, bat_data.volt);
				debug_msg.header.stamp = now_time;
				debug_pub.publish(debug_msg);
				break;
			case 1:
				debug_msg = controller.update_alg1(des, odom_data, imu_data, u, bat_data.volt);
				debug_msg.header.stamp = now_time;
				debug_pub.publish(debug_msg);
				break;

			case 2:
				controller.update_alg2(des, odom_data, imu_data, u, bat_data.volt);
				break;

			default:
				ROS_ERROR("Illegal pose_slover selection!");
				return;
			}
		else if (param.mode == "neural")
			switch (param.policy_modality)
			{
			case 0: // state-based control
				if (state == CMD_CTRL)
					debug_msg = policy.state_based_control(odom_data, u, bat_data.volt);
				else
					debug_msg = controller.update_alg0(des, odom_data, imu_data, u, bat_data.volt);
				debug_msg.header.stamp = now_time;
				debug_pub.publish(debug_msg);
				break;

			case 1: // pixel-based control
				ROS_ERROR("Not implement yet!");
				debug_msg = controller.update_alg0(des, odom_data, imu_data, u, bat_data.volt);
				debug_msg.header.stamp = now_time;
				debug_pub.publish(debug_msg);
				break;
			
			case 2: // pixel-based HIL control
				if (state == CMD_CTRL && hil_cmd_is_received(now_time))
					debug_msg = policy.pixel_based_HIL_control(odom_data, ctrl_FCU_data, u, bat_data.volt);
				else if (state == ATTI_RECOVER)
					debug_msg = controller.attitude_recovery_control(imu_data, u, bat_data.volt);
				else
					debug_msg = controller.update_alg0(des, odom_data, imu_data, u, bat_data.volt);
				debug_msg.header.stamp = now_time;
				debug_pub.publish(debug_msg);
				break;

			default:
				ROS_ERROR("Illegal policy modality!");
				return;
			}
	}

	// STEP4: publish control commands to mavros
	if (param.use_bodyrate_ctrl)
	{
		publish_bodyrate_ctrl(u, now_time);
		// std::cout<< u.bodyrates.transpose() << std::endl;
		// std::cout<< u.thrust << std::endl;
	}
	else
	{
		publish_attitude_ctrl(u, now_time);
	}

	// STEP5: Detect if the drone has landed
	land_detector(state, des, odom_data);
	// cout << takeoff_land.landed << " ";
	// fflush(stdout);

	// STEP6: Clear flags beyound their lifetime
	rc_data.enter_hover_mode = false;
	rc_data.enter_command_mode = false;
	rc_data.toggle_reboot = false;
	takeoff_land_data.triggered = false;
}

void PX4CtrlFSM::motors_idling(const Imu_Data_t &imu, Controller_Output_t &u)
{
	u.q = imu.q;
	u.bodyrates = Eigen::Vector3d::Zero();
	u.thrust = 0.04;
}

void PX4CtrlFSM::land_detector(const State_t state, const Desired_State_t &des, const Odom_Data_t &odom)
{
	static State_t last_state = State_t::MANUAL_CTRL;
	if (last_state == State_t::MANUAL_CTRL && (state == State_t::AUTO_HOVER || state == State_t::AUTO_TAKEOFF))
	{
		takeoff_land.landed = false; // Always holds
	}
	last_state = state;

	if (state == State_t::MANUAL_CTRL && !state_data.current_state.armed)
	{
		takeoff_land.landed = true;
		return; // No need of other decisions
	}

	// land_detector parameters
	constexpr double POSITION_DEVIATION_C = -0.5; // Constraint 1: target position below real position for POSITION_DEVIATION_C meters.
	constexpr double VELOCITY_THR_C = 0.1;		  // Constraint 2: velocity below VELOCITY_MIN_C m/s.
	constexpr double TIME_KEEP_C = 3.0;			  // Constraint 3: Time(s) the Constraint 1&2 need to keep.

	static ros::Time time_C12_reached; // time_Constraints12_reached
	static bool is_last_C12_satisfy;
	if (takeoff_land.landed)
	{
		time_C12_reached = ros::Time::now();
		is_last_C12_satisfy = false;
	}
	else
	{
		bool C12_satisfy = (des.p(2) - odom.p(2)) < POSITION_DEVIATION_C && odom.v.norm() < VELOCITY_THR_C;
		if (C12_satisfy && !is_last_C12_satisfy)
		{
			time_C12_reached = ros::Time::now();
		}
		else if (C12_satisfy && is_last_C12_satisfy)
		{
			if ((ros::Time::now() - time_C12_reached).toSec() > TIME_KEEP_C) // Constraint 3 reached
			{
				takeoff_land.landed = true;
			}
		}

		is_last_C12_satisfy = C12_satisfy;
	}
}

Desired_State_t PX4CtrlFSM::get_hover_des()
{
	Desired_State_t des;
	des.p = hover_pose.head<3>();
	des.v = Eigen::Vector3d::Zero();
	des.a = Eigen::Vector3d::Zero();
	des.j = Eigen::Vector3d::Zero();
	des.yaw = hover_pose(3);
	des.yaw_rate = 0.0;

	return des;
}

Desired_State_t PX4CtrlFSM::get_cmd_des()
{
	Desired_State_t des;
	des.p = cmd_data.p;
	des.v = cmd_data.v;
	des.a = cmd_data.a;
	des.j = cmd_data.j;
	des.yaw = cmd_data.yaw;
	des.yaw_rate = cmd_data.yaw_rate;

	return des;
}

Desired_State_t PX4CtrlFSM::get_rotor_speed_up_des(const ros::Time now)
{
	double delta_t = (now - takeoff_land.toggle_takeoff_land_time).toSec();
	double des_a_z = exp((delta_t - AutoTakeoffLand_t::MOTORS_SPEEDUP_TIME) * 6.0) * 7.0 - 7.0; // Parameters 6.0 and 7.0 are just heuristic values which result in a saticfactory curve.
	if (des_a_z > 0.1)
	{
		ROS_ERROR("des_a_z > 0.1!, des_a_z=%f", des_a_z);
		des_a_z = 0.0;
	}

	Desired_State_t des;
	des.p = takeoff_land.start_pose.head<3>();
	des.v = Eigen::Vector3d::Zero();
	des.a = Eigen::Vector3d(0, 0, des_a_z);
	des.j = Eigen::Vector3d::Zero();
	des.yaw = takeoff_land.start_pose(3);
	des.yaw_rate = 0.0;

	return des;
}

Desired_State_t PX4CtrlFSM::get_takeoff_land_des(const double speed)
{
	ros::Time now = ros::Time::now();
	double delta_t = (now - takeoff_land.toggle_takeoff_land_time).toSec() - (speed > 0 ? AutoTakeoffLand_t::MOTORS_SPEEDUP_TIME : 0); // speed > 0 means takeoff
	// takeoff_land.last_set_cmd_time = now;

	// takeoff_land.start_pose(2) += speed * delta_t;

	Desired_State_t des;
	des.p = takeoff_land.start_pose.head<3>() + Eigen::Vector3d(0, 0, speed * delta_t);
	des.v = Eigen::Vector3d(0, 0, speed);
	des.a = Eigen::Vector3d::Zero();
	des.j = Eigen::Vector3d::Zero();
	des.yaw = takeoff_land.start_pose(3);
	des.yaw_rate = 0.0;

	return des;
}

void PX4CtrlFSM::set_hov_with_odom()
{
	hover_pose.head<3>() = odom_data.p;
	hover_pose(3) = get_yaw_from_quaternion(odom_data.q);

	last_set_hover_pose_time = ros::Time::now();
}

void PX4CtrlFSM::set_hov_with_odom_for_cross()
{
	// hover_pose.head<3>() = Eigen::Vector3d(1.5, -0.5, 2.0); // x + 0.5m Todo
	hover_pose.head<3>() = param.post_hover_pos;
	hover_pose(3) = get_yaw_from_quaternion(odom_data.q);
	last_set_hover_pose_time = ros::Time::now();

	done = true;
}

void PX4CtrlFSM::set_hov_with_rc()
{
	ros::Time now = ros::Time::now();
	double delta_t = (now - last_set_hover_pose_time).toSec();
	last_set_hover_pose_time = now;

	hover_pose(0) += rc_data.ch[1] * param.max_manual_vel * delta_t * (param.rc_reverse.pitch ? 1 : -1);
	hover_pose(1) += rc_data.ch[0] * param.max_manual_vel * delta_t * (param.rc_reverse.roll ? 1 : -1);
	hover_pose(2) += rc_data.ch[2] * param.max_manual_vel * delta_t * (param.rc_reverse.throttle ? 1 : -1);
	hover_pose(3) += rc_data.ch[3] * param.max_manual_vel * delta_t * (param.rc_reverse.yaw ? 1 : -1);

	if (hover_pose(2) < -0.3)
		hover_pose(2) = -0.3;

	// if (param.print_dbg)
	// {
	// 	static unsigned int count = 0;
	// 	if (count++ % 100 == 0)
	// 	{
	// 		cout << "hover_pose=" << hover_pose.transpose() << endl;
	// 		cout << "ch[0~3]=" << rc_data.ch[0] << " " << rc_data.ch[1] << " " << rc_data.ch[2] << " " << rc_data.ch[3] << endl;
	// 	}
	// }
}

void PX4CtrlFSM::set_start_pose_for_takeoff_land(const Odom_Data_t &odom)
{
	takeoff_land.start_pose.head<3>() = odom_data.p;
	takeoff_land.start_pose(3) = get_yaw_from_quaternion(odom_data.q);

	takeoff_land.toggle_takeoff_land_time = ros::Time::now();
}


// void PX4CtrlFSM::check_inference(const Odom_Data_t &odom)
// {
// 	const float MAX_INFERENCE_TIME_MS = 2.0 / 3.0 * 1000.0 / param.ctrl_freq_max;

// 	const int WINDOW_SIZE = 3 * param.ctrl_freq_max; // 3s statistics
// 	const int MIN_SAMPLES = WINDOW_SIZE;
// 	static std::vector<float> inference_times;
// 	static std::vector<Eigen::Vector4f> outputs;
// 	static float time_sum = 0;
// 	static Eigen::Vector4f output_sum = Eigen::Vector4f::Zero();

// 	static float current_max_time = 0;
// 	static Eigen::Vector4f current_max_cmd = Eigen::Vector4f::Zero();
// 	static Eigen::Vector4f current_min_cmd = Eigen::Vector4f::Zero();

// 	// Convert the state to tensor
// 	// [p, v, euler, last_act] // TODO: check training setup (especially for last_act)
// 	auto start_time = std::chrono::high_resolution_clock::now();
	
// 	torch::NoGradGuard no_grad;

//     // Convert the state to tensor
//     // [p, v, euler, last_act] // TODO: check training setup (especially for last_act)
//     policy.state_eigen.segment<3>(0) = odom.p.cast<float>();
//     policy.state_eigen.segment<3>(3) = odom.v.cast<float>();
//     Eigen::Vector3d euler = quat2euler(odom.q);
//     policy.state_eigen.segment<3>(6) = euler.cast<float>();
//     policy.state_eigen.segment<4>(9) = Eigen::Vector4f::Zero(); // last action
//     policy.state_cpu = torch::from_blob(policy.state_eigen.data(), {1, 13}, torch::kFloat32);  // copy data to tensor

//     // Move to gpu
//     policy.state_gpu.copy_(policy.state_cpu);

//     // Inference
//     policy.action_gpu = policy.inference(policy.state_gpu);  

// 	// Inference
// 	auto future = std::async(std::launch::async, [&] { return policy.inference(policy.state_gpu); });
// 	// torch::Tensor action = policy.inference(state);
// 	auto status = future.wait_for(std::chrono::milliseconds(static_cast<int>(MAX_INFERENCE_TIME_MS)));

// 	if (status == std::future_status::timeout)
// 	{
// 		ROS_ERROR("[px4ctrl] Inference timeout!");
// 		is_unsafe_time = true;
// 		return;
// 	}

// 	torch::Tensor action;
// 	try {
// 		action = future.get().cpu(); // move to cpu
// 	}
// 	catch (const std::exception &e)
// 	{
// 		ROS_ERROR("[px4ctrl] Inference error: %s", e.what());
// 		is_unsafe_time = true;
// 		return;
// 	}
// 	auto end_time = std::chrono::high_resolution_clock::now();
// 	float inference_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

// 	// Convert nn out to control commands
// 	Eigen::Vector4f norm_cmd, cmd;
// 	norm_cmd(0) = action[0][0].item<float>();
// 	norm_cmd(1) = action[0][1].item<float>();
// 	norm_cmd(2) = action[0][2].item<float>();
// 	norm_cmd(3) = action[0][3].item<float>();
// 	cmd = policy.denormalize_cmd(norm_cmd);

// 	policy.last_cmd = cmd;

// 	inference_times.push_back(inference_time);
// 	outputs.push_back(cmd);
// 	time_sum += inference_time;
// 	output_sum += cmd;

// 	if (inference_times.size() > WINDOW_SIZE)
// 	{
// 		time_sum -= inference_times.front();
// 		output_sum -= outputs.front();
// 		inference_times.erase(inference_times.begin());
// 		outputs.erase(outputs.begin());
// 	}

// 	if (inference_times.size() < MIN_SAMPLES)
// 		return;

// 	float avg_time = time_sum / inference_times.size();
// 	Eigen::Vector4f avg_cmd = output_sum / outputs.size();

// 	float max_time = *std::max_element(inference_times.begin(), inference_times.end());

// 	bool output_stable = true;
// 	for (const auto &output : outputs)
// 	{
// 		for (int i = 0; i < output.size(); i++)
// 		{
// 			float ratio = output[i] / avg_cmd[i];
// 			if (ratio < 0.95 || ratio > 1.05)
// 			{
// 				output_stable = false;
// 				break;
// 			}
// 		}
// 		if (!output_stable)
// 			break;
// 	}

// 	if (max_time <= 2 * avg_time && output_stable)
// 	{
// 		is_nn_warmup = true;
// 		ROS_WARN("\033[32[px4ctrl] Check inference done!\033[32");

// 		inference_times.clear();
// 		outputs.clear();
// 		time_sum = 0;
// 		output_sum.setZero();
// 	}
// 	else
// 	{
// 		ROS_INFO("[px4ctrl] Checking inference: %d/%d, inference_time=%4f, avg_time=%.4f, max_time=%.4f, output_stable=%d",
// 				 inference_times.size(), MIN_SAMPLES, avg_time, max_time, output_stable);
// 	}
// }

bool PX4CtrlFSM::rc_is_received(const ros::Time &now_time)
{
	return (now_time - rc_data.rcv_stamp).toSec() < param.msg_timeout.rc;
}

bool PX4CtrlFSM::cmd_is_received(const ros::Time &now_time)
{
	return (now_time - cmd_data.rcv_stamp).toSec() < param.msg_timeout.cmd;
}

bool PX4CtrlFSM::odom_is_received(const ros::Time &now_time)
{
	return (now_time - odom_data.rcv_stamp).toSec() < param.msg_timeout.odom;
}

bool PX4CtrlFSM::imu_is_received(const ros::Time &now_time)
{
	return (now_time - imu_data.rcv_stamp).toSec() < param.msg_timeout.imu;
}

bool PX4CtrlFSM::bat_is_received(const ros::Time &now_time)
{
	return (now_time - bat_data.rcv_stamp).toSec() < param.msg_timeout.bat;
}

bool PX4CtrlFSM::recv_new_odom()
{
	if (odom_data.recv_new_msg)
	{
		odom_data.recv_new_msg = false;
		return true;
	}

	return false;
}

bool PX4CtrlFSM::hil_cmd_is_received(const ros::Time &now_time)
{
	return (now_time - ctrl_FCU_data.rcv_stamp).toSec() < param.msg_timeout.hil_cmd;
}

void PX4CtrlFSM::publish_bodyrate_ctrl(const Controller_Output_t &u, const ros::Time &stamp)
{
	mavros_msgs::AttitudeTarget msg;

	msg.header.stamp = stamp;
	msg.header.frame_id = std::string("FCU");

	msg.type_mask = mavros_msgs::AttitudeTarget::IGNORE_ATTITUDE;

	msg.body_rate.x = u.bodyrates.x();
	msg.body_rate.y = u.bodyrates.y();
	msg.body_rate.z = u.bodyrates.z();

	msg.thrust = u.thrust;

	ctrl_FCU_pub.publish(msg);
}

void PX4CtrlFSM::publish_attitude_ctrl(const Controller_Output_t &u, const ros::Time &stamp)
{
	mavros_msgs::AttitudeTarget msg;

	msg.header.stamp = stamp;
	msg.header.frame_id = std::string("FCU");

	msg.type_mask = mavros_msgs::AttitudeTarget::IGNORE_ROLL_RATE |
					mavros_msgs::AttitudeTarget::IGNORE_PITCH_RATE |
					mavros_msgs::AttitudeTarget::IGNORE_YAW_RATE;

	msg.orientation.x = u.q.x();
	msg.orientation.y = u.q.y();
	msg.orientation.z = u.q.z();
	msg.orientation.w = u.q.w();

	msg.thrust = u.thrust;

	ctrl_FCU_pub.publish(msg);
}

void PX4CtrlFSM::publish_trigger(const nav_msgs::Odometry &odom_msg)
{
	geometry_msgs::PoseStamped msg;
	msg.header.frame_id = "world";
	msg.pose = odom_msg.pose.pose;

	traj_start_trigger_pub.publish(msg);
}

bool PX4CtrlFSM::toggle_offboard_mode(bool on_off)
{
	mavros_msgs::SetMode offb_set_mode;

	if (on_off)
	{
		state_data.state_before_offboard = state_data.current_state;
		if (state_data.state_before_offboard.mode == "OFFBOARD") // Not allowed
			state_data.state_before_offboard.mode = "POSCTL";  // MANUAL POSCTL POSITION

		offb_set_mode.request.custom_mode = "OFFBOARD";
		if (!(set_FCU_mode_srv.call(offb_set_mode) && offb_set_mode.response.mode_sent))
		{
			ROS_ERROR("Enter OFFBOARD rejected by PX4!");
			return false;
		}
	}
	else
	{
		offb_set_mode.request.custom_mode = state_data.state_before_offboard.mode;
		if (!(set_FCU_mode_srv.call(offb_set_mode) && offb_set_mode.response.mode_sent))
		{
			ROS_ERROR("Exit OFFBOARD rejected by PX4!");
			return false;
		}
	}

	return true;

	// if (param.print_dbg)
	// 	printf("offb_set_mode mode_sent=%d(uint8_t)\n", offb_set_mode.response.mode_sent);
}

bool PX4CtrlFSM::toggle_arm_disarm(bool arm)
{
	mavros_msgs::CommandBool arm_cmd;
	arm_cmd.request.value = arm;
	if (!(arming_client_srv.call(arm_cmd) && arm_cmd.response.success))
	{
		if (arm) //{
				 // 	//kdkd
				 // 		px4_node_state.current_node_state = 4;
				 // 		px4_node_state.debug_info = "ARM rejected by PX4!";
				 // 		px4_state_to_drone_node_pub.publish(px4_node_state);

			ROS_ERROR("ARM rejected by PX4!");
		// }
		else
		{
			ROS_ERROR("DISARM rejected by PX4!");
		}

		return false;
	}

	return true;
}

void PX4CtrlFSM::reboot_FCU()
{
	// https://mavlink.io/en/messages/common.html, MAV_CMD_PREFLIGHT_REBOOT_SHUTDOWN(#246)
	mavros_msgs::CommandLong reboot_srv;
	reboot_srv.request.broadcast = false;
	reboot_srv.request.command = 246; // MAV_CMD_PREFLIGHT_REBOOT_SHUTDOWN
	reboot_srv.request.param1 = 1;	  // Reboot autopilot
	reboot_srv.request.param2 = 0;	  // Do nothing for onboard computer
	reboot_srv.request.confirmation = true;

	reboot_FCU_srv.call(reboot_srv);

	ROS_INFO("Reboot FCU");

	// if (param.print_dbg)
	// 	printf("reboot result=%d(uint8_t), success=%d(uint8_t)\n", reboot_srv.response.result, reboot_srv.response.success);
}
