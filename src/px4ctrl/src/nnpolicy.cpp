/*************************************************************/
/* Acknowledgement: @wdww */
/*************************************************************/

#include "nnpolicy.h"

#include <algorithm>
using namespace uav_utils;

NNPolicy::NNPolicy(Parameter_t &param_) : param(param_),  is_initialized(false), device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU)
{
    if (torch::cuda::is_available()) {
        ROS_WARN("[px4ctrl] CUDA is available. Use GPU for inference.");
    } else {
        ROS_WARN("[px4ctrl] CUDA is not available. Use CPU for inference.");
    }

    last_cmd = Eigen::Vector4f::Zero();  // TODO: hover cmd
}

void NNPolicy::warmup_step(bool print)
{
    if (param.policy_modality == 0)
    {
        std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
        
        torch::Tensor state = torch::zeros({13}, torch::kFloat32);
        state[0] = -4.0;
        state[1] = 0.0;
        state[2] = 1.5;
        state = state.unsqueeze(0);  
        state = state.to(device);

        torch::Tensor action = inference(state);  

        action = action.cpu(); 
        Vector4f action_eigen;
		std::memcpy(action_eigen.data(), action.data_ptr(), 4 * sizeof(float));

        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        if (print)
        {
            ROS_INFO("[px4ctrl] Warmup inference duration=%.4f ms", duration.count());
            if (duration.count() > 2. / 3. * 1000. / param.ctrl_freq_max)
                ROS_WARN("[px4ctrl] Inference time is too long! The first few rounds of inference may take much longer time.");
        }
            
    }
    
}

quadrotor_msgs::Px4ctrlDebug NNPolicy::state_based_control(
    const Odom_Data_t &odom,
    Controller_Output_t &u,
    double voltage)
{
    quadrotor_msgs::Px4ctrlDebug debug;

    if (!is_initialized)
    {
        ROS_FATAL("[px4ctrl] Model is not initialized. Shutting down the node.");
        ros::shutdown();
        return debug;
    }

    torch::NoGradGuard no_grad;

    // Convert the state to tensor
    // [p, v, euler, last_act] 
    state_eigen.segment<3>(0) = odom.p.cast<float>();
    state_eigen.segment<3>(3) = odom.v.cast<float>();
    Eigen::Vector3d euler = quat2euler(odom.q);
    state_eigen.segment<3>(6) = euler.cast<float>();
    state_eigen.segment<4>(9) = last_cmd.cast<float>();
    state_cpu = torch::from_blob(state_eigen.data(), {1, 13}, torch::kFloat32);  // copy data to tensor
    
    // Move to gpu
    state_gpu.copy_(state_cpu);

    // Inference
    action_gpu = inference(state_gpu);  

    // Move to cpu
    action_cpu.copy_(action_gpu);
    std::memcpy(action_eigen.data(), action_cpu.data_ptr(), 4 * sizeof(float));

    // Update last act
    last_cmd = action_eigen;

    // Convert nn out to control commands
    denormalize_cmd(action_eigen);

    double des_acc_norm = static_cast<double>(action_eigen(0));
    u.thrust = computeDesiredCollectiveThrustSignal(des_acc_norm, param, voltage);
    u.bodyrates.x() = static_cast<double>(action_eigen(1));
    u.bodyrates.y() = static_cast<double>(action_eigen(2));
    u.bodyrates.z() = static_cast<double>(action_eigen(3));

    // Debug
    debug.des_q_w = odom.q.w();
    debug.des_q_x = odom.q.x();
    debug.des_q_y = odom.q.y();
    debug.des_q_z = odom.q.z();
    debug.des_thr = des_acc_norm;
    debug.fb_rate_x = u.bodyrates.x();
    debug.fb_rate_y = u.bodyrates.y();
    debug.fb_rate_z = u.bodyrates.z();
    debug.des_v_x = odom.v(0);
    debug.des_v_y = odom.v(1);
    debug.des_v_z = odom.v(2);
    debug.thr_scale_compensate = thr_scale_compensate; //debug
    debug.u_thrust = u.thrust;

    // Used for thrust-accel mapping estimation
    timed_thrust.push(std::pair<ros::Time, double>(ros::Time::now(), u.thrust));
    while (timed_thrust.size() > 100)
      timed_thrust.pop();

    return debug;
}

quadrotor_msgs::Px4ctrlDebug NNPolicy::pixel_based_HIL_control(
    const Odom_Data_t &odom,
    Controller_Output_t &u_hil,
    Controller_Output_t &u,
    double voltage)
{
    quadrotor_msgs::Px4ctrlDebug debug;

    // if (!is_initialized)
    // {
    //     ROS_FATAL("[px4ctrl] Model is not initialized. Shutting down the node.");
    //     ros::shutdown();
    //     return debug;
    // }

    // Transform from HIL cmd to executable cmd
    double des_acc_norm = u_hil.thrust;
    u.thrust = computeDesiredCollectiveThrustSignal(des_acc_norm, param, voltage);
    u.bodyrates.x() = u_hil.bodyrates.x();
    u.bodyrates.y() = u_hil.bodyrates.y();
    u.bodyrates.z() = u_hil.bodyrates.z();

    // Debug
    debug.des_q_w = odom.q.w();
    debug.des_q_x = odom.q.x();
    debug.des_q_y = odom.q.y();
    debug.des_q_z = odom.q.z();
    debug.des_thr = des_acc_norm;
    debug.fb_rate_x = u.bodyrates.x();
    debug.fb_rate_y = u.bodyrates.y();
    debug.fb_rate_z = u.bodyrates.z();
    debug.des_v_x = odom.v(0);
    debug.des_v_y = odom.v(1);
    debug.des_v_z = odom.v(2);
    debug.thr_scale_compensate = thr_scale_compensate; //debug
    debug.u_thrust = u.thrust;
    debug.voltage = voltage;

    // Used for thrust-accel mapping estimation
    timed_thrust.push(std::pair<ros::Time, double>(ros::Time::now(), u.thrust));
    while (timed_thrust.size() > 100)
      timed_thrust.pop();

    return debug;
}

torch::Tensor NNPolicy::inference(const torch::Tensor state)
{
    torch::NoGradGuard no_grad;
    return model.forward({state}).toTensor().tanh();
}

quadrotor_msgs::Px4ctrlDebug NNPolicy::attitude_recovery_control(
    const Imu_Data_t &imu,
    Controller_Output_t &u,
    double voltage)
{
    double yaw = get_yaw_from_quaternion(imu.q);
    Eigen::Vector3d des_ypr(yaw, 0.0, 0.0);
    Eigen::Quaterniond desired_q = ypr_to_quaternion(des_ypr);
    u.q = desired_q;

    // reactive control: solve the problem min(6<=u<=14){-alpha*|T_v - g|-|T_h|}
    double cos_theta = imu.q.w() * imu.q.w() - imu.q.x() * imu.q.x() - imu.q.y() * imu.q.y() + imu.q.z() * imu.q.z();
    double sin_theta = 2 * sqrt(imu.q.x()*imu.q.x()*imu.q.y()*imu.q.y() + imu.q.w()*imu.q.w()*imu.q.z()*imu.q.z() + imu.q.x()*imu.q.x()*imu.q.z()*imu.q.z() + imu.q.w()*imu.q.w()*imu.q.y()*imu.q.y());
    double T_v_max = 14.0 * cos_theta;
    double T_v_min = 6.0 * cos_theta;
    double T_h_max = 14.0 * sin_theta;
    double T_h_min = 6.0 * sin_theta;
    double T_z_balance = std::max(6.0, std::min(14.0, 9.81 / cos_theta));
    double T_v_balance = T_z_balance * cos_theta;
    double T_h_balance = T_z_balance * sin_theta;

    double alpha = 4.0; // object weight
    double object_T_max = -alpha * abs(T_v_max - 9.81) - abs(T_h_max);
    double object_T_min = -alpha * abs(T_v_min - 9.81) - abs(T_h_min);
    double object_T_balance = -alpha * abs(T_v_balance - 9.81) - abs(T_h_balance);

    double des_acc_norm;
    if (object_T_max > object_T_min)
    {
        if (object_T_max > object_T_balance)
        {
            des_acc_norm = 14.0;
        }
        else
        {
            des_acc_norm = T_z_balance;
        }
    }
    else
    {
        if (object_T_min > object_T_balance)
        {
            des_acc_norm = 6.0;
        }
        else
        {
            des_acc_norm = T_z_balance;
        }
    }
    u.thrust = computeDesiredCollectiveThrustSignal(des_acc_norm, param, voltage);

    quadrotor_msgs::Px4ctrlDebug debug;
    debug.des_q_w = desired_q.w();
    debug.des_q_x = desired_q.x();
    debug.des_q_y = desired_q.y();
    debug.des_q_z = desired_q.z();
    debug.des_thr = des_acc_norm;
    debug.thr_scale_compensate = thr_scale_compensate; //debug
    debug.u_thrust = u.thrust;
    debug.voltage = voltage;

    // Used for thrust-accel mapping estimation
    timed_thrust.push(std::pair<ros::Time, double>(ros::Time::now(), u.thrust));
    while (timed_thrust.size() > 100)
      timed_thrust.pop();

    return debug;
}

double NNPolicy::computeDesiredCollectiveThrustSignal(
    const double &des_acc_norm,
    const Parameter_t &param,
    double voltage)
{
    // This compensates for an acceleration component in thrust direction due
    // to the square of the body-horizontal velocity. // TODO: add drag
    // des_acc_norm -= param.rt_drag.k_thrust_horz * (pow(est_v.x(), 2.0) + pow(est_v.y(), 2.0));
    double normalized_thrust;

    debug.des_thr = des_acc_norm; //debug

    if (param.thr_map.accurate_thrust_model)
    {
        normalized_thrust = param.thr_map.accurate_thrust_scale * 1.06 * AccurateThrustAccMapping(des_acc_norm, voltage, param);
        // normalized_thrust = AccurateThrustAccMapping(des_acc_norm, voltage, param);
    }
    else
    {
        ROS_ERROR("[px4ctrl] Nn policy must use accurate thrust mapping!");
        normalized_thrust = -100.0; // TODO: error handling
    }

    return normalized_thrust;
}



double NNPolicy::AccurateThrustAccMapping(
    const double des_acc_z,
    double voltage,
    const Parameter_t &param) const
{
  if (voltage < param.low_voltage)
  {
    voltage = param.low_voltage;
    ROS_ERROR("Low voltage!");
  }
  if (voltage > 1.5 * param.low_voltage)
  {
    voltage = 1.5 * param.low_voltage;
  }

  // F=K1*Voltage^K2*(K3*u^2+(1-K3)*u)
  double a = param.thr_map.K3;
  double b = 1 - param.thr_map.K3;
  double c = -(param.mass * des_acc_z) / (param.thr_map.K1 * pow(voltage, param.thr_map.K2));
  double b2_4ac = pow(b, 2) - 4 * a * c;
  if (b2_4ac <= 0)
    b2_4ac = 0;
  double thrust = (-b + sqrt(b2_4ac)) / (2 * a);
  // if (thrust <= 0) thrust = 0; // This should be avoided before calling this function
  return thrust;
}

bool NNPolicy::estimateThrustModel(
    const Eigen::Vector3d &est_a,
    const double voltage,
    const Parameter_t &param)
{

  ros::Time t_now = ros::Time::now();
  while (timed_thrust.size() >= 1)
  {
    // Choose data before 35~45ms ago
    std::pair<ros::Time, double> t_t = timed_thrust.front();
    double time_passed = (t_now - t_t.first).toSec();
    if (time_passed > 0.045) // 45ms
    {
      // printf("[estimate_scale] Continue, time_passed=%f\n", time_passed);
      timed_thrust.pop();
      continue;
    }
    if (time_passed < 0.035) // 35ms
    {
      // printf("[estimate_scale] skip, time_passed=%f\n", time_passed);
      return false;
    }

    /***********************************************************/
    /* Recursive least squares algorithm with vanishing memory */
    /***********************************************************/
    double thr = t_t.second;
    timed_thrust.pop();
    if (param.thr_map.accurate_thrust_model)
    {
      /**************************************************************************/
      /* Model: thr = thr_scale_compensate * AccurateThrustAccMapping(est_a(2)) */
      /**************************************************************************/
      double thr_fb = AccurateThrustAccMapping(est_a(2), voltage, param);
      double gamma = 1 / (rho2 + thr_fb * P * thr_fb);
      double K = gamma * P * thr_fb;
      thr_scale_compensate = thr_scale_compensate + K * (thr - thr_fb * thr_scale_compensate);
      P = (1 - K * thr_fb) * P / rho2;
      // printf("%6.3f,%6.3f,%6.3f,%6.3f\n", thr_scale_compensate, gamma, K, P);
      // fflush(stdout);

      if (thr_scale_compensate > 1.15 || thr_scale_compensate < 0.85)
      {
        // ROS_ERROR("Thrust scale = %f, which shoule around 1. It means the thrust model is nolonger accurate. \
                  Re-calibrate the thrust model!",
                  // thr_scale_compensate);
        thr_scale_compensate = thr_scale_compensate > 1.15 ? 1.15 : thr_scale_compensate;
        thr_scale_compensate = thr_scale_compensate < 0.85 ? 0.85 : thr_scale_compensate;
      }

      debug.thr_scale_compensate = thr_scale_compensate; //debug
      debug.voltage = voltage;
      if (param.thr_map.print_val)
      {
        ROS_WARN("thr_scale_compensate = %f", thr_scale_compensate);
      }
    }
    else
    {
      /***********************************/
      /* Model: est_a(2) = thr2acc * thr */
      /***********************************/
      double gamma = 1 / (rho2 + thr * P * thr);
      double K = gamma * P * thr;
      thr2acc = thr2acc + K * (est_a(2) - thr * thr2acc);
      P = (1 - K * thr) * P / rho2;
      //printf("%6.3f,%6.3f,%6.3f,%6.3f\n", thr2acc, gamma, K, P);
      //fflush(stdout);
      const double hover_percentage = param.gra / thr2acc;
      if (hover_percentage > 0.8 || hover_percentage < 0.1)
      {
        ROS_ERROR("Estimated hover_percentage >0.8 or <0.1! Perhaps the accel vibration is too high!");
        thr2acc = hover_percentage > 0.8 ? param.gra / 0.8 : thr2acc;
        thr2acc = hover_percentage < 0.1 ? param.gra / 0.1 : thr2acc;
      }
      debug.hover_percentage = hover_percentage; // debug
      if (param.thr_map.print_val)
      {
        ROS_WARN("hover_percentage = %f", debug.hover_percentage);
      }
    }

    return true;
  }

  return false;
}

void NNPolicy::resetThrustMapping(void)
{
  thr2acc = param.gra / param.thr_map.hover_percentage;
  thr_scale_compensate = 1.0;
  P = 1e6;
}
