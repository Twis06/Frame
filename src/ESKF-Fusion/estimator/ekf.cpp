#include <estimator/ekf.h>

namespace estimator {
Eigen::Matrix3d analysis_diff_R1R2_wrt_R1(const Eigen::Matrix3d &R1, const Eigen::Matrix3d &R2) {
	return Sophus::SO3d::jr_inv(Sophus::SO3d(R1 * R2).log()) * R2.transpose();
}
// delta Quaternion
constexpr double eps = 1e-6;
constexpr double kEps = 1e-6;
Eigen::Quaterniond deltaQ(const int index) {
  Eigen::Vector3d delta = Eigen::Vector3d::Zero();
  delta(index) = eps;
  Eigen::Matrix<double, 3, 1> half_theta = delta;
  half_theta /= static_cast<double>(2.0);
  Eigen::Quaternion<double> dq(1.0, half_theta.x(), half_theta.y(),
                               half_theta.z());
  return dq;
}

Eigen::Quaterniond deltaQ(Eigen::Vector3d delta) {
  Eigen::Matrix<double, 3, 1> half_theta = delta;
  half_theta /= static_cast<double>(2.0);
  Eigen::Quaternion<double> dq(1.0, half_theta.x(), half_theta.y(),
                               half_theta.z());
  return dq;
}

Eigen::Matrix3d numeric_diff_R_Skew_w_bw_WRT_R(const Eigen::Matrix3d &R, const Eigen::Vector3d w, const Eigen::Vector3d bw) {
  Eigen::Quaterniond q(R * Sophus::SO3d::hat(w - bw));
	q.normalize();
	Eigen::Vector3d result = Sophus::SO3d(q.toRotationMatrix()).log();

  Eigen::Matrix3d jacobian;
  for (size_t i = 0; i < 3; i++) {
    Eigen::Matrix3d pert_R = R * deltaQ(i).toRotationMatrix();
    Eigen::Vector3d pert_R_lie = Sophus::SO3d(Eigen::Quaterniond(pert_R * Sophus::SO3d::hat(w - bw).matrix()).normalized()).log();
    jacobian.col(i) = (pert_R_lie - result) / kEps;
  }
  return jacobian;
}

Eigen::Matrix3d numeric_diff_R_Skew_w_bw_WRT_bw(const Eigen::Matrix3d &R, Eigen::Vector3d w, Eigen::Vector3d bw) {
  Eigen::Quaterniond q(R * Sophus::SO3d::hat(w - bw));
	q.normalize();
	Eigen::Vector3d result = Sophus::SO3d(q.toRotationMatrix()).log();

  Eigen::Matrix3d jacobian;
  for (size_t i = 0; i < 3; i++) {
    Eigen::Vector3d vec_per = Eigen::Vector3d::Zero();
    vec_per(i) = kEps;
    Eigen::Vector3d pert_bw = bw + vec_per;
    Eigen::Vector3d pert_bw_lie = Sophus::SO3d(Eigen::Quaterniond(R * Sophus::SO3d::hat(w - pert_bw).matrix()).normalized()).log();
    jacobian.col(i) = (pert_bw_lie - result) / kEps;
  }
  return jacobian;
}

bool EKFFusion::predict(const common::IMU &imu){
	if (!first_pose_set_) {
		return false;
	}
	if (first_imu_flag_) {
		first_imu_flag_ = false;
		current_time_ = imu.timestamp_;
		last_time_ = current_time_;
		LOG(INFO) << std::fixed << "Receive First Imu " << current_time_;
		return false;
	}

	// update current time
	current_time_ = imu.timestamp_;
	// LOG(INFO) << "Receive Imu " << current_time_ << ", " << x_state_;

	// calculate dt
	double dt = current_time_ - last_time_;

	Eigen::Quaterniond q = x_state_.q_;
	Eigen::Vector3d p = x_state_.p_;
	Eigen::Vector3d v = x_state_.v_;
	Eigen::Vector3d bg = x_state_.bg_;
	Eigen::Vector3d ba = x_state_.ba_;
	Eigen::Vector3d g = /*options_.g_scale_ * */Eigen::Vector3d(0, 0, -kGravity);

	Eigen::Vector3d acc = options_.R_odom_body_ * (imu.acce_ * options_.g_scale_);
	Eigen::Vector3d gyro = options_.R_odom_body_ * imu.gyro_;

	Eigen::Matrix3d R = q.toRotationMatrix();
	Eigen::Matrix3d I = Eigen::Matrix3d::Identity();

	Eigen::MatrixXd Ft(Eigen::MatrixXd::Identity(kErrorStateSize, kErrorStateSize));
	Ft.block<3, 3>(0, 6) = Eigen::Matrix3d::Identity() * dt; 
	// Ft.block<3, 3>(3, 3) = I + numeric_diff_R_Skew_w_bw_WRT_R(R, gyro, bg) * dt; // R/R
	Ft.block<3, 3>(3, 9) = -Eigen::Matrix3d::Identity() * dt;
	// Ft.block<3, 3>(3, 9) = numeric_diff_R_Skew_w_bw_WRT_bw(R, gyro, bg) * dt;    // R/bg
	Ft.block<3, 3>(6, 3) = -q.toRotationMatrix() * SO3::hat(acc - ba) * dt; 
	Ft.block<3, 3>(6, 12) = -q.toRotationMatrix() * dt;
	// Ft.block<3, 3>(3, 3) = (Eigen::Matrix3d::Identity() - SO3::hat(gyro - bg)) * dt;

	Eigen::MatrixXd Bt(Eigen::MatrixXd::Zero(kErrorStateSize, kInputSize));
	Bt.block<3, 3>(3, 0) = -Eigen::Matrix3d::Identity() * dt;
	Bt.block<3, 3>(6, 3) = -q.toRotationMatrix() * dt;
	// Bt.block<3, 3>(3, 0) = numeric_diff_R_Skew_w_bw_WRT_bw(R, gyro, bg) * dt;

	// Eigen::Quaterniond delta_q(1.0, 0.5 * (gyro(0) - bg(0)) * dt, 0.5 * (gyro(1) - bg(1)) * dt, 0.5 * (gyro(2) - bg(2)) * dt);
  // Eigen::Quaterniond upate_q = (q * delta_q).normalized();
	// Eigen::Vector3d p_now = p + v * dt + 0.5 * (q.toRotationMatrix() * (acc - ba)) * dt * dt + 0.5 * g * dt * dt; 
	// Eigen::Vector3d v_now = v + q.toRotationMatrix() * (acc - ba) * dt + g * dt;
	// Eigen::Quaterniond q_now = upate_q;

	Eigen::Vector3d new_p = p + v * dt + 0.5 * (q.toRotationMatrix() * (imu.acce_ - ba)) * dt * dt +
               						0.5 * g * dt * dt;
	Eigen::Vector3d new_v = v + q.toRotationMatrix() * (imu.acce_ - ba) * dt + g * dt;
  Eigen::Matrix3d new_R = q.toRotationMatrix() * SO3::exp((imu.gyro_ - bg) * dt).matrix();
	Eigen::Vector3d p_now = new_p;
	Eigen::Vector3d v_now = new_v;
	Eigen::Quaterniond q_now(new_R);

	q_now.normalize();

	x_state_.p_ = p_now;
	x_state_.v_ = v_now;
	x_state_.q_ = q_now;
	x_state_.R_ = q_now.toRotationMatrix();
	x_state_.timestamp_ = current_time_;

	Pt = Ft * Pt * Ft.transpose() + Bt * Qt * Bt.transpose();

	// std::cout << "g :"   << g.transpose() << std::endl;
	// std::cout << "acc:"  << acc.transpose() << std::endl;
	// std::cout << "gyro:" << gyro.transpose() << std::endl;
	// std::cout << "Qt:\n" << Qt << std::endl;
	// std::cout << "Ft:\n" << Ft << std::endl;
	// std::cout << "Bt:\n" << Bt << std::endl;
	// std::cout << "Pt:\n" << Pt << std::endl;

	last_time_ = current_time_;
	
	// LOG(INFO) << current_time_ << " Predict: " << x_state_;
	return true;
}

bool EKFFusion::observeSE3(const common::NavStated &state){
	if (!first_pose_set_ || first_imu_flag_) {
		first_pose_set_ = true;
		SO3 R = state.GetSE3().so3();
		Eigen::Vector3d p = state.GetSE3().translation();
		x_state_.R_ = R;
		x_state_.p_ = p;
		x_state_.q_ = R.unit_quaternion();
		LOG(INFO) << "First Pose Set, pos: " << x_state_.p_.transpose()
						  << " RPY: " << common::RotationUtility::R2ypr(R.matrix()).transpose();
		return true;
	}

	SE3 obs_pose = state.GetSE3();
	Eigen::MatrixXd Ct(Eigen::MatrixXd::Zero(kPoseMeasureSize - 1, kErrorStateSize));
	Eigen::MatrixXd Wt(Eigen::MatrixXd::Identity(kPoseMeasureSize - 1, kPoseMeasureSize - 1));

	Ct.block<3, 3>(0, 0) = Eigen::MatrixXd::Identity(3, 3);
	Ct.block<3, 3>(3, 3) = Eigen::MatrixXd::Identity(3, 3);

	Eigen::MatrixXd Kt = Eigen::MatrixXd::Identity(kErrorStateSize, kPoseMeasureSize - 1); 
	Kt = Pt * Ct.transpose() * (Ct * Pt * Ct.transpose() + Wt * Rt * Wt.transpose()).inverse();

	Eigen::VectorXd innovation = Eigen::VectorXd::Zero(kPoseMeasureSize - 1);
	innovation.segment<3>(0) = obs_pose.translation() - x_state_.p_;
	innovation.segment<3>(3) = SO3(x_state_.q_.inverse() * obs_pose.unit_quaternion()).log();

	Eigen::VectorXd dx = Eigen::VectorXd::Zero(kErrorStateSize);
	dx = Kt * innovation;

	// std::cout << "Pt: "         << Pt.rows() << ", " << Pt.cols() << "\n" << Pt << std::endl;
	// std::cout << "Rt: "         << Rt.rows() << ", " << Rt.cols() << "\n" << Rt << std::endl;
	// std::cout << "Ct: "         << Ct.rows() << ", " << Ct.cols() << "\n" << Ct << std::endl;
	// std::cout << "Wt: "         << Wt.rows() << ", " << Wt.cols() << "\n" << Wt << std::endl;
	// std::cout << "Kalman Gain " << Kt.rows() << ", " << Kt.cols() << "\n" << Kt << std::endl;
	// std::cout << "Error:  "     << innovation.rows() << "\n"   << innovation.transpose() << std::endl;
	// std::cout << "Dx:  "        << dx.rows() << "\n" << dx.transpose() << std::endl;

	updateState(dx);

	Pt = Pt - Kt * Ct * Pt;

	// LOG(INFO) << "Update: " << x_state_;
	// std::cout << "Update Pt: " << Pt.rows() << ", " << Pt.cols() << "\n" << Pt << std::endl;

	return true;
}

bool EKFFusion::observeSpeedWorld(const Eigen::Vector3d &vel_w) {
	if (!first_pose_set_) {
		return false;
	}

	Eigen::MatrixXd Ct(Eigen::MatrixXd::Zero(kVelMeasureSize, kErrorStateSize));
	Eigen::MatrixXd Wt(Eigen::MatrixXd::Identity(kVelMeasureSize, kVelMeasureSize));

	Ct.block<3, 3>(0, 6) = Eigen::MatrixXd::Identity(3, 3);

	Eigen::MatrixXd Kt = Eigen::MatrixXd::Identity(kErrorStateSize, kVelMeasureSize);
	Kt = Pt * Ct.transpose() * (Ct * Pt * Ct.transpose() + Wt * Rt_vel * Wt.transpose()).inverse();

	Eigen::VectorXd innovation = Eigen::VectorXd::Zero(kVelMeasureSize);
	innovation.segment<3>(0) = vel_w - x_state_.v_;
	Eigen::VectorXd dx = Kt * innovation;

	updateState(dx);

	Pt = Pt - Kt * Ct * Pt;

	return true;
}

bool EKFFusion::observeSpeedBodyXY(const Eigen::Vector2d &vel_b_xy) {
	if (!first_pose_set_) {
		return false;
	}
	Eigen::Vector2d vel_b_xy_state = (x_state_.R_.inverse() * x_state_.v_).head<2>();

	Eigen::MatrixXd Ct(Eigen::MatrixXd::Zero(kVelMeasureSize - 1, kErrorStateSize));
	Eigen::MatrixXd Wt(Eigen::MatrixXd::Identity(kVelMeasureSize - 1, kVelMeasureSize - 1));
	// Ct.block<2, 2>(0, 6) = Eigen::MatrixXd::Identity(2, 2);

	Eigen::Matrix<double, 2, 3> H_rot;
	for (int i = 0; i < 3; i++) {
		Eigen::Vector3d delta = Eigen::Vector3d::Zero();
		delta(i) = eps;
		Eigen::Matrix3d dR = SO3::exp(delta).matrix();
		Eigen::Vector3d vel_b_perturbed = (x_state_.R_.matrix() * dR).inverse() * x_state_.v_;
		H_rot.col(i) = (vel_b_perturbed.head<2>() - vel_b_xy_state) / eps;
	}

	Eigen::Matrix3d dvel_b_dR = SO3::hat(x_state_.R_.inverse() * x_state_.v_);
	// std::cout << "H_rot: \n" << H_rot << std::endl;
	// std::cout << "dvel_b_dR: \n" << dvel_b_dR << std::endl;

	Ct.block<2, 3>(0, 3) = H_rot;
	Ct.block<2, 3>(0, 6) = x_state_.R_.inverse().matrix().topRows<2>();
	
	Eigen::MatrixXd Kt = Eigen::MatrixXd::Identity(kErrorStateSize, kVelMeasureSize - 1);
	Kt = Pt * Ct.transpose() * (Ct * Pt * Ct.transpose() + Wt * Rt_vel.block<2, 2>(0, 0) * Wt.transpose()).inverse();

	Eigen::VectorXd innovation = Eigen::VectorXd::Zero(kVelMeasureSize - 1);
	innovation.segment<2>(0) = vel_b_xy - vel_b_xy_state;
	Eigen::VectorXd dx = Kt * innovation;

	updateState(dx);

	Pt = Pt - Kt * Ct * Pt;

	return true;
}

bool EKFFusion::observeSpeedWorld(const Eigen::Vector2d &vel_w_xy) {
	if (!first_pose_set_) {
		return false;
	}

	Eigen::MatrixXd Ct(Eigen::MatrixXd::Zero(kVelMeasureSize - 1, kErrorStateSize));
	Eigen::MatrixXd Wt(Eigen::MatrixXd::Identity(kVelMeasureSize - 1, kVelMeasureSize - 1));

	Ct.block<2, 2>(0, 6) = Eigen::MatrixXd::Identity(2, 2);

	Eigen::MatrixXd Kt = Eigen::MatrixXd::Identity(kErrorStateSize, kVelMeasureSize - 1);
	Kt = Pt * Ct.transpose() * (Ct * Pt * Ct.transpose() + Wt * Rt_vel.block<2, 2>(0, 0) * Wt.transpose()).inverse();

	Eigen::VectorXd innovation = Eigen::VectorXd::Zero(kVelMeasureSize - 1);
	innovation.segment<2>(0) = vel_w_xy - x_state_.v_.segment<2>(0);
	Eigen::VectorXd dx = Kt * innovation;

	updateState(dx);

	Pt = Pt - Kt * Ct * Pt;

	return true;
}

bool EKFFusion::observeSpeedWorld(const double &vel_w_z) {
	if (!first_pose_set_) {
		return false;
	}

	Eigen::MatrixXd Ct(Eigen::MatrixXd::Zero(1, kErrorStateSize));
	Eigen::MatrixXd Wt(Eigen::MatrixXd::Identity(1, 1));

	// Ct.block<1, 1>(2, 8) = Eigen::MatrixXd::Identity(1, 1);
	Ct(0, 8) = 1.0;

	Eigen::MatrixXd Kt = Eigen::MatrixXd::Identity(kErrorStateSize, kVelMeasureSize - 1);
	Kt = Pt * Ct.transpose() * (Ct * Pt * Ct.transpose() + Wt * Rt_vel.block<1, 1>(2, 2) * Wt.transpose()).inverse();

	Eigen::VectorXd innovation = Eigen::VectorXd::Zero(1);
	innovation(0) = vel_w_z - x_state_.v_.z();
	Eigen::VectorXd dx = Kt * innovation;

	updateState(dx);

	Pt = Pt - Kt * Ct * Pt;

	return true;
}

bool EKFFusion::updateState(const Eigen::VectorXd &dx) {
	Eigen::Matrix3d dR = SO3::exp(dx.segment<3>(3)).matrix();

	x_state_.p_ += dx.segment<3>(0);
	x_state_.q_ = x_state_.q_ * dR;
	x_state_.q_.normalize();
	x_state_.R_ = x_state_.q_.toRotationMatrix();

	x_state_.v_ += dx.segment<3>(6);
	x_state_.bg_ += dx.segment<3>(9);
	x_state_.ba_ += dx.segment<3>(12);

	return true;
}

void EKFFusion::setInitOptions(const EKFFusion::Options& options) {
  options_ = options;
  buildNoise();

	x_state_.q_.setIdentity();

}

void EKFFusion::buildNoise() {
	double position_cov = options_.lvio_pos_noise_;
	double rot_cov = options_.lvio_ang_noise_;

	double acc_cov = options_.acce_var_;
	double gyro_cov = options_.gyro_var_;
	// double eg = options.bias_gyro_var_;
	// double ea = options.bias_acce_var_;

	Pt = Eigen::MatrixXd::Identity(kErrorStateSize, kErrorStateSize); 
	// Pt.block<3, 3>(0, 0) = 0.0001 * Pt.block<3, 3>(0, 0);
	// Pt.block<3, 3>(3, 3) = 0.0000001 * Pt.block<3, 3>(3, 3);
	// Pt.block<3, 3>(6, 6) = 0.000001 * Pt.block<3, 3>(6, 6);
	// Pt.block<3, 3>(9, 9) = 0.000001 * Pt.block<3, 3>(9, 9);
	// Pt.block<3, 3>(12, 12) = 0.00001 * Pt.block<3, 3>(12, 12);

	Pt.block<3, 3>(0, 0) = options_.pos_priori_cov * Pt.block<3, 3>(0, 0);
	Pt.block<3, 3>(3, 3) = options_.rot_priori_cov * Pt.block<3, 3>(3, 3);
	Pt.block<3, 3>(6, 6) = options_.vel_priori_cov * Pt.block<3, 3>(6, 6);
	Pt.block<3, 3>(9, 9) = options_.bg_bias_priori_cov * Pt.block<3, 3>(9, 9);
	Pt.block<3, 3>(12, 12) = options_.ba_bias_priori_cov * Pt.block<3, 3>(12, 12);

	Qt = Eigen::MatrixXd::Identity(6, 6); 
  Rt = Eigen::MatrixXd::Identity(kPoseMeasureSize - 1, kPoseMeasureSize - 1); 
	Rt_vel = Eigen::MatrixXd::Identity(kVelMeasureSize, kVelMeasureSize);      

	Rt_vel.topLeftCorner(3, 3) = options_.vel_xy_noise_ * Rt_vel.topLeftCorner(3, 3);
	Rt_vel(2, 2) = options_.vel_z_noise_;

  Qt.topLeftCorner(3, 3) = gyro_cov * Qt.topLeftCorner(3, 3);
  Qt.bottomRightCorner(3, 3) = acc_cov * Qt.bottomRightCorner(3, 3);
  Rt.topLeftCorner(3, 3) = position_cov * Rt.topLeftCorner(3, 3);
  Rt.bottomRightCorner(3, 3) = rot_cov * Rt.bottomRightCorner(3, 3);
  Rt.bottomRightCorner(1, 1) = rot_cov * Rt.bottomRightCorner(1, 1);
	LOG(INFO) << GREEN "acc var "      << options_.acce_var_       << ", gyro var "      << options_.gyro_var_ << TAIL;
	LOG(INFO) << GREEN "acc bias var " << options_.bias_acce_var_  << ", gyro bias var " << options_.bias_gyro_var_ << TAIL;
	LOG(INFO) << GREEN "position var " << options_.lvio_pos_noise_ << ", rotation var "  << options_.lvio_ang_noise_ << TAIL;
	LOG(INFO) << GREEN "g scale "      << options_.g_scale_ << TAIL;

	LOG(INFO) << GREEN "Pt:\n" << Pt << TAIL;
	LOG(INFO) << GREEN "Qt:\n" << Qt << TAIL;
	LOG(INFO) << GREEN "Rt:\n" << Rt << TAIL;
	LOG(INFO) << GREEN "Rt_vel:\n" << Rt_vel << TAIL;
}

}  // namespace estimator