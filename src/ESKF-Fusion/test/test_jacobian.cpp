#include <iostream>
#include <sophus/se3.hpp>

constexpr double kEps = 1e-6;

Eigen::Quaterniond deltaQ(const int index) {
  Eigen::Vector3d delta = Eigen::Vector3d::Zero();
  delta(index) = kEps;
  Eigen::Matrix<double, 3, 1> half_theta = delta;
  half_theta /= static_cast<double>(2.0);
  Eigen::Quaternion<double> dq(1.0, half_theta.x(), half_theta.y(),
                               half_theta.z());
  return dq;
}

Eigen::Matrix3d numeric_diff_R1R2_wrt_R1(Eigen::Matrix3d &R1, Eigen::Matrix3d &R2) {
  Sophus::SO3d result(R1 * R2);
  Eigen::Matrix3d diff_R1 = Eigen::Matrix3d::Zero();
  double eps = 1e-6;
  for (int i = 0; i < 3; ++i) {
    Eigen::Matrix3d pert_R1_plus_eps = R1 * deltaQ(i).toRotationMatrix();
    diff_R1.col(i) = (Sophus::SO3d((pert_R1_plus_eps * R2)).log() - result.log()) / eps;
  }
  return diff_R1;
}

Eigen::Matrix3d analysis_diff_R1R2_wrt_R1(Eigen::Matrix3d &R1, Eigen::Matrix3d &R2) {
  return Sophus::SO3d::jr_inv(Sophus::SO3d(R1 * R2).log()) * R2.transpose();
}

Eigen::Matrix3d numeric_diff_R_Exp_w_bw_WRT_R(Eigen::Matrix3d &R, Eigen::Vector3d w, Eigen::Vector3d bw) {
  Eigen::Vector3d result = Sophus::SO3d(R * Sophus::SO3d::exp(w - bw).matrix()).log();

  Eigen::Matrix3d jacobian;
  for (size_t i = 0; i < 3; i++) {
    Eigen::Matrix3d pert_R = R * deltaQ(i).toRotationMatrix();
    Eigen::Vector3d pert_R_lie = Sophus::SO3d(pert_R * Sophus::SO3d::exp(w - bw).matrix()).log();
    jacobian.col(i) = (pert_R_lie - result) / kEps;
  }
  return jacobian;
}

Eigen::Matrix3d numeric_diff_R_Skew_w_bw_WRT_R(Eigen::Matrix3d &R, Eigen::Vector3d w, Eigen::Vector3d bw) {
  Eigen::Vector3d result = Sophus::SO3d(R * Sophus::SO3d::hat(w - bw)).log();

  Eigen::Matrix3d jacobian;
  for (size_t i = 0; i < 3; i++) {
    Eigen::Matrix3d pert_R = R * deltaQ(i).toRotationMatrix();
    Eigen::Vector3d pert_R_lie = Sophus::SO3d(pert_R * Sophus::SO3d::hat(w - bw).matrix()).log();
    jacobian.col(i) = (pert_R_lie - result) / kEps;
  }
  return jacobian;
}

Eigen::Matrix3d numeric_diff_R_Skew_w_bw_WRT_bw(Eigen::Matrix3d &R, Eigen::Vector3d w, Eigen::Vector3d bw) {
  Eigen::Vector3d result = Sophus::SO3d(R * Sophus::SO3d::hat(w - bw)).log();

  Eigen::Matrix3d jacobian;
  for (size_t i = 0; i < 3; i++) {
    Eigen::Vector3d vec_per = Eigen::Vector3d::Zero();
    vec_per(i) = kEps;
    Eigen::Vector3d pert_bw = bw + vec_per;
    Eigen::Vector3d pert_bw_lie = Sophus::SO3d(R * Sophus::SO3d::hat(w - pert_bw).matrix()).log();
    jacobian.col(i) = (pert_bw_lie - result) / kEps;
  }
  return jacobian;
}

Eigen::Matrix3d analysis_diff_R_Exp_w_bw_WRT_R(Eigen::Matrix3d &R, Eigen::Vector3d w, Eigen::Vector3d bw) {
  Eigen::Matrix3d jacobian;
  Eigen::Matrix3d R1 = R;
  Eigen::Matrix3d R2 = Sophus::SO3d::exp(w - bw).matrix();
  jacobian = analysis_diff_R1R2_wrt_R1(R1, R2);
  return jacobian;
}

int main(int argc, char const *argv[])
{
  {
    std::cout << "\n=========numeric_diff_R1R2_wrt_R1=========\n";
    Eigen::Matrix3d R1 = Eigen::Quaterniond::UnitRandom().toRotationMatrix();
    Eigen::Matrix3d R2 = Eigen::Quaterniond::UnitRandom().toRotationMatrix();
    std::cout << "R1:\n" << R1 << std::endl;
    std::cout << "R2:\n" << R2 << std::endl;

    Eigen::Matrix3d numeric_diff_R1 = numeric_diff_R1R2_wrt_R1(R1, R2);
    Eigen::Matrix3d analysis_diff_R1 = analysis_diff_R1R2_wrt_R1(R1, R2);
    std::cout << "Numeric diff R1:\n" << numeric_diff_R1 << std::endl;
    std::cout << "Analysis diff R1:\n" << analysis_diff_R1 << std::endl;
  }
  {
    std::cout << "\n=========numeric_diff_R_Exp_w_bw_WRT_R=========\n";
    Eigen::Matrix3d R1 = Eigen::Quaterniond::UnitRandom().toRotationMatrix();
    Eigen::Vector3d w = Eigen::Vector3d::Random();
    Eigen::Vector3d bw = Eigen::Vector3d::Random();

    Eigen::Matrix3d numerif_diff = numeric_diff_R_Exp_w_bw_WRT_R(R1, w, bw);
    Eigen::Matrix3d analysis_diff = analysis_diff_R_Exp_w_bw_WRT_R(R1, w, bw);
    std::cout << "numerif_diff:\n" << numerif_diff << std::endl;
    std::cout << "analysis_diff:\n" << analysis_diff << std::endl;
  }

  return 0;
}
