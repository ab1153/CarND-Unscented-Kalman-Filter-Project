#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 1;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */
  time_us_ = 0;

  n_x_ = 5;
  // Augmented state dimension
  n_aug_ = 7;

  lambda_ = 3 - n_aug_;




  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  weights_ = VectorXd::Zero(2 * n_aug_ + 1);
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  weights_.tail(2 * n_aug_).fill(0.5 / (n_aug_ + lambda_));

  H_laser_ = MatrixXd(2, n_x_);
  H_laser_ << 1, 0, 0, 0, 0,
    0, 1, 0, 0, 0;

  R_laser_ = MatrixXd(2, 2);
  R_laser_ << std_laspx_ * std_laspx_, 0,
    0, std_laspy_ * std_laspy_;

  is_initialized_ = false;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */
  if (!is_initialized_) {
    init(meas_package);
    is_initialized_ = true;
    return;
  }

  double dt = meas_package.timestamp_ - time_us_;

  double xy = x_(0) * x_(0) + x_(1) * x_(1);
  if (dt < 0.001 || xy < 0.001) {
    return;
  }

  time_us_ = meas_package.timestamp_;

  Prediction(dt / 1000000.0);

  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    UpdateRadar(meas_package);
  } else {
    UpdateLidar(meas_package);
  }

}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  AugmentSigmaPoints(&Xsig_aug);

  PredictSigmaPoints(Xsig_aug, delta_t);

  PredictMeanAndCovariance();
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
  VectorXd z = meas_package.raw_measurements_;
  VectorXd z_pred = H_laser_ * x_;
  VectorXd y = z - z_pred;
  MatrixXd Ht = H_laser_.transpose();
  MatrixXd S = H_laser_ * P_ * Ht + R_laser_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  //new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_laser_) * P_;

}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
  int n_z = 3;

  VectorXd z = meas_package.raw_measurements_;

  VectorXd z_pred = VectorXd(n_z);
  MatrixXd S = MatrixXd(n_z, n_z);
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  PredictRadar(&z_pred, &S, &Zsig);

  // cross correlation
  MatrixXd Tc = MatrixXd(n_x_, n_z);

  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    while (x_diff(3) > M_PI) x_diff(3) -= 2. * M_PI;
    while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  MatrixXd K = Tc * S.inverse();

  VectorXd z_diff = z - z_pred;

  while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
  while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

  // update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();
}

void UKF::init(MeasurementPackage meas_package) {
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    double rho = meas_package.raw_measurements_(0);
    double phi = meas_package.raw_measurements_(1);
    double rho_dot = meas_package.raw_measurements_(2);
    x_ << rho * cos(phi), rho * sin(phi), 0, 0, 0;
  } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
    double p_x = meas_package.raw_measurements_(0);
    double p_y = meas_package.raw_measurements_(1);
    x_ << p_x, p_y, 0, 0, 0;
  }
  time_us_ = meas_package.timestamp_;
  P_ = MatrixXd::Identity(n_x_, n_x_);
}

void UKF::AugmentSigmaPoints(MatrixXd* Xsig_out) {

  VectorXd x_aug = VectorXd(n_aug_);
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  x_aug.fill(0.0);
  x_aug.head(n_x_) = x_;

  P_aug.fill(0.0);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug.bottomRightCorner(n_aug_ - n_x_, n_aug_ - n_x_) <<
    std_a_ * std_a_, 0, 0, std_yawdd_ * std_yawdd_;

  MatrixXd L = P_aug.llt().matrixL();

  Xsig_aug.col(0) = x_aug;
  for (int i = 0; i < n_aug_; i++) {
    Xsig_aug.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
    Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
  }

  *Xsig_out = Xsig_aug;
}

void UKF::PredictSigmaPoints(const MatrixXd& Xsig_aug, const double delta_t) {

  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    VectorXd x = Xsig_aug.col(i);
    double v = x(2);
    double yaw = x(3);
    double yawd = x(4);
    double nu_a = x(5);
    double nu_yawdd = x(6);
    VectorXd delta_x(5);
    VectorXd nu_x(5);

    if (fabs(yawd) < 0.001) {
      delta_x << v * cos(yaw) * delta_t, v * sin(yaw) * delta_t, 0, 0, 0;
    } else {
      delta_x << v * (sin(yaw + yawd * delta_t) - sin(yaw)) / yawd,
        v * (cos(yaw) - cos(yaw + yawd * delta_t)) / yawd,
        0, yawd * delta_t, 0;
    }

    nu_x << 0.5 * delta_t * delta_t * cos(yaw) * nu_a,
      0.5 * delta_t * delta_t * sin(yaw) * nu_a,
      delta_t * nu_a,
      0.5 * delta_t * delta_t * nu_yawdd,
      delta_t * nu_yawdd;
    Xsig_pred_.col(i) = x.head(n_x_) + delta_x + nu_x;
  }

}

void UKF::PredictMeanAndCovariance() {
  VectorXd x = VectorXd(n_x_);
  MatrixXd P = MatrixXd(n_x_, n_x_);

  x.fill(0.0);

  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    x = x + weights_(i) * Xsig_pred_.col(i);
  }

  P.fill(0.0);

  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    VectorXd x_diff = Xsig_pred_.col(i) - x;
    while (x_diff(3) > M_PI) x_diff(3) -= 2. * M_PI;
    while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;
    P = P + weights_(i) * x_diff * x_diff.transpose();
  }

  x_ = x;
  P_ = P;
}

void UKF::PredictRadar(VectorXd* z_out, MatrixXd* S_out, MatrixXd* Zsig_out) {
  int n_z = 3;
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  // sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    double p_x = Xsig_pred_(0, i);
    double p_y = Xsig_pred_(1, i);
    double v = Xsig_pred_(2, i);
    double yaw = Xsig_pred_(3, i);
    double v_x = cos(yaw) * v;
    double v_y = sin(yaw) * v;
    // measurement model r, phi, r_dot
    Zsig(0, i) = sqrt(p_x * p_x + p_y * p_y);
    Zsig(1, i) = atan2(p_y, p_x);
    Zsig(2, i) = (p_x * v_x + p_y * v_y) / sqrt(p_x * p_x + p_y * p_y);
  }

  // mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }
  // measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;
    S = S + weights_(i) * z_diff * z_diff.transpose();
  }
  // measurement noise
  MatrixXd R = MatrixXd(n_z, n_z);
  R << std_radr_* std_radr_, 0, 0,
    0, std_radphi_ * std_radphi_, 0,
    0, 0, std_radrd_ * std_radrd_;
  S = S + R;

  *z_out = z_pred;
  *S_out = S;
  *Zsig_out = Zsig;
}