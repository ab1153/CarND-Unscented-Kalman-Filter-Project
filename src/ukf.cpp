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
  std_a_ = 2;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.2;

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

  // state dimension
  n_x_ = 5;

  // agumented state dimension
  n_aug_ = 7;

  // spreading parameter for sigma points
  lambda_ = 3 - n_aug_;

  // NIS
  NIS_laser_ = 0.;
  NIS_radar_ = 0.;

  // previous measurement time
  time_us_ = 0;

  // set weights
  weights_ = VectorXd(2 * n_aug_ + 1);
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  weights_.tail(2 * n_aug_).fill(0.5 / (n_aug_ + lambda_));

  // initial state vector
  x_ = VectorXd(n_x_);
  x_.fill(1.);


  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);

  // predicted sigma points
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  // agumented sigma points
  Xsig_aug_ = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  is_initialized_ = false;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  //*************** initialization *********************

  if (!is_initialized_) {
    Initialization(meas_package);
    return;
  }



  double delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0;	//dt - expressed in seconds
  time_us_ = meas_package.timestamp_;

#if UKF_DEBUG
  std::cout << "delta_t = " << delta_t << std::endl;
  std::cout << "before prediction" << std::endl;
  std::cout << "x= \n" << x_ << std::endl;
  std::cout << "P= \n" << P_ << std::endl;
#endif

  //*************** Prediction ************************

  Prediction(delta_t);

#if UKF_DEBUG
  std::cout << "after prediction" << std::endl;
  std::cout << "x= \n" << x_ << std::endl;
  std::cout << "P= \n" << P_ << std::endl;
#endif

  //**************** Update **************************

  if (meas_package.sensor_type_ == MeasurementPackage::LASER) {

    UpdateLidar(meas_package);

#if UKF_DEBUG
    std::cout << "after laser update" << std::endl;
    std::cout << "x= \n" << x_ << std::endl;
    std::cout << "P= \n" << P_ << std::endl;
#endif
  }

  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {

    UpdateRadar(meas_package);

#if UKF_DEBUG
    std::cout << "after radar update" << std::endl;
    std::cout << "x= \n" << x_ << std::endl;
    std::cout << "P= \n" << P_ << std::endl;
#endif
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
  // calculate agumented sigma points, populate Xsig_agu_
  CalculateAugmentedSigmaPoints();

#if UKF_DEBUG
  std::cout << "agumented sigma points:\n" << Xsig_aug_ << std::endl;
#endif

  // calculate prediction on sigma points, populate Xsig_pred_
  PredictSigmaPoints(delta_t);

#if UKF_DEBUG
  std::cout << "predicted sigma points:\n" << Xsig_pred_ << std::endl;
#endif

  // calculate predicted mean and covariance, update x_ and P_ from k|k to k+1|k.
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
  /**
  Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.
  */

  int n_z = 2; // laser only measures px,py

  VectorXd z = meas_package.raw_measurements_;

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  //transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    //2n+1 simga points

    // extract values for better readibility
    double p_x = Xsig_pred_(0, i);
    double p_y = Xsig_pred_(1, i);

    // measurement model
    Zsig(0, i) = p_x;        //r
    Zsig(1, i) = p_y;        //phi
  }

  //mean predicted measurement z_pred
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    //2n+1 simga points
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  //add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_z, n_z);
  R << std_laspx_ * std_laspx_, 0,
    0, std_laspy_ * std_laspy_;
  S = S + R;


  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);


  //calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    //2n+1 simga points

    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3) -= 2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3) += 2.*M_PI;
    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();

  }

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //residual
  VectorXd z_diff = z - z_pred;


  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();

  //calculate NIS
  NIS_radar_ = (z - z_pred).transpose() * S.inverse() * (z - z_pred);
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
  /**
  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.
  */

  int n_z = 3; // radar measures rho,phi,rho_dot

  VectorXd z = meas_package.raw_measurements_;

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  //transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    //2n+1 simga points

    // extract values for better readibility
    double p_x = Xsig_pred_(0, i);
    double p_y = Xsig_pred_(1, i);
    double v = Xsig_pred_(2, i);
    double yaw = Xsig_pred_(3, i);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;


    // measurement model
    Zsig(0, i) = sqrt(p_x*p_x + p_y*p_y);                        //r
    Zsig(1, i) = atan2(p_y, p_x);                                 //phi
    Zsig(2, i) = (p_x*v1 + p_y*v2) / Zsig(0, i);                 //r_dot
  }

  //mean predicted measurement z_pred
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }


  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    //2n+1 simga points
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1) -= 2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1) += 2.*M_PI;

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }


  //add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_z, n_z);
  R << std_radr_*std_radr_, 0, 0,
    0, std_radphi_*std_radphi_, 0,
    0, 0, std_radrd_*std_radrd_;
  S = S + R;


  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);


  //calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    //2n+1 simga points

    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1) -= 2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1) += 2.*M_PI;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3) -= 2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3) += 2.*M_PI;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //residual
  VectorXd z_diff = z - z_pred;

  //angle normalization
  while (z_diff(1)> M_PI) z_diff(1) -= 2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1) += 2.*M_PI;

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();


  //calculate NIS
  NIS_radar_ = (z - z_pred).transpose() * S.inverse() * (z - z_pred);
}

/**
* Calculate the agumented sigma points by sampling the P_ with sigma points,
* and agument the state vectors with the mean of the process noise values (have zero mean).
* This function update Xsig_aug_
*/
void UKF::CalculateAugmentedSigmaPoints() {

  VectorXd x_aug = VectorXd(n_aug_);

  //create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

  //create augmented mean state
  x_aug.head(n_x_) = x_;
  x_aug(n_x_) = 0;
  x_aug(n_x_ + 1) = 0;

  //create augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5, 5) = P_;
  P_aug(5, 5) = std_a_*std_a_;
  P_aug(6, 6) = std_yawdd_*std_yawdd_;

  //create square root matrix
  MatrixXd L = P_aug.llt().matrixL();

  //create augmented sigma points
  Xsig_aug_.col(0) = x_aug;
  for (int i = 0; i< n_aug_; i++) {
    Xsig_aug_.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
    Xsig_aug_.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
  }

};


/**
* Calculate the predicted sigma points.
* Run all the sigma points through the nonlinear process model,
* to get the sigma points at k+1|k.
* These sigma points will be used later in PredictMeanAndCovariance to construct the state x_ and covariance P_ at k+1|k.
* This function update Xsig_pred_
*/
void UKF::PredictSigmaPoints(double delta_t) {


  //predict sigma points
  for (int i = 0; i< 2 * n_aug_ + 1; i++) {
    //extract values for better readability
    double p_x = Xsig_aug_(0, i);
    double p_y = Xsig_aug_(1, i);
    double v = Xsig_aug_(2, i);
    double yaw = Xsig_aug_(3, i);
    double yawd = Xsig_aug_(4, i);
    double nu_a = Xsig_aug_(5, i);
    double nu_yawdd = Xsig_aug_(6, i);

    //predicted state values
    double px_p, py_p;

    //avoid division by zero
    if (fabs(yawd) > 0.001) {
      px_p = p_x + v / yawd * (sin(yaw + yawd*delta_t) - sin(yaw));
      py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd*delta_t));
    } else {
      px_p = p_x + v*delta_t*cos(yaw);
      py_p = p_y + v*delta_t*sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;

    //add noise
    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;

    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    // avoid zero division
    if (fabs(px_p) < 0.0001 && fabs(py_p) < 0.0001) px_p = 0.0001;

    //write predicted sigma point into right column
    Xsig_pred_(0, i) = px_p;
    Xsig_pred_(1, i) = py_p;
    Xsig_pred_(2, i) = v_p;
    Xsig_pred_(3, i) = yaw_p;
    Xsig_pred_(4, i) = yawd_p;
  }

};


/**
* Update the state and covariance using the predicted sigma points.
* Update the state and covariance to k+1|k.
* This function update x_ and P_
*/
void UKF::PredictMeanAndCovariance() {

  //predicted state mean
  x_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
    x_ = x_ + weights_(i) * Xsig_pred_.col(i);
  }


  //predicted state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
                                              // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    if (x_diff(3) > 1000000) {
      std::cout << "error: predicted angle too large: " << x_diff(3) << std::endl;
      std::exit(0);
    }
    while (x_diff(3)> M_PI) x_diff(3) -= 2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3) += 2.*M_PI;

    P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
  }


};

void UKF::Initialization(MeasurementPackage meas_package) {

  /*
  Initialize the state vector using the first measurment
  */

  time_us_ = meas_package.timestamp_;

  if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
    // initialize state vector using laser data

    x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0, 0, 0;

  } else if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    // initialize state vector using rader data

    double rho = meas_package.raw_measurements_[0];
    double phi = meas_package.raw_measurements_[1];
    double px = rho*cos(phi);
    double py = rho*sin(phi);

    x_ << px, py, fabs(meas_package.raw_measurements_[2]), 0, 0;
  }

  /*
  Initialize the covariance
  */

  P_.fill(0.);
  for (int i = 0;i<n_x_;i++) P_.diagonal()[i] = 1.;

  is_initialized_ = true;

};
