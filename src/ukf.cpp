#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>
#include <cmath>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 * This is scaffolding, do not modify
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
  // ** This value is too high for a bicycle!
  std_a_ = 30;

  // Process noise standard deviation yaw acceleration in rad/s^2
  // ** This value is too high for a bicycle!
  std_yawdd_ = 30;
  
  //DO NOT MODIFY measurement noise values below these are provided by the sensor manufacturer.
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
  //DO NOT MODIFY measurement noise values above these are provided by the sensor manufacturer.
  
  previous_timestamp_ = 0;

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */
  n_x_ = 5;
  n_aug_ = 7;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  std::string measurementType = "";

  /* ***************
   * INITIALIZATION
   * ************** */
  if(!is_initialized_)
  {
    
    // Initialize our process covariance matrix
    // TODO initialise differently depending on sensor
    P_ << 1, 0, 0, 0, 0,
          0, 1, 0, 0, 0,
          0, 0, 1, 0, 0,
          0, 0, 0, 1, 0,
          0, 0, 0, 0, 1;

    float x, y = 0;
    // We are processing the first measurement
    switch(meas_package.sensor_type_){
      case MeasurementPackage::RADAR :
      {
        // Since we use block specific variables (rho & phi) we run this code
        // with the curly braces {} 
        float rho = meas_package.raw_measurements_[0];
        float phi = meas_package.raw_measurements_[1];

        x = rho * cos(phi);
        y = rho * sin(phi);
        break;
      }
  
      case MeasurementPackage::LASER :
        x = meas_package.raw_measurements_[0];
        y = meas_package.raw_measurements_[1];      
        break;

      default:
        cout << "Unknown measurement package: " << meas_package.sensor_type_ << endl;
        return;
    }

    // TODO we probably want to start with a reasonable velocity and 0 angle
    x_ << x, y, 0, 0, 0;

    previous_timestamp_ = meas_package.timestamp_;

    is_initialized_ = true;
    return;
  }

  measurementType = meas_package.sensor_type_ == MeasurementPackage::LASER ? "LASER" : "RADAR";
  //compute the time elapsed between the current and previous measurements
  float dt = (meas_package.timestamp_ - previous_timestamp_) / 1000000.0; //dt - expressed in seconds
  previous_timestamp_ = meas_package.timestamp_;


  MatrixXd Xsig = GenerateSigmaPoints();
  MatrixXd Xsig_aug = GenerateAugmentedSigmaPoints(&Xsig);

}


/**
 * Generates the augmented sigma points from the current sigma points
 * @param Xsig The Current sigma points
 * @return The augmented sigma points
 * */
MatrixXd UKF::GenerateAugmentedSigmaPoints(MatrixXd* Xsig){
  // Define our spreading factor lambda
  double lambda = 3 - n_aug_;

  // create augmented mean vector
  VectorXd x_aug = VectorXd(7);

  // create augmented state covariance
  MatrixXd P_aug = MatrixXd(7, 7);
  // Making sure it is set to zero
  P_aug.setZero();

  // create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  // create augmented mean state
  // since the distributions of our noice and yaw acceleration are centered
  // around zero, the means of both are zero, hence why we do not set
  // x_aug(5) and x_aug(6) 
  x_aug.head(n_x_) = *Xsig; 

  //Initialise augmented covariance Matrix
  P_aug.block(0, 0, n_x_, n_x_) = P_;

  // Define the process noice covariance matrux
  MatrixXd Q = MatrixXd(2, 2) ;
  Q << std_a_ * std_a_, 0,
       0, std_yawdd_ * std_yawdd_;
  P_aug.block(n_x_, n_x_, n_aug_ - n_x_, n_aug_ - n_x_) = Q;

  //create square root matrix
  MatrixXd P_aug_lambda = (lambda_ + n_aug_) * P_aug;
  MatrixXd P_aug_lambda_sqrt = P_aug_lambda.llt().matrixL();
  
  //create augmented sigma points
  Xsig_aug.col(0) = x_aug;
  Xsig_aug.block(0, 1, n_aug_, n_aug_) = P_aug_lambda_sqrt.colwise() + x_aug; 
  Xsig_aug.block(0, n_aug_ + 1, n_aug_, n_aug_) = (-1 * P_aug_lambda_sqrt).colwise() + x_aug;

  return Xsig_aug;
}


/**
 * Generates 2 * n_x + 1 sigma points from the current x vector and process
 * covariance matrix P
 * @return the generated sigma points */
MatrixXd UKF::GenerateSigmaPoints(){
  // Generate Sigma points
  MatrixXd Xsig = MatrixXd(n_x_, 2 * n_x_ + 1);

  // Easily set the first sigma point, which is the current position
  Xsig.col(0) = x_;

   // Now onto calculating the more complex terms
  double lambda = 3 - n_x_;
  double lambda_plus_nx = lambda + n_x_;
  MatrixXd P_sqrt = lambda_plus_nx * P_;
  P_sqrt = P_sqrt.llt().matrixL();
  
  Xsig.block(0, 1, n_x_, n_x_) = P_sqrt.colwise() + x_;
  Xsig.block(0, n_x_ + 1, n_x_, n_x_) = (-1 * P_sqrt).colwise() + x_;

  return Xsig;
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
}
