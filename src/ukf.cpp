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
  std_a_ = 2;

  // Process noise standard deviation yaw acceleration in rad/s^2
  // ** This value is too high for a bicycle!
  std_yawdd_ = M_PI / 4.0;
  
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

  auto mType = meas_package.sensor_type_;
  if (!use_laser_ && mType == MeasurementPackage::LASER){
    cout << "Not using LIDAR measurement " << endl;
    return;
  }

  if (!use_radar_ && mType == MeasurementPackage::RADAR){
    cout << "Not using RADAR measurement " << endl;
    return;
  }

  std::string measurementType = mType == MeasurementPackage::LASER ? "LIDAR" : "RADAR";

  /* ***************
   * INITIALIZATION
   * ************** */
  if(!is_initialized_)
  {
    
    // Initialize our process covariance matrix
    // In the future initialise differently depending on sensor
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

        //angle normalization
        while (phi > M_PI) phi -= 2.*M_PI;
        while (phi < -M_PI) phi += 2.*M_PI;

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
    // But let's be super naive for now and assume we don't know bike velocity
    // nor yaw rate. While this is probably a terrible setup, the whole point
    // of the Kalman filter is that it should be able to work out where the bike
    // is and its speed and turn rate for appropriately configured covariance matrix
    // and process noise values (i.e. it may take a bit longer for my filter to 
    // "converge")
    x_ << x, y, 0, 0, 0;

    previous_timestamp_ = meas_package.timestamp_;
    is_initialized_ = true;

    cout << "Initialised ukf with " << measurementType << " x = "<< x_ << endl;
    return;
  }

  measurementType = meas_package.sensor_type_ == MeasurementPackage::LASER ? "LASER" : "RADAR";
  //compute the time elapsed between the current and previous measurements
  double dt = (meas_package.timestamp_ - previous_timestamp_) / 1000000.0; //dt - expressed in seconds
  previous_timestamp_ = meas_package.timestamp_;

  // Prediction step is the same regardless of the type of sensor measuremrnt we received
  Prediction(dt);

  switch(mType){
    case MeasurementPackage::RADAR:
      UpdateRadar(meas_package);
      break;
    
    case MeasurementPackage::LASER:
      UpdateLidar(meas_package);  
      break;
    
    default:
      cout << "Unknown measurement type: " << mType << endl;
  }
}


/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  // Generating the augmented sigma points for timestep (tk + 1)
  MatrixXd Xsig_aug = GenerateAugmentedSigmaPoints();

  // Predicting sigma points for timestep (tk + 1)
  Xsig_pred_ = PredictSigmaPoints(Xsig_aug, delta_t);

  // We are now in a position to predict mean vector and state covariance matrix
  x_ = PredictMean(Xsig_pred_);
  P_ = PredictStateCovarianceMatrix(Xsig_pred_, x_);
}

/**
 * Generates the augmented sigma points using current state vector
 * and state covariance matrix
 * @return The augmented sigma points
 * */
MatrixXd UKF::GenerateAugmentedSigmaPoints(){
  // Define our spreading factor lambda
  double lambda = 3 - n_aug_;

  // create augmented mean vector
  VectorXd x_aug = VectorXd(7);

  // create augmented state covariance
  MatrixXd P_aug = MatrixXd(7, 7);
  // Making sure it is set to zero
  P_aug.setZero();

  // create augmented sigma points matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  // create augmented mean state
  // since the distributions of our noice and yaw acceleration are centered
  // around zero, the means of both are zero, hence why we do not set
  // x_aug(5) and x_aug(6) 
  x_aug.head(n_x_) = x_; 
  x_aug(5) = 0;
  x_aug(6) = 0;  

  //Initialise augmented covariance Matrix
  P_aug.block(0, 0, n_x_, n_x_) = P_;

  // Define the process noice covariance matrux
  MatrixXd Q = MatrixXd(2, 2) ;
  Q << std_a_ * std_a_, 0,
       0, std_yawdd_ * std_yawdd_;
  P_aug.block(n_x_, n_x_, n_aug_ - n_x_, n_aug_ - n_x_) = Q;

  //create square root matrix
  MatrixXd P_aug_lambda = (lambda + n_aug_) * P_aug;
  MatrixXd P_aug_lambda_sqrt = P_aug_lambda.llt().matrixL();
  
  //create augmented sigma points
  Xsig_aug.col(0) = x_aug;
  Xsig_aug.block(0, 1, n_aug_, n_aug_) = P_aug_lambda_sqrt.colwise() + x_aug; 
  Xsig_aug.block(0, n_aug_ + 1, n_aug_, n_aug_) = (-1 * P_aug_lambda_sqrt).colwise() + x_aug;

  return Xsig_aug;
}


/**
 *  Predicts the sigma points at time (tk + 1).
 * @param Xsig_aug The augmented sigma points at time tk
 * @param delta_t The time difference ([tk + 1] - tk)
 * @return The Predicted sigma points at time (tk + 1)
 */
MatrixXd UKF::PredictSigmaPoints(const MatrixXd &Xsig_aug, double delta_t){
  MatrixXd Xsig_pred = MatrixXd(n_x_, 2 * n_aug_ + 1);

  for(int i = 0; i < Xsig_aug.cols(); ++i){
      VectorXd Xsig_aug_col = Xsig_aug.col(i);
      VectorXd Xk = Xsig_aug_col.head(n_x_);
      float v_k = Xsig_aug_col(2);
      float phi = Xsig_aug_col(3);
      float phi_dot = Xsig_aug_col(4);
      
      float delta_t_sq = delta_t * delta_t;
      float acc_noise = Xsig_aug_col(5);
      float yaw_rate_noise = Xsig_aug_col(6);
      
      // The Q vector is a common to both compute branch: phi_dot ~ 0 and phi_dot != 0
      // So we pre-compute it ahead of time
      VectorXd Q = VectorXd(5);
      Q << 0.5 * delta_t_sq * cos(phi) * acc_noise,
      0.5 * delta_t_sq * sin(phi) * acc_noise,
      delta_t * acc_noise,
      0.5 * delta_t_sq * yaw_rate_noise,
      delta_t * yaw_rate_noise;
      
      // Vector u's value depends on whether phi_dot is zero
      VectorXd u = VectorXd(5);
      
      if (fabs(phi_dot) <= 0.001){
        u << v_k * cos(phi) * delta_t,
        v_k * sin(phi) * delta_t,
        0,
        phi_dot * delta_t,
        0;
      }else{
        float divider = v_k / phi_dot;
        float phi_plus_phi_dot_T = phi + delta_t * phi_dot;
      
        u << divider * (sin(phi_plus_phi_dot_T) - sin(phi)),
        divider * (-cos(phi_plus_phi_dot_T) + cos(phi)),
        0,
        phi_dot * delta_t,
        0;
      }
      
      VectorXd Xk_plus_one = Xk + u + Q;
      Xsig_pred.col(i) = Xk_plus_one;
  }

  return Xsig_pred;
}


/**
 * Predicts the mean from our distribution of predicted sigma points
 * @param X_sig_prem the predicted sigma points
 * @return the predicted mean vector calculated from the sigma points
 */
VectorXd UKF::PredictMean(const MatrixXd &Xsig_pred){
  VectorXd x = VectorXd(n_x_);
  x.setZero();
  
  // define spreading parameter
  double lambda = 3 - n_aug_;

  //set weights  
  double w_i_0 =  lambda / (lambda + n_aug_);
  double w_i_1_plus = 1 / (2 * (lambda + n_aug_));

  // predict state mean
  for(int i = 0; i < Xsig_pred.cols(); ++i){
    VectorXd col_i = Xsig_pred.col(i);
    double w_i = i == 0 ? w_i_0 : w_i_1_plus;
    x += (w_i * col_i);
  }

  return x;
}

/**
 * Predicts the state covariance matrix for the next timestep
 * @param Xsig_pred the predicted sigma points
 * @param x The mean state vector
 * @return the predicted state covariance matrix
 */ 
MatrixXd UKF::PredictStateCovarianceMatrix(const MatrixXd &Xsig_pred, const VectorXd &x){
  //create covariance matrix for prediction
  MatrixXd P = MatrixXd(n_x_, n_x_);
  P.setZero();

  // define spreading parameter
  double lambda = 3 - n_aug_;

  // set weights
  double w_i_0 =  lambda / (lambda + n_aug_);
  double w_i_1_plus = 1 / (2 * (lambda + n_aug_));

  // predict state covariance matrix
  for(int i = 0; i < Xsig_pred.cols(); ++i){
    float w_i = i == 0 ? w_i_0 : w_i_1_plus;
    VectorXd col_i = Xsig_pred.col(i);
    VectorXd diff =  col_i - x;
    P += w_i * diff * diff.transpose();
  }  

  return P;
}


/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  // TODO store this in a variable so that it can be reused in future calls
  // as opposed to always redefined
  auto conv_fn = [] (const VectorXd &Xsig_pred_col){
      double px = Xsig_pred_col(0);
      double py = Xsig_pred_col(1);
      
      VectorXd z_col = VectorXd(2);
      z_col << px, py;

      return z_col;
  };

  //define spreading parameter
  double lambda = 3 - n_aug_;

  int n_z = 2;
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

    // set weights
  double w_i_0 =  lambda / (lambda + n_aug_);
  double w_i_1_plus = 1 / (2 * (lambda + n_aug_));

  // setting up mean measurement vector
  VectorXd z_pred = VectorXd(n_z);
  z_pred.setZero();
  
  // transform sigma points into measurement space
  for(int i = 0; i < Xsig_pred_.cols(); ++i){      
      VectorXd z_col = conv_fn(Xsig_pred_.col(i));      
      Zsig.col(i) = z_col;
      
      // Now accumulate into the predicted mean
      // So basically we are iteratively calculating the mean predicted state
      float w_i = i == 0 ? w_i_0 : w_i_1_plus;
      z_pred += w_i * z_col;
  }
  
  // Setting up measurement noise matrix
  MatrixXd R = MatrixXd(2, 2);
  R << std_laspx_ * std_laspx_, 0,
       0, std_laspy_ * std_laspy_;       
  
  // measurement covariance matrix
  MatrixXd S = MatrixXd(n_z, n_z);
  S.setZero();

  //calculate measurement covariance matrix S
  for(int i = 0; i < Zsig.cols(); ++i){
    double w_i = i == 0 ? w_i_0 : w_i_1_plus;
    VectorXd Zsig_i = Zsig.col(i);
    
    VectorXd diff = Zsig_i - z_pred;
    S += w_i * (diff * diff.transpose());
  }
  S += R;


  // create matrix for cross-correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.setZero();

  //calculate cross correlation matrix
  for(int i = 0; i < Zsig.cols(); ++i){
      VectorXd Zsig_col = Zsig.col(i);
      VectorXd Xsig_pred_col = Xsig_pred_.col(i);
      
      VectorXd diff_Zsig = Zsig_col - z_pred;
      VectorXd diff_Xsig_pred = Xsig_pred_col - x_;
      
      double w_i = i == 0 ? w_i_0 : w_i_1_plus;
      Tc += w_i * (diff_Xsig_pred * diff_Zsig.transpose());
  }

  // Setting up measurement vector
  VectorXd z = VectorXd(2);
  // and initialising it with the actual lidar measurement at timestep (tk + 1)
  z << meas_package.raw_measurements_[0],
       meas_package.raw_measurements_[1];      

  VectorXd z_diff = VectorXd(n_z);
  z_diff = z - z_pred;

  //calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();
  //update state mean and covariance matrix
  x_ = x_ + K * (z_diff);
  P_ = P_ - (K * S * K.transpose());  
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  // TODO store this in a variable so that it can be reused in future calls
  // as opposed to always redefined
  auto conv_fn = [] (const VectorXd &Xsig_pred_col){
      double px = Xsig_pred_col(0);
      double py = Xsig_pred_col(1);
      double v = Xsig_pred_col(2);
      double psi = Xsig_pred_col(3);
      
      double rho = sqrt(px * px + py *py);
      double phi = atan2(py, px); 
      double phi_dot = (px * cos(psi) * v + py * sin(psi) * v) / rho;
      
      VectorXd z_col = VectorXd(3);
      z_col << rho, phi, phi_dot;

      return z_col;
  };

  //define spreading parameter
  double lambda = 3 - n_aug_;

  int n_z = 3;
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

    // set weights
  double w_i_0 =  lambda / (lambda + n_aug_);
  double w_i_1_plus = 1 / (2 * (lambda + n_aug_));

  // setting up mean measurement vector
  VectorXd z_pred = VectorXd(n_z);
  z_pred.setZero();
  
  // transform sigma points into measurement space
  for(int i = 0; i < Xsig_pred_.cols(); ++i){      
      VectorXd z_col = conv_fn(Xsig_pred_.col(i));      
      Zsig.col(i) = z_col;
      
      // Now accumulate into the predicted mean
      // So basically we are iteratively calculating the mean predicted state
      float w_i = i == 0 ? w_i_0 : w_i_1_plus;
      z_pred += w_i * z_col;
  }
  
  // Setting up measurement noise matrix
  MatrixXd R = MatrixXd(3, 3);
  R << std_radr_ * std_radr_, 0, 0,
       0, std_radphi_ * std_radphi_, 0,
       0, 0, std_radrd_ * std_radrd_;
  
  // measurement covariance matrix
  MatrixXd S = MatrixXd(n_z,n_z);
  S.setZero();

  //calculate measurement covariance matrix S
  for(int i = 0; i < Zsig.cols(); ++i){
    double w_i = i == 0 ? w_i_0 : w_i_1_plus;
    VectorXd Zsig_i = Zsig.col(i);
    
    VectorXd diff = Zsig_i - z_pred;
    //angle normalization
    while (diff(1) > M_PI) diff(1)-= 2.0 * M_PI;
    while (diff(1) < -M_PI) diff(1)+= 2.0 * M_PI;
    S += w_i * (diff * diff.transpose());
  }
  S += R;



  // create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.setZero();

  //calculate cross correlation matrix
  for(int i = 0; i < Zsig.cols(); ++i){
      VectorXd Zsig_col = Zsig.col(i);
      VectorXd Xsig_pred_col = Xsig_pred_.col(i);
      
      VectorXd diff_Zsig = Zsig_col - z_pred;
      VectorXd diff_Xsig_pred = Xsig_pred_col - x_;

      //angle normalization
    while (diff_Zsig(1)> M_PI) diff_Zsig(1)-=2.*M_PI;
    while (diff_Zsig(1)<-M_PI) diff_Zsig(1)+=2.*M_PI;

    //angle normalization
    while (diff_Xsig_pred(3)> M_PI) diff_Xsig_pred(3)-=2.*M_PI;
    while (diff_Xsig_pred(3)<-M_PI) diff_Xsig_pred(3)+=2.*M_PI;

      
      double w_i = i == 0 ? w_i_0 : w_i_1_plus;
      Tc += w_i * (diff_Xsig_pred * diff_Zsig.transpose());
  }

  // Setting up measurement vector
  VectorXd z = VectorXd(3);
  // and initialising it with the actual radar measurement at timestep (tk + 1)
  z << meas_package.raw_measurements_[0],
       meas_package.raw_measurements_[1],
       meas_package.raw_measurements_[2];

  VectorXd z_diff = VectorXd(3);
  z_diff = z - z_pred;
   //angle normalization
  while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;


  //calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();
  //update state mean and covariance matrix
  x_ = x_ + K * (z_diff);
  P_ = P_ - (K * S * K.transpose()); 
}
