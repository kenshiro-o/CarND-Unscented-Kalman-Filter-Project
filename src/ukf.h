#ifndef UKF_H
#define UKF_H

#include "measurement_package.h"
#include "Eigen/Dense"
#include <vector>
#include <string>
#include <fstream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

class UKF {
public:

  ///* initially set to false, set to true in first call of ProcessMeasurement
  bool is_initialized_;

  ///* if this is false, laser measurements will be ignored (except for init)
  bool use_laser_;

  ///* if this is false, radar measurements will be ignored (except for init)
  bool use_radar_;

  ///* state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
  VectorXd x_;

  ///* state covariance matrix
  MatrixXd P_;

  ///* predicted sigma points matrix
  MatrixXd Xsig_pred_;

  ///* time when the state is true, in us
  long long time_us_;

  ///* Process noise standard deviation longitudinal acceleration in m/s^2
  double std_a_;

  ///* Process noise standard deviation yaw acceleration in rad/s^2
  double std_yawdd_;

  ///* Laser measurement noise standard deviation position1 in m
  double std_laspx_;

  ///* Laser measurement noise standard deviation position2 in m
  double std_laspy_;

  ///* Radar measurement noise standard deviation radius in m
  double std_radr_;

  ///* Radar measurement noise standard deviation angle in rad
  double std_radphi_;

  ///* Radar measurement noise standard deviation radius change in m/s
  double std_radrd_ ;

  ///* Weights of sigma points
  VectorXd weights_;

  ///* State dimension
  int n_x_;

  ///* Augmented state dimension
  int n_aug_;

  ///* Sigma point spreading parameter
  double lambda_;

  // Recording the previous timestamp
  long long previous_timestamp_;

  /**
   * Constructor
   */
  UKF();

  /**
   * Destructor
   */
  virtual ~UKF();

  /**
   * Generates 2 * n_x + 1 sigma points from the current x vector and process
   * covariance matrix P
   * @return the generated sigma points */
  MatrixXd GenerateSigmaPoints();


  /**
   * Generates the augmented sigma points from the current sigma points
   * @param Xsig The Current sigma points
   * @return The augmented sigma points
   * */
  MatrixXd GenerateAugmentedSigmaPoints(MatrixXd* Xsig);

  /**
  * Predicts the sigma points at time (tk + 1).
  * @param Xsig_aug The augmented sigma points at time tk
  * @param delta_t The time difference ([tk + 1] - tk)
  * @return The Predicted sigma points at time (tk + 1)
  * */
  MatrixXd PredictSigmaPoints(MatrixXd* Xsig_aug, double delta_t);


  /**
   * Predicts the mean from our distribution of predicted sigma points
   * @param Xsig_pred the predicted sigma points
   * @return The predicted mean vector calculated from the sigma points
   */
  VectorXd PredictMean(MatrixXd* Xsig_pred);


  /**
   * Predicts the state covariance matrix for the next timestep
   * @param Xsig_pred the predicted sigma points
   * @param x The mean state vector
   * @return the predicted state covariance matrix
   */ 
  MatrixXd PredictStateCovarianceMatrix(MatrixXd* Xsig_pred, VectorXd* x);

  /**
   * ProcessMeasurement
   * @param meas_package The latest measurement data of either radar or laser
   */
  void ProcessMeasurement(MeasurementPackage meas_package);

  /**
   * Prediction Predicts sigma points, the state, and the state covariance
   * matrix
   * @param delta_t Time between k and k+1 in s
   */
  void Prediction(double delta_t);


  

  /**
   * Updates the state and the state covariance matrix using a laser measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateLidar(MeasurementPackage meas_package);

  /**
   * Updates the state and the state covariance matrix using a radar measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateRadar(MeasurementPackage meas_package);
};

#endif /* UKF_H */
