#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  // We have a vector rmse that will store the RMSE values for:
  // (rmse_pos_x, rmse_pos_y, rmse_v_x, rmse_v_y)
  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;

  int estSize = estimations.size();
  int gtSize = ground_truth.size();

  // Size of estimations and ground truth vectors should be poisitive
  if (estSize == 0 || gtSize == 0)
  {
    cout << "Size of estimations or ground truth vector 0:" << std::endl;
    return rmse;
  }

  // They should also match...
  if (estSize != gtSize)
  {
    cout << "Size of estimations and ground truth vectors differ." << std::endl;
    return rmse;
  }

  //accumulate squared residuals
  for (int i = 0; i < estSize; ++i)
  {
    VectorXd est = estimations.at(i);
    VectorXd g = ground_truth.at(i);

    // cout << "Estimations: " << est << std::endl;
    // cout << "Ground Tuth: " << g << std::endl;

    VectorXd res = est - g;
    VectorXd res_pow = res.array().square();
    res_pow /= estSize;

    // cout << "Res Pow: " << res_pow
    //      << std::endl;
    rmse = rmse + res_pow;
  }

  rmse = rmse.array().sqrt();

  // cout << "RMSE = " << rmse << std::endl;

  return rmse;
}