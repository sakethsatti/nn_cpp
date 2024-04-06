#ifndef NN_H
#define NN_H

#include "layer.h"
#include <eigen3/Eigen/Core>

class NeuralNetwork {
  public:
    std::vector<Layer> layers;
    NeuralNetwork(const int& input_size, const std::vector<int>& hidden_sizes, const int& output_size);
    
    double mse(const Eigen::VectorXd& y, const Eigen::VectorXd& y_pred);
    Eigen::VectorXd grad_mse(const Eigen::VectorXd& y, const Eigen::VectorXd& y_pred);
  
    Eigen::VectorXd call(const Eigen::VectorXd& inp);
    void train_step(const Eigen::VectorXd& y, const Eigen::VectorXd& y_pred, const double& lr);
};

#endif
