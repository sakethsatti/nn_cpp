#ifndef LAYER_H
#define LAYER_H

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <eigen3/Eigen/src/Core/util/Constants.h>

class Layer {
  public:
    int num_nodes;
    int n_prev_nodes;

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> weights;
    Eigen::VectorXd biases;
    Eigen::VectorXd last_inputs;
    Eigen::VectorXd last_activations;
    
    Layer(const int& num_nodes, const int& n_prev_nodes);

    Eigen::VectorXd activation(const Eigen::VectorXd& weighted_sum);
    Eigen::VectorXd dActivation(const Eigen::VectorXd& out);
    void wb_init();

    Eigen::VectorXd forward(const Eigen::VectorXd& inp);
    Eigen::VectorXd back(const Eigen::VectorXd& output_error, double learning_rate);
};

#endif
