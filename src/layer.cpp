#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <random>
#include "headers/layer.h"

Layer::Layer(const int& num_nodes, const int& n_prev_nodes)
  : num_nodes(num_nodes), n_prev_nodes(n_prev_nodes)
{
  weights.resize(num_nodes, n_prev_nodes);
  biases.resize(num_nodes);
  last_inputs.resize(n_prev_nodes);
  last_activations.resize(num_nodes);

  wb_init();
}

Eigen::VectorXd Layer::activation(const Eigen::VectorXd& weighted_sum)
{
  Eigen::VectorXd activs(num_nodes);

  for (int i = 0; i < num_nodes; ++i)
  {
    activs(i) = (weighted_sum(i) > 0.0) ? weighted_sum(i) : 0.0;
  }

  return activs;
}

Eigen::VectorXd Layer::dActivation(const Eigen::VectorXd& outs)
{
  Eigen::VectorXd dActivs(num_nodes); 

  for (int i = 0; i < num_nodes; ++i)
  {
    dActivs(i) = (outs(i) >= 0.0) ? 1.0 : 0.0;
  }

  return dActivs;
}

void Layer::wb_init()
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(-0.05, 0.05);

  for (int row = 0; row < num_nodes; ++row)
  {
    for (int col = 0; col < n_prev_nodes; ++col)
    {
      weights(row, col) = dis(gen);
    }
    biases(row) = dis(gen);
  }
}

Eigen::VectorXd Layer::forward(const Eigen::VectorXd& inps)
{
  last_inputs = inps;

  assert(inps.size() == n_prev_nodes);
  // weights: (num_nodes, prev_nodes) * inps: (prev_nodes)  = (num_nodes)
  // biases: (num_nodes)
  last_activations = activation(weights * inps + biases);
  
  return last_activations;
}

Eigen::VectorXd Layer::back(const Eigen::VectorXd& output_error, double learning_rate)
{
  // gradient of the activation function and gradient biases
  // element-wise multiplication
  Eigen::VectorXd activ_grad = output_error.array() * dActivation(last_activations).array();

  // gradients of weights
  Eigen::MatrixXd weight_grad = activ_grad * last_inputs.transpose();
  
  // gradient of inputs
  Eigen::VectorXd input_error = weights.transpose() * activ_grad;
  
  // Update weights and biases
  weights -= learning_rate * weight_grad;
  biases -= learning_rate * activ_grad;

  return input_error;

}

