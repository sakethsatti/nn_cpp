#include "headers/nn.h"
#include "headers/layer.h"
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <iostream>

NeuralNetwork::NeuralNetwork(const int& input_size, const std::vector<int>& hidden_sizes, const int& output_size)
{
  // Hidden layers + output layer
  layers.reserve(hidden_sizes.size() + 1); 
  
  // First hidden layer n_prev_nodes = input_size
  layers.emplace_back(Layer(hidden_sizes[0], input_size));
  
  // Next hidden layers
  for (int l = 1; l < hidden_sizes.size(); ++l)
  {
    layers.emplace_back(hidden_sizes[l], hidden_sizes[l - 1]);
  }
  
  // Output layer
  layers.emplace_back(Layer(output_size, hidden_sizes.back()));
}

Eigen::VectorXd NeuralNetwork::call(const Eigen::VectorXd& inp)
{
  Eigen::VectorXd out = inp;
  for (int l = 0; l < layers.size(); ++l)
  {
    out = layers[l].forward(out);
  }

  return out;
}

double NeuralNetwork::mse(const Eigen::VectorXd& y, const Eigen::VectorXd& y_pred)
{
  auto diff = (y - y_pred).cwiseAbs2();
  
  return 1.0/y.size() * diff.sum();
}

Eigen::VectorXd NeuralNetwork::grad_mse(const Eigen::VectorXd& y, const Eigen::VectorXd& y_pred)
{
  return 2 * (y_pred - y)/y.size();
}

void NeuralNetwork::train_step(const Eigen::VectorXd& y, const Eigen::VectorXd& y_pred, const double& lr, const bool& verbose)
{ 
  if (verbose)
    std::cout << "Loss: " << mse(y, y_pred) << std::endl;
  
  Eigen::VectorXd error = grad_mse(y, y_pred);
  
  for (int l = layers.size() - 1; l >= 0; --l)
  {
    error = layers[l].back(error, lr);
  }
}



