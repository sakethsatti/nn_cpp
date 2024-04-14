#include "headers/nn.h"
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <iostream>

/*
* Testing for matrix size mismatches
* Ignore this
*/

int main()
{
  NeuralNetwork nn = NeuralNetwork(10, {16, 32, 16}, 2);
  
  auto inp = Eigen::VectorXd::Zero(10);

  auto outs = nn.call(inp);
  
  Eigen::VectorXd correctTest = Eigen::VectorXd::Ones(2);
  
  nn.train_step(correctTest, outs, 0.01);
  
  std::cout << "End" << std::endl;
  
  return 0;
}
