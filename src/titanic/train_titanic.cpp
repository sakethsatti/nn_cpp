#include <iostream>
#include <eigen3/Eigen/Core>
#include "../headers/nn.h"
#include "../headers/load_titanic.h"

int main() {
  auto dataset = titanic_loader(); // Eigen::MatrixXd
  

  Eigen::VectorXd target = dataset.col(0); // First column
  Eigen::MatrixXd features = dataset.block(0, 1, dataset.rows(), dataset.cols() - 1); // Everything except the first column
  

  NeuralNetwork nn = NeuralNetwork(features.cols(), {32, 32, 16, 8}, target.cols()); // Make Neural Network
  
 
  for (int idx = 0; idx < features.rows(); ++idx)
  {
    bool verbose = false;
    Eigen::VectorXd inp = features.row(idx);

    if (idx % 40 == 0) {
      std::cout << "Datapoint " << idx << std::endl;
      verbose = true;
    }
    
    auto y_pred = nn.call(inp);
    nn.train_step(target.row(idx), y_pred, 0.1, verbose);
  }
}
