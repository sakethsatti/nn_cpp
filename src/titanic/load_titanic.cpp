#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <fstream>
#include <unordered_map>
#include "../headers/load_titanic.h"
#include <iostream>
#include <sstream>

Eigen::MatrixXd titanic_loader() 
{
  std::ifstream titanic_open;
  titanic_open.open("src/titanic/train.csv");
  
  if (titanic_open.is_open())
  {
    std::cout << "is open" << std::endl;
  }

  std::unordered_map <char, int> embarked_map;
  embarked_map.insert({ 'C', 1 });
  embarked_map.insert({ 'Q', 2 });
  embarked_map.insert({ 'S', 3 });

  std::unordered_map <std::string, int> gender_map;
  gender_map.insert({ "male", 0});
  gender_map.insert({ "female", 1});

  char delim = ',';
  


  // 890 datapoints, 7 features
  // 1 target
  // Features: Pclass, Sex (male: 0, female: 1), Ages, SibSp (# siblings), parch (# parents), Fare, Embarked (1: C, 2: Q, 3: S)  
  Eigen::Matrix<double, 891, 8> dataset;
  
  std::vector<int> remove = {0, 3, 4, 9, 11};
  
  
  // Start at -1 so that first real row will be 0
  int row = -1;
  std::string line;

  while (std::getline(titanic_open, line))
  {
    std::stringstream ss(line);
    std::string cell;
     
    int csv_column = 0;
    int real_column = 0;
    
    // Skip first row because it only contains column names
    if (row == -1) 
    {
      ++row;
      continue;
    }
    

    while(std::getline(ss, cell, delim))
    {
      if (std::count(remove.begin(), remove.end(), csv_column))
      {
        ++csv_column;
        continue;
      }

      
      if (isdigit(cell[0]))
      {
        dataset(row, real_column) = std::stod(cell);
      }
      else {
        if (csv_column == 5)
          dataset(row, real_column) = gender_map[cell]; 
        else
          dataset(row, real_column) = embarked_map[cell[0]]; 
      }
      ++csv_column;
      ++real_column;
    }

  
    ++row;
  }
  
  titanic_open.close();

  return dataset;
}
