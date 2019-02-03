//
//  ImageAlign.cpp
//

#include <iostream>
#include <vector>
#include <cmath>

#include "eigen-eigen-323c052e1731/eigen-eigen-323c052e1731/Eigen/Dense"

#define cimg_display 0
#include "./CImg_latest/CImg-2.5.0_pre011819/CImg.h"
#include <filesystem>
#include <assert.h> 
namespace fs = std::experimental::filesystem;

using namespace Eigen;
using namespace cimg_library;
using namespace std;

std::vector< CImg<unsigned char>> loadImages()
{
  std::vector< CImg<unsigned char>> imgs;
  std::string path = "./images";
  for (const auto & entry : fs::directory_iterator(path))
  {
    auto imgFileName = entry.path().string();
    CImg<unsigned char> img(imgFileName.c_str());
    imgs.push_back(img.RGBtoYCbCr().channel(0));
  }
  return imgs;
}

MatrixXd imgsToMatrix(std::vector<CImg<unsigned char>>& imgs)
{
  assert(imgs.size() > 0); //no images loaded
  int nbrPixels = imgs[0].width() * imgs[0].height();
  MatrixXd imgsMatrix(imgs.size(), nbrPixels);
  for (int indexImg = 0; indexImg < imgs.size(); ++indexImg)
  {
    CImg<unsigned char> img = imgs[indexImg];
    for (int indexPixel = 0; indexPixel < nbrPixels; ++indexPixel)
    {
      imgsMatrix(indexImg, indexPixel) = img(indexPixel);
    }
  }
  return imgsMatrix;
}

MatrixXd calculateCovarianceMatrix(MatrixXd& data)
{
  int N = data.cols();
  auto averagePerImg = data.rowwise().mean();
  VectorXd ones(N);
  ones.setZero();
  auto covarianceMatrix = ((data - averagePerImg * ones.transpose()) * (data - averagePerImg * ones.transpose()).transpose()) / (N - 1) ;
  return covarianceMatrix;
}

std::vector<VectorXd> calculateEigenFaces(MatrixXd& varianceCovarianceMatrix)
{
  std::vector<VectorXd> eigenFaces; 
  EigenSolver<MatrixXd> solver(varianceCovarianceMatrix);
  int maxIndex;
  solver.eigenvalues().real().maxCoeff(&maxIndex);
  int minIndex;
  solver.eigenvalues().real().minCoeff(&minIndex);
  for (int i = maxIndex; i <= minIndex; ++i)
  {
    VectorXd eigenVector((solver.eigenvectors().col(maxIndex)).real());
    eigenFaces.push_back(eigenVector);
  }

  return eigenFaces;
}

int main(int argc, const char * argv[]) 
{
  std::vector< CImg<unsigned char>> imgs = loadImages();
  auto imgMatrix = imgsToMatrix(imgs);
  auto varianceCovarianceMatrix = calculateCovarianceMatrix(imgMatrix);
  auto eigenFaces = calculateEigenFaces(varianceCovarianceMatrix);
}
