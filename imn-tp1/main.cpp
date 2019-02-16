/**
 * IMN430 - Visualisation - Travail 1
 * Reduction de la dimension d'un ensemble de donnees
 * 
 * Gabriel Beauchemin-Dauphinais 
 * Samuel Larouche 15071681
 *
 * 7 fev 2019
 */

#include <iostream>
#include <vector>

#include "./deps/eigen/Eigen/Dense"
#include "./deps/cimg/CImg.h"
#include <filesystem>

// par souci de lisibilite a la correction
#define image_t CImg<float>
namespace fs = std::experimental::filesystem;
using namespace fs;
using namespace cimg_library;
using namespace Eigen;

// parametres modifiables
const int kValue = 5; // nombre de meilleurs resultats a sortir
const int nValue = 20; // nombre de eigenfaces a utiliser
const std::string outputPath = "./eigenfaces/";
const std::string outputFilePrefix = "face";
const std::string outputFileFormat = ".ppm";
const std::string inputImg = "./inputImage/";
const std::string reconstructedImgPath = "./reconstructedFaces/";

// structures utilitaires
struct ImgDistance
{
	double distance = 0;
	image_t img;

	ImgDistance(double d, image_t i) : distance{ d }, img{ i } {};
};

struct ImgDistanceComparator
{
	inline bool operator() (const ImgDistance& struct1, const ImgDistance& struct2)
	{
		return (struct1.distance < struct2.distance);
	}
};

// ==========================================================================
/// LoadInputImage
/// Charge l'image pour laquelle on cherche les k plus similaires. Retourne
/// une image CIMG de type image_t.
// ==========================================================================
image_t loadInputImage()
{
	auto it = directory_iterator(inputImg);
	return image_t(it->path().string().c_str());
}

// ==========================================================================
/// LoadReferenceImages
/// Charge les images de reference avec lesquelles on va produire les 
/// n eigenvectors. Retourne un vecteur d'images de type image_t contenant 
/// toutes les images de references.
// ==========================================================================
std::vector<image_t> loadReferenceImages()
{
	std::vector<image_t> imgs;
	std::string path = "./imageset";
	for (const auto & entry : directory_iterator(path))
	{
		auto imgFileName = entry.path().string();
		image_t img(imgFileName.c_str());
		imgs.push_back(img.RGBtoYCbCr().channel(0));
	}
	return imgs;
}

// ==========================================================================
/// ComputeMeanImage
/// Calcule l'image moyenne des images passees en parametres. Retourne
/// l'image moyenne sous le type image_t.
// ==========================================================================
image_t computeMeanImage(std::vector<image_t> imgs)
{
	int imageCount = imgs.size();
	image_t meanImg = imgs[0];

	for (int i = 1; i < imageCount; ++i)
	{
		meanImg += imgs[i];
	}

	cimg_forXY(meanImg, x, y)
	{
		meanImg[x, y] = meanImg[x, y] / imageCount;
	}

	return meanImg;
}

// ==========================================================================
/// SubstractMeanFromImages
/// Soustrait l'image moyenne donnee à toutes les images du vecteur d'image
/// passe en parametre. La modification est faire directement au contenu du
/// vecteur. La fonction ne retourne rien.
// ==========================================================================
void substractMeanFromImages(std::vector<image_t>& imgs, image_t mean) 
{ 
	for (image_t img : imgs)
	{
		img -= mean;
	}
}

// ==========================================================================
/// CreateImageVector
/// Prend une image de type image_t en parametre et retourne un VectorXd
/// contenant les valeurs de chaque pixel de l'image.
// ==========================================================================
VectorXd createImageVector(image_t img)
{
	int imgSize = img.width() * img.height();
	VectorXd imgVector(imgSize);
	for (int indexPixel = 0; indexPixel < imgSize; ++indexPixel)
	{
		imgVector(indexPixel) = img(indexPixel);
	}
	
	return imgVector;
}

// ==========================================================================
/// CreateImageMatrix
/// Prend un vecteur d'images image_t en parametre et retourne un matrice
/// composee de toutes les images, dont chaque colonne correspond aux 
/// pixels d'une image.
// ==========================================================================
MatrixXd createImageMatrix(std::vector<image_t>& imgs)
{
	int imageCount = imgs.size();
	int imageSize = imgs[0].size();
	MatrixXd imgsMatrix(imageSize, imageCount);
	for (int indexImg = 0; indexImg < imageCount; ++indexImg)
	{
		image_t img = imgs[indexImg];
		for (int indexPixel = 0; indexPixel < imageSize; ++indexPixel)
		{
			imgsMatrix(indexPixel, indexImg) = img(indexPixel);
		}
	}
	return imgsMatrix;
}

// ==========================================================================
/// CreateCovarianceMatrix
/// Prend la matrice construite a partir des images de references et retourne
/// la matrice diminuee des covariances.
// ==========================================================================
MatrixXd createCovarianceMatrix(MatrixXd& data)
{
	double N = data.rows();
	auto covarianceMatrix = (data).transpose() * (data) / (N - 1);
	return covarianceMatrix;
}

// ==========================================================================
/// ComputeEigenFaces
/// Prend la matrice des images de reference et sa matrice de covariance
/// diminuee et retourne un vecteur contenant les n vecteurs propres qui
/// serviront a produire les n eigenfaces.
// ==========================================================================
std::vector<VectorXd> computeEigenFaces(MatrixXd& varianceCovarianceMatrix, MatrixXd& data)
{
	std::vector<VectorXd> eigenFaces;
	EigenSolver<MatrixXd> solver(varianceCovarianceMatrix);

	int maxIndex;
	solver.eigenvalues().real().maxCoeff(&maxIndex);
	int minIndex;
	solver.eigenvalues().real().minCoeff(&minIndex);

	int nbrEigenFacesGenerated = 0;
	for (int i = maxIndex; i <= minIndex; ++i)
	{
		VectorXd eigenVector((solver.eigenvectors().col(i)).real());
		VectorXd eigenFace = data * eigenVector;
		eigenFace.normalize();
		eigenFaces.push_back(eigenFace);

		++nbrEigenFacesGenerated;
		if (nbrEigenFacesGenerated >= nValue) //enough eigenFaces
			break;
	}

	return eigenFaces;
}

// ==========================================================================
/// SaveImagesToDisk
/// Prend un vecteur d'images image_t et les sauvegardes au path, prefixe et
/// format prevus dans les parametres du programme
// ==========================================================================
void saveImagesToDisk(std::vector<image_t> imgs)
{
	int imageCount = imgs.size();
	for (int imageIdx = 0; imageIdx < imageCount; ++imageIdx)
	{
		std::string fullPath = outputPath + outputFilePrefix + outputFileFormat;
		imgs[imageIdx].save(fullPath.c_str(), imageIdx);
	}
}

// ==========================================================================
/// ComputeVectorDistance
/// Calcule la distance entre deux vecteurs
// ==========================================================================
double computeVectorDistance(VectorXd vec1, VectorXd vec2)
{
	double total = 0;
	for (int i= 0; i < vec1.size(); ++i)
	{
		total += pow(vec1(i) - vec2(i), 2);
	}
	return sqrt(total);
}

// ==========================================================================
/// Fonction principale du programme
// ==========================================================================
int main()
{
	std::vector<image_t> imgs = loadReferenceImages();

	int imgHeight = imgs[0].height();
	int imgWidth = imgs[0].width();

	// (1) generer les eigenvectors a partir des images de reference
	image_t meanImage = computeMeanImage(imgs);
	substractMeanFromImages(imgs, meanImage);
	MatrixXd imageMatrix = createImageMatrix(imgs);

	MatrixXd cov = createCovarianceMatrix(imageMatrix);
	std::vector<VectorXd> eigenfacesData = computeEigenFaces(cov, imageMatrix);
	std::vector<image_t> eigenfacesImgs;

	for (int imageIdx = 0; imageIdx < std::min((int)eigenfacesData.size(),nValue); ++imageIdx)
	{
		image_t image(imgHeight, imgWidth);
		int pixelIdx = 0;
		for (image_t::iterator it = image.begin(); it < image.end(); ++it)
		{
			*it = eigenfacesData[imageIdx][pixelIdx];
			pixelIdx++;
		}
		image.normalize(0, 255);
		eigenfacesImgs.push_back(image);
	}
	saveImagesToDisk(eigenfacesImgs);

	// (2) generer un espace a partir des eigenvectors
	auto eigenSpace = createImageMatrix(eigenfacesImgs);

	// (3) representer chacune des images dans cet espace reduit (changement de base)
	std::vector<VectorXd> imgsEigenSpace;
	for (auto& img : imgs)
	{
		VectorXd imgEigenSpace = eigenSpace.colPivHouseholderQr().solve(createImageVector(img));
		imgsEigenSpace.push_back(imgEigenSpace);
	}

	// (4) charger l'image de reference et on la represente dans notre espace reduit
	auto inputImg = loadInputImage();
	VectorXd inputImgEigenSpace = eigenSpace.colPivHouseholderQr().solve(createImageVector(inputImg));

	// (5) calculer la distance entre l'image analysee et chacune des images de reference
	std::vector<ImgDistance> distByImg;
	for (int i = 0; i < imgs.size(); ++i)
	{
		auto imgEigenSpace = imgsEigenSpace[i];
		auto img = imgs[i];
		double distance = computeVectorDistance(inputImgEigenSpace, imgEigenSpace);
		distByImg.push_back(ImgDistance(distance, img));
	}

	// (6) sauvegarder l'image analysee telle que reconstruite a partir des eigenfaces
	int max = inputImgEigenSpace.size();
	auto reconstructedInitialImg = inputImgEigenSpace(0) * eigenfacesImgs[0];
	for (int i = 1; i < max; ++i)
	{
		reconstructedInitialImg += inputImgEigenSpace[i] * eigenfacesImgs[i];
	}
	reconstructedInitialImg.normalize(0, 255);
	reconstructedInitialImg.save( (reconstructedImgPath + "reconstructedImg_" + std::to_string(nValue) + ".ppm" ).c_str());

	// (7) afficher les k images les plus similaire a l'image analysee
	std::sort(distByImg.begin(), distByImg.end(), ImgDistanceComparator());
	for (int i = 0; i < std::min(kValue, (int)distByImg.size()); ++i)
	{
		distByImg[i].img.display();
	}
}