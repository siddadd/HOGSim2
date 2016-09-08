#include <stdio.h>
#include <ios>
#include <fstream>
#include <stdexcept>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>

using namespace std;
using namespace cv;

string toLowerCase(string inStr);
vector <string> splitString(string inStr, char delim[]);
//void loadDescriptorVectorFromFile(const char* fileName, Mat& descriptor);
Mat loadDescriptorVectorFromFile(const char* fileName);
std::vector<float> loadMOCDescriptorVectorFromFile(const char* fileName);
Mat convertSVM(Mat& svmData);
void nonMaxSurpression(vector<cv::Rect>& rectList, vector<double>& weights, int groupThreshold, double eps);
void displayDotP(Mat processingImage, Mat dotp_output, int dotPThreshold);

// Trained feature using SVM. Will use this as our ROI template and compute dot products by 
// running this model over the hog-space image to find valid detections
// HW Dalal
static string hwd_hmdescriptorVectorFile = "hwd_hmdescriptorvector.dat";
// HW Pedro
static string hwp_hmdescriptorVectorFile = "hwp_hmdescriptorvector.dat";

// Default opencv features will use this file -> obtained from hog.cpp in modules\objdetect\src\hog.cpp
static std::string oc_hmdescriptorVectorFile = "oc_hmdescriptorvector.dat";

// Self-trained opencv features will use this file
static std::string moc_hmdescriptorVectorFile = "moc_hmdescriptorvector.dat";

// Software HOG parameters
static const Size padding = Size(0, 0);
static const Size winStride = Size(8, 8);

enum HARDWAREHOGPARAMS : uint32_t
{
	KERNELWIDTH = 3,
	KERNELHEIGHT = 3,
	NUMLAYERS = 64,
	BASESCALE = 1,
	SVMROIWIDTH = 14,
	SVMROIHEIGHT = 6,
	SVMPEDROROICHANNELS = 31,
	SVMDALALROICHANNELS = 36,
	CELLSIZE = 8,
	ROIPIXWIDTH = 64,
	ROIPIXHEIGHT = 128,
	PEDRODETECTIONTHRESHOLD = 122000,  // 61000x2
	DALALDETECTIONTHRESHOLD = 106750,   // 61000x1.75 
	BITSPERSVMBIN = 32,
	NUMSVMBIN = 18,
	NUMRHABIN = 32
};