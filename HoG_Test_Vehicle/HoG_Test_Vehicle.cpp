/**********************************************************************************************************************************/
/* 		HOG - Person Detection Hardware Equivalent Model								                                          */
/*		Penn State University - CSE-MDL	2015						                                                              */
/*		HOGSim	: Version 2.00										                                                              */
/*		Author - Siddharth Advani, Yasuki Tanabe					                                                              */
/*      Dependencies     : OpenCV 2.4.11 (64 bit source)    			                                                          */
/*					     : Dirent	1.20.1																					      */			
/*					     : hwHOGDll	(see lib folder)																		      */
/*		Assumptions      : Inputs are RGB Images																				  */
/*		Usage	         : HoG_Test_Vehicle.exe -i <ImageDirectory> -o <OutputDirectory> -m <-1/0/1/2> -gt <GroundTruthDirectory> */
/*      Revision History : Adding support for Dalal, Adding evaluation pathway													  */				
/**********************************************************************************************************************************/

#include <dirent.h>
#include "hwHOGdll.h"
#include "HoG_Test_Vehicle.h"

#include <sstream>
#include <fstream>
#include <iomanip>
#include <string>
#include <iostream>
#include <stdlib.h>

// for OpenCV2
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/gpu/gpu.hpp"
#include "opencv2/highgui/highgui.hpp"

#define SVMLIB libSVM

// By default run hardware model unless over-ridden by command arguements
#define RUN_HARDWARE true

// Internal debugging. Keep false
//#define DISPLAY_INTERMEDIATE_LAYERS 

#pragma comment(lib, "hwHOGDll.lib")

#ifdef _DEBUG
#pragma comment(lib, "opencv_contrib2411d.lib")
#pragma comment(lib, "opencv_core2411d.lib")
#pragma comment(lib, "opencv_imgproc2411d.lib")
#pragma comment(lib, "opencv_objdetect2411d.lib")
#pragma comment(lib, "opencv_gpu2411d.lib")
#pragma comment(lib, "opencv_features2d2411d.lib")
#pragma comment(lib, "opencv_highgui2411d.lib")
#pragma comment(lib, "opencv_calib3d2411d.lib")
#pragma comment(lib, "opencv_legacy2411d.lib")
#pragma comment(lib, "opencv_ml2411d.lib")

#else
#pragma comment(lib, "opencv_contrib2411.lib")
#pragma comment(lib, "opencv_core2411.lib")
#pragma comment(lib, "opencv_imgproc2411.lib")
#pragma comment(lib, "opencv_objdetect2411.lib")
#pragma comment(lib, "opencv_gpu2411.lib")
#pragma comment(lib, "opencv_features2d2411.lib")
#pragma comment(lib, "opencv_highgui2411.lib")
#pragma comment(lib, "opencv_calib3d2411.lib")
#pragma comment(lib, "opencv_legacy2411.lib")
#pragma comment(lib, "opencv_ml2411.lib")

#endif

#define mymax(x,y) ((x>=y)? x : y)
#define mymin(x,y) ((x>=y)? y : x)

int main(int argc, char* argv[])
{
	bool runHardware = RUN_HARDWARE;
	bool runhwDalal = false;
	bool runhwPedro = false;
	bool runOpenCVDefault = false;
	String imageDirectory = "";
	String outDirectory = "";
	String gtDirectory = ""; 

	if (argc < 3)
	{
		cerr << "[ERROR]: Usage is HoG_Test_Vehicle.exe " // command name
			<< " [-i <ImageDirectory>] [-o <OutputDirectory>] [-m <-1/0/1/2>] [-gt <GroundTruthDirectory>]" << endl
			<< "Demo program to simulate hardware HOG" << endl
			<< " -i : input directory" << endl
			<< " -o : output directory" << endl
			<< " -m : -1 = default software version (Dalal-OpenCV), 0 = self-trained software version (Dalal-OpenCV), 1 = hardware version (Dalal), 2 = hardware version (Pedro)" << endl
			<< " -gt : groundtruth directory (optional)" << endl
			<< "Press escape to exit" << endl;		
		return 1;
	}

	// Parse through command arguements
	while (argc > 1 && argv[1][0] == '-')
	{
		if (strncmp(argv[1], "-i", 2) == 0)
		{
			imageDirectory = String(argv[2]).c_str();
			argc-=2; argv+=2;
		}
		else if (strncmp(argv[1], "-o", 2) == 0)
		{
			outDirectory = String(argv[2]).c_str();
			argc -= 2; argv += 2;
		}
		else if (strncmp(argv[1], "-m", 2) == 0)
		{
			if (atoi(argv[2]) == 0 || atoi(argv[2]) == -1)
			{
				runHardware = false;
				if (atoi(argv[2]) == -1)
					runOpenCVDefault = true;
			}
			else // run hardware
			{
				if (atoi(argv[2]) == 1)
					runhwDalal = true;
				else
					runhwPedro = true;
			}
			argc -= 2; argv += 2;
		}
		else if (strncmp(argv[1], "-gt", 3) == 0)
		{
			gtDirectory = String(argv[2]).c_str();
			argc -= 2; argv += 2;
		}
	}

	static vector <string> validExtensions;
	validExtensions.push_back("jpg");
	validExtensions.push_back("png");
	validExtensions.push_back("ppm");

	// Get Files from Directory
	static vector <string> inputFiles;
	static vector <string> outputFiles;
	static vector <string> outputgtFiles;
	static vector <string> gtFiles;
	static vector <int> fileIDs;

	// GT Image
	Mat gt_img; 

	cout << "[INFO]: Opening Directory " << imageDirectory << endl;
	struct dirent* ep;
	size_t extensionLocation;
	DIR* inputDir = opendir(imageDirectory.c_str());
	int fileID; 

	if (inputDir != NULL)
	{
		while ((ep = readdir(inputDir)))
		{
			// Ignore sub-directories
			if (ep->d_type & DT_DIR)
			{
				continue;
			}

			extensionLocation = string(ep->d_name).find_last_of(".");  // Assume the last point marks beginning of extension like file.ext

			// Check if extension is matching the wanted ones
			string tempExt = toLowerCase(string(ep->d_name).substr(extensionLocation + 1));

			if (find(validExtensions.begin(), validExtensions.end(), tempExt) != validExtensions.end())
			{
				string tempID = string(ep->d_name).substr(extensionLocation - 3, 3); // since there are 288 files, use 3 chars before .
				
				fileID = atoi(tempID.c_str()) + 1;

				cout << "[INFO]: Found matching data file " << ep->d_name << "-> FILE ID:" << fileID << endl;

				inputFiles.push_back(imageDirectory + ep->d_name);
				outputFiles.push_back(outDirectory + ep->d_name);
				fileIDs.push_back(fileID); 
				size_t substrlen = (size_t)string(ep->d_name).length() - tempExt.length() - 1;
				gtFiles.push_back(gtDirectory + string(ep->d_name).substr(0,substrlen) + ".txt");   // Assuming groundtruth is *.txt
				outputgtFiles.push_back(outDirectory + "GT_" + ep->d_name);
			}
			else
				cout << "[INFO]: Found file does not match required file type, skipping " << ep->d_name << endl;
		}
	}
	else
	{
		cout << "[ERROR]: Cannot open directory " << inputDir << endl;
		return 0;
	}

	cout << "[INFO]: Total " << inputFiles.size() << " image files" << endl;

	// Write detections to output text file for plotting P-R curves
	ofstream detFile;
	detFile.open(outDirectory + "V000.txt");

	int dotPThreshold = 0;

	// Process each file
	for (int fn = 0; fn < inputFiles.size(); fn++)
	{
		Mat inria_img = imread(inputFiles.at(fn), cv::IMREAD_COLOR);
		Mat src_img = inria_img.clone();

		int KernelWidth = KERNELWIDTH;
		int KernelHeight = KERNELHEIGHT;
		double bgnScale = BASESCALE;
		int numLayer = NUMLAYERS; // number of pyramid layers to use
		double scaleF = 1.1; // scale between pyramid layer	
		int roi_pix_height = ROIPIXHEIGHT;
		int roi_pix_width = ROIPIXWIDTH;
		int cell_pix_height = CELLSIZE;
		int cell_pix_width = CELLSIZE;
		double finalThreshold = 2.0; // Used in NMS

		vector<Rect> foundLocations;
		vector<double> foundWeights;
		Mat scaledImage;
		double im_scale;

		/****************
		// { // C MODEL
		/****************/
		if (runHardware == true)
		{	
			IHOG *hwHOG = NULL;
			Mat svmRoIData;
			Mat svmData;
			Mat inputImage;
			Mat dotP;

			// C Model Instantiation
			if (runhwDalal == true)
			{
				hwHOG = new HardwareDalalHOG;
				dotPThreshold = DALALDETECTIONTHRESHOLD;
				//svmRoIData = Mat::zeros(SVMDALALROICHANNELS, SVMROIWIDTH, CV_32SC1);
				svmData = Mat::zeros(SVMDALALROICHANNELS*SVMROIWIDTH*SVMROIHEIGHT, 1, CV_32SC1);
			}
			else if (runhwPedro == true)
			{
				hwHOG = new HardwarePedroHOG;
				dotPThreshold = PEDRODETECTIONTHRESHOLD;
				//svmRoIData = Mat::zeros(SVMPEDROROICHANNELS, SVMROIWIDTH, CV_32SC1);
				svmData = Mat::zeros(SVMPEDROROICHANNELS*SVMROIWIDTH*SVMROIHEIGHT, 1, CV_32SC1);
			}

			// for each layer
			for (int layer = 0; layer < numLayer; layer++)
			{
				cout << "Layer: " << layer << endl;
				if (layer == 0)
					im_scale = bgnScale;
				else
					im_scale *= (1.0f / scaleF);

				int maxScoreLayer = 0; // variable for debug/tuning		

				resize(src_img, scaledImage, Size(src_img.cols*im_scale, src_img.rows*im_scale), NULL, NULL, CV_INTER_CUBIC);
				Size imageSize = Size((int)(((double)src_img.cols * im_scale) / (double)cell_pix_width) * cell_pix_width,
					(int)(((double)src_img.rows * im_scale) / (double)cell_pix_height) * cell_pix_height);

				// Crop image into multiple of CellSize
				inputImage = scaledImage(Rect(0, 0, imageSize.width, imageSize.height));

				if (inputImage.cols < roi_pix_width || inputImage.rows < roi_pix_height)
				{
					cout << "Here" << endl;
					break;
				}

				//imshow("Input", inputImage);
				//waitKey(0);

				/**************************/
				// { // C MODEL - RHA
				/**************************/

				__int32* bins = hwHOG->RHA(inputImage);

				/**************************/
				// } // C MODEL - RHA
				/**************************/

				double dotPFrameSum = 0; // variable to get average dotProduct sum. (For debug/tuning)
				_Uint32t dotPFrameCnt = 0; // variable to get average dotProduct sum. (For debug/tuning)
				int maxScoreFrame = 0; // variable for debug/tuning		
				
				//Mat dotP = Mat::zeros(inputImage.rows / roi_pix_height, inputImage.cols / roi_pix_width, CV_32SC1);
				//Mat normdotP;

				double dpMaxVal = NULL;
				double dpMinVal = NULL;
				
				if (runhwDalal == true)
				{
					svmData = loadDescriptorVectorFromFile(hwd_hmdescriptorVectorFile.c_str());
					svmRoIData = svmData.reshape(SVMDALALROICHANNELS, SVMROIWIDTH);
				}
				else if (runhwPedro == true)
				{
					svmData = loadDescriptorVectorFromFile(hwp_hmdescriptorVectorFile.c_str());
					svmRoIData = svmData.reshape(SVMPEDROROICHANNELS, SVMROIWIDTH);
				}

				hwHOG->svm = convertSVM(svmRoIData);

				/**************************/
				// { // C MODEL - NCA
				/**************************/

				dotP = hwHOG->NCA(bins, roi_pix_height, roi_pix_width, 1);

				/**************************/
				// } // C MODEL - NCA
				/**************************/

				uint16_t imageCellWidth = inputImage.cols / cell_pix_width;
				uint16_t imageCellHeight = inputImage.rows / cell_pix_height;

#ifdef DISPLAY_INTERMEDIATE_LAYERS
				displayDotP(inputImage, dotP, dotPThreshold);
#endif

				// This needs to change to hardware dotproduct output 
				cv::minMaxLoc(dotP, &dpMinVal, &dpMaxVal, NULL, NULL);

				//dotP.convertTo(normdotP, -1, 1 / dpMaxVal, 0);

				for (int v = 0; v < dotP.rows; v++)
				{
					for (int h = 0; h < dotP.cols; h++)
					{
						if (0 < v && (v * cell_pix_height + roi_pix_height) < inputImage.rows
							&&
							0 < h && (h * cell_pix_width + roi_pix_width) < inputImage.cols
							)
						{
							dotPFrameSum += dotP.ptr<__int32>(v)[h];
							dotPFrameCnt += 1;
							maxScoreLayer = mymax(maxScoreLayer, dotP.ptr<__int32>(v)[h]);
						}

						if (dotPThreshold <= dotP.ptr<__int32>(v)[h])  // Limits the number of valid detections to the best ones 
						{
							int curVal = dotP.ptr<__int32>(v)[h];

							Rect curRect(
								(int)((double)h / im_scale) * cell_pix_width,
								(int)((double)v / im_scale) * cell_pix_height,
								(int)((double)roi_pix_width / im_scale),
								(int)((double)roi_pix_height / im_scale)
								);

							foundLocations.push_back(curRect);
							foundWeights.push_back((double)curVal);
						}
					}
					// debug
					maxScoreFrame = mymax(maxScoreFrame, maxScoreLayer);
				}
			} // End each layer 

			// Using Non Max Supression from OpenCV
			nonMaxSurpression(foundLocations, foundWeights, (int)finalThreshold, 0.2);
			delete hwHOG;
		}
		/****************
		// } // C MODEL
		/****************/
		else
		{
			// OPENCV Version
			cout << "[INFO]: Running OpenCV HOG Model" << endl;
			cv::HOGDescriptor swDHOG;
			const int hitThreshold = 0;

			if (runOpenCVDefault == true)
			{
				//std::vector <float> ocDetector = swDHOG.getDefaultPeopleDetector();
				std::vector<float> ocDetector = loadMOCDescriptorVectorFromFile(oc_hmdescriptorVectorFile.c_str());
				swDHOG.setSVMDetector(ocDetector);
				swDHOG.detectMultiScale(src_img, foundLocations, foundWeights, hitThreshold, winStride, padding, scaleF, 2.0, false);
			}
			else
			{
				// Load SVM model
				cout << "[INFO]: Loading moc model" << endl;
				std::vector<float> mocDetector = loadMOCDescriptorVectorFromFile(moc_hmdescriptorVectorFile.c_str());
				swDHOG.setSVMDetector(mocDetector);
				swDHOG.detectMultiScale(src_img, foundLocations, foundWeights, hitThreshold, winStride, padding, scaleF, 2.0, false);
			}
		}

		// Visualize and Write Final Output
		if (foundLocations.size() > 0)
		{
			stringstream ss;
			string detStr;
			
			for (unsigned j = 0; j < foundLocations.size(); j++) {
				ss.str("");  // Reset 
				cv::Rect r = foundLocations[j];
				ss << fileIDs.at(fn);
				ss << ",";
				ss << r.x;
				ss << ",";                                    
				ss << r.y;
				ss << ",";
				ss << r.width;
				ss << ",";
				ss << r.height;
				ss << ",";
				string foundWeightsStr;
				if (runHardware)
					foundWeightsStr = to_string((int)foundWeights[j]);
				else
					foundWeightsStr = to_string((double)foundWeights[j]);

				ss << foundWeightsStr;
					
				detStr = ss.str();
				detFile << detStr << endl;
				cout << detStr << endl;

				putText(src_img, foundWeightsStr, Point(r.x, r.y - 10), CV_FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 0, 255), 2);
				rectangle(src_img, r.tl(), r.br(), cv::Scalar(255, 0, 0), 2);
			}
		}

		// Get Groundtruth
		if (argc > 3)
		{
			cout << "[INFO]: Opening Groundtruth" << endl;
			gt_img = inria_img.clone();
			String gtFilename = gtFiles.at(fn);
			ifstream gtFile(gtFilename.c_str());
			string line;
			int header = 0;
			if (gtFile.is_open())
			{
				while (getline(gtFile, line))
				{
					cout << line << endl;
					if (header == 0) // ignore header
					{
						header++;
						continue;
					}
					else
					{
						char delim[] = { " " };
						vector <string> elems = splitString(line, delim);  // split at space

						if (elems.size() > 4)
						{
							rectangle(gt_img, Rect(atoi(elems.at(1).c_str()), atoi(elems.at(2).c_str()), atoi(elems.at(3).c_str()), atoi(elems.at(4).c_str())), cv::Scalar(0, 255, 0), 2);
						}

					}
				}
				gtFile.close();
			}
#ifdef DISPLAY_INTERMEDIATE_LAYERS			
			cv::imshow("GroundTruth", gt_img);
			waitKey(0);
#endif
		}
#ifdef DISPLAY_INTERMEDIATE_LAYERS			
		cv::imshow("Detections", src_img);
		waitKey(0);
#endif
		// Write to Output
		string outDirWintemp = string(outDirectory.begin(), outDirectory.end());
		LPCSTR outDirWin = outDirWintemp.c_str();

		if (CreateDirectory(outDirWin, NULL) || ERROR_ALREADY_EXISTS == GetLastError())
		{
			//string hogMode = "Pedro";
			cv::imwrite(outputFiles.at(fn), src_img);
			if (argc > 3)
				cv::imwrite(outputgtFiles.at(fn), gt_img);
		}
		else
			cout << "[ERROR]: Failed to create output directory" << endl;

		
	} // End process each file
	
	detFile.close();
	(void)closedir(inputDir);
	return 0;
}



	/**************************************************************/
	/*							FUNCTIONS			    		  */
	/**************************************************************/
	/**
	* Splits a string 
	* @param inStr
	* @param delim
	*/
	vector <string> splitString(string inStr, char delim[])
	{
		vector <string> outStr; 
		string token;
		stringstream ss (inStr);
		while (getline(ss, token, *delim))
		{
			outStr.push_back(token);
		}

		return outStr;
	}


	/**
	* Converts a string from upper to lower case	
	* @param inStr
	*/
	string toLowerCase(string inStr)
	{
		string outStr;
		for (string::const_iterator i = inStr.begin(); i != inStr.end(); i++)
		{
			outStr += tolower(*i);
		}

		return outStr;
	}
	
	
	/**
	* Loads the given descriptor vector from a file
	* @param descriptorVector the descriptor vector to load
	* @param fileName
	*/
	Mat loadDescriptorVectorFromFile(const char* fileName)
	{
		printf("Loading descriptor vector from file '%s'\n", fileName);

		CvMLData* mlData = new CvMLData;

		int status = mlData->read_csv(fileName);

		if (status != 0)
		{
			printf("Error reading svm data file\n");
		}

		cv::Mat tmpdescriptor = mlData->get_values();

		return tmpdescriptor;
	}

	/**
	* Loads the my opencv descriptor vector from a file
	* @param descriptorVector the descriptor vector to load
	* @param fileName
	*/
	std::vector<float> loadMOCDescriptorVectorFromFile(const char* fileName)
	{
		printf("Loading descriptor vector from file '%s'\n", fileName);
		CvMLData* mlData = new CvMLData;
		int status = mlData->read_csv(fileName);
		
		if (status != 0)
		{
			printf("Error reading svm data file\n");
		}

		cv::Mat tmpdescriptor = mlData->get_values();
		std::vector<float> descriptor;
		descriptor.resize(tmpdescriptor.rows*tmpdescriptor.cols);

		for (int i = 0; i < tmpdescriptor.rows*tmpdescriptor.cols; i++)
		{
			descriptor.at(i) = tmpdescriptor.at<float>(i);
		}

		delete mlData;
		return descriptor;
	}


	/**
	* Convert SVM data from floating point (LibSVM / OpenCV) to fixed point for hardware SVM
	* Used Yasuki's mat2bin tool as reference
	* @param svmData the trained SVM weights
	*/
	Mat convertSVM(Mat& svmData)
	{
		// Normalize svmData Globally
		double svmMinVal;
		double svmMaxVal;
		minMaxLoc(svmData, &svmMinVal, &svmMaxVal, NULL, NULL);

		//cout << "Min: " << svmMinVal << endl;
		//cout << "Max" << svmMaxVal << endl; 

		double svmAbsMaxVal = mymax(abs(svmMinVal), abs(svmMaxVal));

		Mat normsvmData;

		svmData.convertTo(normsvmData, CV_64FC1, 1 / svmAbsMaxVal, 0);

		int org_roi_height = normsvmData.rows;
		int org_roi_width = normsvmData.cols;
		int org_roi_bin = normsvmData.channels();

		// Check if dimensions is Pedro or Dalal
		assert((org_roi_bin == 31) || (org_roi_bin == 36));

		int tgt_roi_width = ((org_roi_width + 7) / 8) * 8; // bigger nearest multiply of 8 value
		int tgt_roi_height = ((org_roi_height + 7) / 8) * 8; // bigger nearest multiply of 8 value
		int tgt_roi_bin = 36;

		Mat svmHW(tgt_roi_height, tgt_roi_width, CV_16SC(tgt_roi_bin));

		int space_h = (tgt_roi_width != org_roi_width) ? 1 : 0;
		int space_v = (tgt_roi_height != org_roi_height) ? 1 : 0;

		double maxVal;
		double minVal;
		double *buf = (double *)malloc(sizeof(double) * org_roi_width * org_roi_height * org_roi_bin);
		double *ptr = (double *)normsvmData.ptr<double>(0, 0);

#define REF_BUF(V,H,B) (buf[V*(org_roi_width*org_roi_bin) + H * (org_roi_bin) + B])

		for (int v = 0; v < org_roi_height; v++)
		{
			for (int h = 0; h < org_roi_width; h++)
			{
				for (int bin = 0; bin < org_roi_bin; bin++)
				{
					if (v == 0 && h == 0 && bin == 0)
					{
						maxVal = *ptr;
						minVal = *ptr;
					}
					else
					{
						maxVal = mymax(*ptr, maxVal);
						minVal = mymin(*ptr, minVal);
					}
					REF_BUF(v, h, bin) = *ptr;
					*ptr++;
				}
			}
		}

		cout << "MaxVal=" << maxVal << endl;
		cout << "MinVal=" << minVal << endl;

		FILE *fp = fopen("svm.bin", "wb");
		FILE *txt_fp = fopen("svm.txt", "w");
		FILE *txt_flt_fp = fopen("svm.flt.txt", "w");

		ptr = (double *)normsvmData.ptr<double>(0);
		short int maxSVal = -32768;
		short int minSVal = 32767;

		for (int v = 0; v < tgt_roi_height; v++)
		{
			for (int h = 0; h < tgt_roi_width; h++)
			{
				fprintf(txt_fp, "V=%d, H=%d: ", v, h);
				fprintf(txt_flt_fp, "V=%d, H=%d: ", v, h);
				for (int bin = 0; bin < tgt_roi_bin; bin++)
				{
					int org_v = v - space_v;
					int org_h = h - space_h;
					bool valid = (0 <= org_v && org_v < org_roi_height
						&&
						0 <= org_h && org_h < org_roi_width
						&&
						0 <= bin && bin < org_roi_bin);
					//short int val;
					if (valid)
					{
						double tmp = REF_BUF(org_v, org_h, bin) * 16384.0;
						if (tmp < -16384.0f && 16384.0f<tmp)
						{
							printf("tmp was %f\n", tmp);
							printf("Error: overflow occured value It must be from -1.0 to 1.0\n");
						}
						//val = (short int)tmp;
						// SVM in fixed point for hardware
						svmHW.ptr<short int>(v)[svmHW.channels()*h + bin] = (short int)tmp;
					}
					else
					{
						svmHW.ptr<short int>(v)[svmHW.channels()*h + bin] = 0;
						//val = 0;
					}
					fwrite(&(svmHW.ptr<short int>(v)[svmHW.channels()*h + bin]), sizeof(short int), 1, fp);
					fprintf(txt_fp, "%10d, ", svmHW.ptr<short int>(v)[svmHW.channels()*h + bin]);
					fprintf(txt_flt_fp, "%10f, ", (double)((double)(svmHW.ptr<short int>(v)[svmHW.channels()*h + bin]) / 16384));
					maxSVal = mymax(maxSVal, svmHW.ptr<short int>(v)[svmHW.channels()*h + bin]);
					minSVal = mymin(minSVal, svmHW.ptr<short int>(v)[svmHW.channels()*h + bin]);
				}
				fprintf(txt_fp, "\n");
				fprintf(txt_flt_fp, "\n");
			}
		}

		fclose(txt_fp);
		fclose(txt_flt_fp);
		fclose(fp);
		printf("MaxFixPVal = %d MinSVal = %d\n", maxSVal, minSVal);
		printf("Info: Use (%d * CELL_WIDTH, %d * CELL_HEIGHT) pix  ROI_Size for this SVM\n", tgt_roi_width, tgt_roi_height);
		printf("Info: svm.bin dumped\n");

		return svmHW;
	}


	/**
	* Use OpenCV groupRectangles
	* @param rectList
	* @param weights
	* @param groupThreshold
	* @params epsilon
	*/
	void nonMaxSurpression(vector<cv::Rect>& rectList, vector<double>& weights, int groupThreshold, double eps)
	{
		if (groupThreshold <= 0 || rectList.empty())
		{
			return;
		}

		CV_Assert(rectList.size() == weights.size());

		vector<int> labels;
		int nclasses = partition(rectList, labels, SimilarRects(eps));

		vector<cv::Rect_<double> > rrects(nclasses);
		vector<int> numInClass(nclasses, 0);
		vector<double> foundWeights(nclasses, -std::numeric_limits<double>::max());
		int i, j, nlabels = (int)labels.size();

		for (i = 0; i < nlabels; i++)
		{
			int cls = labels[i];
			rrects[cls].x += rectList[i].x;
			rrects[cls].y += rectList[i].y;
			rrects[cls].width += rectList[i].width;
			rrects[cls].height += rectList[i].height;
			foundWeights[cls] = mymax(foundWeights[cls], weights[i]);
			numInClass[cls]++;
		}

		for (i = 0; i < nclasses; i++)
		{
			// find the average of all ROI in the cluster
			cv::Rect_<double> r = rrects[i];
			double s = 1.0 / numInClass[i];
			rrects[i] = cv::Rect_<double>(cv::saturate_cast<double>(r.x*s),
				cv::saturate_cast<double>(r.y*s),
				cv::saturate_cast<double>(r.width*s),
				cv::saturate_cast<double>(r.height*s));
		}

		rectList.clear();
		weights.clear();

		for (i = 0; i < nclasses; i++)
		{
			cv::Rect r1 = rrects[i];
			int n1 = numInClass[i];
			double w1 = foundWeights[i];
			if (n1 <= groupThreshold)
				continue;
			// filter out small rectangles inside large rectangles
			for (j = 0; j < nclasses; j++)
			{
				int n2 = numInClass[j];

				if (j == i || n2 <= groupThreshold)
					continue;

				cv::Rect r2 = rrects[j];

				int dx = cv::saturate_cast<int>(r2.width * eps);
				int dy = cv::saturate_cast<int>(r2.height * eps);

				if (r1.x >= r2.x - dx &&
					r1.y >= r2.y - dy &&
					r1.x + r1.width <= r2.x + r2.width + dx &&
					r1.y + r1.height <= r2.y + r2.height + dy &&
					(n2 > std::max(3, n1) || n1 < 3))
					break;
			}

			if (j == nclasses)
			{
				rectList.push_back(r1);
				weights.push_back(w1);
			}
		}
	}


	/*
	* Display Dotproduct Output
	* Ref: Yasuki's DebugFunc.cs
	*/
	void displayDotP(Mat processingImage, Mat dotp_output, int dotPThreshold)
	{
		cv::Scalar color_red((double)0.0, (double)0.0, (double)255.0, 1.0f);
		cv::Scalar color_blue((double)255.0, (double)0.0, (double)0.0, 1.0f);
		cv::Scalar color_green((double)0.0, (double)255.0, (double)0.0, 1.0f);

		cv::Size font_size;
		cv::Size font_small_size;

		int* baseline = 0;

		font_size = getTextSize("A", CV_FONT_HERSHEY_COMPLEX, 1, 1, baseline);
		font_small_size = getTextSize("A", CV_FONT_HERSHEY_COMPLEX, 0.5, 0.5, baseline);


		int threshold = dotPThreshold;
		int maxValue = LONG_MIN;
		int minValue = LONG_MAX;
		int maxScoreXPos = 0;
		int maxScoreYPos = 0;
		for (int v = 0; v < dotp_output.rows; v++)
		{
			for (int h = 0; h < dotp_output.cols; h++)
			{
				if (maxValue < dotp_output.ptr<__int32>(v)[h])
				{
					maxValue = dotp_output.ptr<__int32>(v)[h];
					maxScoreXPos = h * CELLSIZE;
					maxScoreYPos = v * CELLSIZE;
				}
				if (dotp_output.ptr<__int32>(v)[h] != LONG_MIN &&
					minValue > dotp_output.ptr<__int32>(v)[h])
				{
					minValue = dotp_output.ptr<__int32>(v)[h];
				}
			}
		}

		Mat cvtImg = Mat(processingImage.rows, processingImage.cols, CV_8UC4);

#if !DISPLAY_RAW
		for (int v = 0; v < cvtImg.rows; v++)
		{
			for (int h = 0; h < cvtImg.cols; h++)
			{
				double y =
					(0.114 * (double)processingImage.at<Vec3b>(v, h).val[0]
					+ 0.587 * (double)processingImage.at<Vec3b>(v, h).val[1]
					+ 0.299 * (double)processingImage.at<Vec3b>(v, h).val[2]);
				cvtImg.at<Vec4b>(v, h).val[0] = (uint8_t)y;
				cvtImg.at<Vec4b>(v, h).val[1] = (uint8_t)y;
				cvtImg.at<Vec4b>(v, h).val[2] = (uint8_t)y;
				cvtImg.at<Vec4b>(v, h).val[3] = (uint8_t)255;
			}
		}
#endif

		int r_min = LONG_MAX;
		int r_max = LONG_MIN;
		int g_min = LONG_MAX;
		int g_max = LONG_MIN;
		int b_min = LONG_MAX;
		int b_max = LONG_MIN;
		for (int v = 0; v < dotp_output.rows; v++)
		{
			for (int h = 0; h < dotp_output.cols; h++)
			{
				double b = 0.0f;
				double g = 0.0f;
				double r = 0.0f;
				int dotP = dotp_output.ptr<__int32>(v)[h];

				if (dotP > threshold)
				{
					double base_ratio = 0.5;
					double ratio = 3.0*((double)(dotP - threshold) / (double)(maxValue - threshold));
					if (ratio < 1.0)
					{
						b = base_ratio + (1.0 - base_ratio) * (ratio - 0.0);
						b_min = mymin(dotP, b_min);
						b_max = mymax(dotP, b_max);
					}
					else if (ratio < 2.0)
					{
						g = base_ratio + (1.0 - base_ratio) * (ratio - 1.0);
						g_min = mymin(dotP, g_min);
						g_max = mymax(dotP, g_max);
					}
					else if (ratio <= 3.0)
					{
						r = base_ratio + (1.0 - base_ratio) * (ratio - 2.0);
						r_min = mymin(dotP, r_min);
						r_max = mymax(dotP, r_max);
					}
					else
					{
						assert(false);
					}

					for (int v_i = 0; v_i < CELLSIZE; v_i++)
					{
						for (int h_i = 0; h_i < CELLSIZE; h_i++)
						{
							int abs_v = v * CELLSIZE + v_i;
							int abs_h = h * CELLSIZE + h_i;
							double y = cvtImg.at<Vec4b>(abs_v, abs_h).val[0];
#if !DISPLAY_RAW

							cvtImg.at<Vec4b>(abs_v, abs_h).val[0] = (uint8_t)(0 * (1.0 - b) + (255.0 * b));
							cvtImg.at<Vec4b>(abs_v, abs_h).val[1] = (uint8_t)(0 * (1.0 - g) + (255.0 * g));
							cvtImg.at<Vec4b>(abs_v, abs_h).val[2] = (uint8_t)(0 * (1.0 - r) + (255.0 * r));

#else
							cvtImg.Data[abs_v, abs_h, 0] = (byte)(ratio * 255.0f);
							cvtImg.Data[abs_v, abs_h, 1] = (byte)(ratio * 255.0f);
							cvtImg.Data[abs_v, abs_h, 2] = (byte)(ratio * 255.0f);
#endif
						}
					}
				}
			}
		}


		int vpos = cvtImg.rows - 4;
		// ---
		string minmax_str = "MaxScore: " + std::to_string(maxValue) + ", MinScore: " + std::to_string(minValue);
		putText(cvtImg, minmax_str, cv::Point(4, vpos), CV_FONT_HERSHEY_COMPLEX, 0.5, color_red, 1);

		vpos -= (font_small_size.height + 4);
		// ---
		string thresh_str = "Threshold: " + to_string(threshold);
		putText(cvtImg, thresh_str, cv::Point(4, vpos), CV_FONT_HERSHEY_COMPLEX, 0.5, color_red, 1);

		rectangle(cvtImg, cv::Rect(maxScoreXPos, maxScoreYPos, ROIPIXWIDTH, ROIPIXHEIGHT), color_red, 1);
		vpos -= (font_small_size.height + 4);

		// --- 
		rectangle(cvtImg, cv::Rect(4, vpos - CELLSIZE, CELLSIZE, CELLSIZE), color_blue, 0);

		string b_minmax_str = "MaxScore: " + std::to_string(b_max) + ", MinScore: " + std::to_string(b_min);
		putText(cvtImg, b_minmax_str, cv::Point(4 + CELLSIZE, vpos), CV_FONT_HERSHEY_COMPLEX, 0.5, color_blue, 1);
		vpos -= (mymax(font_small_size.height, CELLSIZE) + 4);
		// --- 

		rectangle(cvtImg, cv::Rect(4, vpos - CELLSIZE, CELLSIZE, CELLSIZE), color_green, 0);

		string g_minmax_str = "MaxScore: " + std::to_string(g_max) + ", MinScore: " + std::to_string(g_min);
		putText(cvtImg, g_minmax_str, cv::Point(4 + CELLSIZE, vpos), CV_FONT_HERSHEY_COMPLEX, 0.5, color_green, 1);
		vpos -= (mymax(font_small_size.height, CELLSIZE) + 4);
		// --- 

		rectangle(cvtImg, cv::Rect(4, vpos - CELLSIZE, CELLSIZE, CELLSIZE), color_red, 0);

		string r_minmax_str = "MaxScore: " + std::to_string(r_max) + ", MinScore: " + std::to_string(r_min);
		putText(cvtImg, r_minmax_str, cv::Point(4 + CELLSIZE, vpos), CV_FONT_HERSHEY_COMPLEX, 0.5, color_red, 1);
		vpos -= (mymax(font_small_size.height, CELLSIZE) + 4);

		imshow("dotPOut", cvtImg);
		waitKey(0);
	}
