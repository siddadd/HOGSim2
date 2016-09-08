// hwHOGDll.h

#ifdef HWHOGDLL_EXPORTS
#define HWHOGDLL_API __declspec(dllexport)
#else
#define HWHOGDLL_API __declspec(dllimport)
#endif

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cstdint>

using namespace std;
using namespace cv;

class IHOG
{
public:
	// Abstract Class

	virtual ~IHOG() {};
	virtual __int32* RHA(Mat& imageData) = 0;
	virtual Mat NCA(__int32* Bins, const int roi_pix_height, const int roi_pix_width, const int mode) = 0;

	int num_features;
	Mat svm;
};

// Derived Classes

// 1. Pedro
class HardwarePedroHOG : public IHOG
{
private:
	bool hog_debug;
	int num_bins;
	int kernel_size;
	int cell_size;
	int image_height;
	int image_width;
	int image_channels;


	int hw_rha_nca_scalefactor;

	bool enable_projection;
	int nca_debug;

	int hw_num_bins;
	int hw_scale_adjust;
	int hw_epsilon;
	int hw_hysterisis;
	int hw_nmlz_scale;
	double hw_projection_scale;
	double hw_texton_scale;

	__int16* createDerivativeFilter_x();
	__int16* createDerivativeFilter_y();
	__int32* computeDerivative(Mat& imageData, __int16* Kernel);
	tuple<__int64*, __int32*, __int32*> computeDominantChannel(Mat& imageData);
	tuple<__int32*, __int32*> computeGradients(__int64*, __int32*, __int32*);
	__int32* computeHistogram(__int32*, __int32*);
	__int32* computeRawHistogramFeatures(Mat& imageData);
	int projectFeatures();

public:
	HWHOGDLL_API HardwarePedroHOG();
	HWHOGDLL_API ~HardwarePedroHOG();
	HWHOGDLL_API virtual __int32* RHA(Mat& imageData);
	HWHOGDLL_API virtual Mat NCA(__int32* Bins, const int roi_pix_height, const int roi_pix_width, const int mode);
};

// 2. Dalal
class HardwareDalalHOG : public IHOG
{
private:
	bool hog_debug;
	int num_bins;
	int kernel_size;
	int cell_size;
	int image_height;
	int image_width;
	int image_channels;

	int hw_rha_nca_scalefactor;

	bool enable_projection;
	int nca_debug;

	int hw_num_bins;
	int hw_scale_adjust;
	int hw_epsilon;
	int hw_hysterisis;
	int hw_nmlz_scale;

	// Not used 
	double hw_projection_scale;
	double hw_texton_scale;

	__int16* createDerivativeFilter_x();
	__int16* createDerivativeFilter_y();
	__int32* computeDerivative(Mat& imageData, __int16* Kernel);
	tuple<__int64*, __int32*, __int32*> computeDominantChannel(Mat& imageData);
	tuple<__int32*, __int32*> computeGradients(__int64*, __int32*, __int32*);
	__int32* computeHistogram(__int32*, __int32*);
	__int32* computeRawHistogramFeatures(Mat& imageData);

public:
	HWHOGDLL_API HardwareDalalHOG();
	HWHOGDLL_API ~HardwareDalalHOG();
	HWHOGDLL_API virtual __int32* RHA(Mat& imageData);
	HWHOGDLL_API virtual Mat NCA(__int32* Bins, const int roi_pix_height, const int roi_pix_width, const int mode);
};

/*---------------------------------------------------------------*/