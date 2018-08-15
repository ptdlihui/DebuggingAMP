#include "opencv2/opencv.hpp"
#include "hpc.h"
#include "hpcutil.h"
#include "Methods.h"

int main()
{
    ParameterTable table;
    cv::Mat image = cv::imread(ResourcePath + std::string("girl.jpg"));
    std::vector<float> gaussian = DebuggableAMP::GetGaussianKernel(2, 2, 2);
    cv::Mat gaussianMat(5, 5, CV_32FC1, gaussian.data());

#if 1
    cv::Mat imageInt;
    image.convertTo(imageInt, CV_32SC3);
    DebuggableAMP::ImageFilter<int, 3, 1080, 1920, 60, 120, 5, 5> filter;
#else
    cv::resize(image, image, cv::Size(480, 640));
    cv::Mat imageInt;
    image.convertTo(imageInt, CV_32SC3);
    DebuggableAMP::ImageFilter<int, 3, 640, 480, 160, 120, 5, 5> filter;
#endif
    ParameterTable params;
    params[ImageStr].AsPointer<cv::Mat>(&imageInt);
    params[KernelStr].AsPointer<cv::Mat>(&gaussianMat);

    filter.SetParameters(params);
    filter.Process();
    

    return 0;
}