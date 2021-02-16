#include "opencv2/opencv.hpp"
#include "hpc.h"
#include "hpcutil.h"
#include "Methods.h"

int main()
{
    ParameterTable table;
    cv::Mat image = cv::imread(ResourcePath + std::string("girl.jpg"));
    cv::Mat imageInt;
    image.convertTo(imageInt, CV_32SC3);

    ParameterTable params;
    params[ImageStr].AsPointer<cv::Mat>(&imageInt);

    cv::Vec3f color0, color1, color2;
    params[Color0Str].AsPointer<cv::Vec3f>(&color0);
    params[Color1Str].AsPointer<cv::Vec3f>(&color1);
    params[Color2Str].AsPointer<cv::Vec3f>(&color2);

    DebuggableAMP::ImageMajorColors<int> GetMajorColor;
    GetMajorColor.SetParameters(params);
    GetMajorColor.Process();

    cv::Mat gray(image.rows, image.cols, CV_32FC1);

    params[ResultImageStr].AsPointer<cv::Mat>(&gray);
    params[MajorStr].AsPointer<cv::Vec3f>(&color2);

    DebuggableAMP::BGR2Gray<int, 3> toGray;
    toGray.SetParameters(params);
    toGray.Process();

    cv::Mat cvGray;
    cv::cvtColor(image, cvGray, cv::COLOR_BGR2GRAY);

    return -1;
}