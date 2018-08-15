#pragma once
#include <opencv2/opencv.hpp>
#include "Interface.h"
#include "hpc.h"

extern const std::string ImageStr;
extern const std::string Color0Str;
extern const std::string Color1Str;
extern const std::string Color2Str;

extern const std::string ResultImageStr;
extern const std::string MajorStr;

extern const std::string KernelStr;

namespace DebuggableAMP
{
template <typename dataType>
class ImageMajorColors : public IAlgorithm
{
public:
    ImageMajorColors()
        : m_pAxis0(nullptr)
        , m_pAxis1(nullptr)
        , m_pAxis2(nullptr) {};

    virtual void SetParameters(ParameterTable&) override;
    virtual Error Process() override;

protected:
    cv::Mat m_rgb;
    cv::Vec3f* m_pAxis0;
    cv::Vec3f* m_pAxis1;
    cv::Vec3f* m_pAxis2;
};

template <typename dataType, unsigned int dim>
class BGR2Gray : public IAlgorithm
{
public:
    BGR2Gray() : m_pOutput(nullptr)
    {};

    virtual void SetParameters(ParameterTable&) override;
    virtual Error Process() override;

protected:
    cv::Mat m_rgb;
    cv::Mat* m_pOutput;
    cv::Vec<float, (int)dim> m_major;
};

std::vector<float>
GetGaussianKernel(int halfRow, int halfCol, float sigma);

template <typename dataType, unsigned int dim
        , unsigned int imageRowSize, unsigned int imageColSize
        , unsigned int rowBlockNumber, unsigned int colBlockNumber
        , unsigned int windowRowSize, unsigned int windowColSize>
class ImageFilter : public IAlgorithm
{
public:
    ImageFilter() 
    : m_pImage(nullptr){};

    virtual void SetParameters(ParameterTable&) override;
    virtual Error Process() override;
protected:
    cv::Mat* m_pImage;
    cv::Mat m_kernel;
};

};