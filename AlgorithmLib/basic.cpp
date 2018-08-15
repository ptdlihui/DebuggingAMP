#include "Methods.h"
#include "hpcutil.h"
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <opencv2/core/matx.hpp>

const std::string ImageStr("Image");
const std::string Color0Str("Color0");
const std::string Color1Str("Color1");
const std::string Color2Str("Color2");

namespace DebuggableAMP
{
template <typename dataType>
void
ImageMajorColors<dataType>::SetParameters(ParameterTable& params)
{
    if (params.find(ImageStr) != params.cend())
    {
        m_rgb = *(params[ImageStr].AsPointer<cv::Mat>());
    }

    if (params.find(Color0Str) != params.cend())
    {
        m_pAxis0 = params[Color0Str].AsPointer<cv::Vec3f>();
    }

    if (params.find(Color1Str) != params.cend())
    {
        m_pAxis1 = params[Color1Str].AsPointer<cv::Vec3f>();
    }

    if (params.find(Color2Str) != params.cend())
    {
        m_pAxis2 = params[Color2Str].AsPointer<cv::Vec3f>();
    }
}

template void ImageMajorColors<int>::SetParameters(ParameterTable& params);

template <typename dataType>
IAlgorithm::Error 
ImageMajorColors<dataType>::Process()
{
    typedef cv::Vec<dataType, 3> PixelType;
    Eigen::Matrix3d covariance;
    covariance.setZero();

    double weight = 1 / (double)(m_rgb.cols * m_rgb.rows);

    for (int x = 0; x < m_rgb.cols ; x++)
        for (int y = 0; y < m_rgb.rows; y++)
        {
            PixelType p = m_rgb.at<PixelType>(cv::Point(x, y));

            double xx = p(0) * p(0) * weight;
            double xy = p(0) * p(1) * weight;
            double xz = p(0) * p(2) * weight;

            double yy = p(1) * p(1) * weight;
            double yz = p(1) * p(2) * weight;

            double zz = p(2) * p(2) * weight;

            covariance(0, 0) += xx;
            covariance(0, 1) += xy;
            covariance(0, 2) += xz;

            covariance(1, 1) += yy;
            covariance(1, 2) += yz;

            covariance(2, 2) += zz;
        }

    
    covariance(1, 0) = covariance(0, 1);

    covariance(2, 0) = covariance(0, 2);
    covariance(2, 1) = covariance(1, 2);

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigenSolver(covariance);

    Eigen::Matrix3d vectors = eigenSolver.eigenvectors();
    Eigen::Vector3d values = eigenSolver.eigenvalues();

    Eigen::Vector3d vector0 = vectors.col(0);
    Eigen::Vector3d vector1 = vectors.col(1);
    Eigen::Vector3d vector2 = vectors.col(2);
    vector0.normalize();
    vector1.normalize();
    vector2.normalize();

    if (m_pAxis0)
    {
        *m_pAxis0 = cv::Vec3f((float)vector0(0), (float)vector0(1), (float)vector0(2));
    }

    if (m_pAxis1)
    {
        *m_pAxis1 = cv::Vec3f((float)vector1(0), (float)vector1(1), (float)vector1(2));
    }

    if (m_pAxis2)
    {
        *m_pAxis2 = cv::Vec3f((float)vector2(0), (float)vector2(1), (float)vector2(2));
    }

    return IAlgorithm::eSuccess;
}

template
IAlgorithm::Error ImageMajorColors<int>::Process();


std::vector<float>
GetGaussianKernel(int halfRow, int halfCol, float sigma)
{
    int row = 2 * halfRow + 1;
    int col = 2 * halfCol + 1;

    std::vector<float> cpuBuffer;
    cpuBuffer.resize(row * col);

    float sum = 0;

    float weight = 0.5f * 1.f / (CV_PI * sigma * sigma);
    float expWeight = 0.5f * 1 / (sigma * sigma);
    for (int y = -halfRow; y < halfRow + 1; y++)
        for (int x = -halfCol; x < halfCol + 1; x++)
        {
            int index = (y + halfRow) * (2 * halfCol + 1) + x + halfCol;
            cpuBuffer[index] = weight * std::expf(-(x * x + y * y) * expWeight);
            sum += cpuBuffer[index];
        }

    sum = 1.f / sum;
    for (size_t i = 0; i < cpuBuffer.size(); i++)
        cpuBuffer[i] *= sum;

    return cpuBuffer;
}
}