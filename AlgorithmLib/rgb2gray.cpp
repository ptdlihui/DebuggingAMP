#include "Methods.h"
#include "hpcutil.h"

const std::string ResultImageStr("ResultImage");
const std::string MajorStr("Major");

#define PRECISION 1000
const int MaxColor = (int)std::round(std::sqrt(255 * 255 * 3));


template <typename dataType, unsigned int dim>
class BGR2GrayKernel
{
public:
    typedef typename hpc::BasicTypeTraits<dataType, 3>::VectorType PixelType;
    typedef typename hpc::Buffer<PixelType, 2> SourceImageType;
    typedef typename hpc::BasicTypeTraits<float, dim>::VectorType FloatVectorType;

    BGR2GrayKernel(const SourceImageType& source, FloatVectorType& major, hpc::Buffer<float, 2>& output, hpc::Buffer<int, 1>& minmax)
        : m_sourceImage(source)
        , m_major(major)
        , m_output(output)
        , m_minmax(minmax)
    {}

    void operator()(hpc::gpu::index<2> idx) const kernel_code
    {
        const PixelType pixel = m_sourceImage[idx];

        FloatVectorType fPixel(pixel);

        float value = hpc::BasicTypeTraits<float, dim>::dot(fPixel, m_major);

        m_output[idx] = value;

        int scale = (int)hpc::kernel_math::ceil(value * PRECISION + 0.5f);

        hpc::kernel_atomic::atomic_min(&m_minmax[0], scale);
        hpc::kernel_atomic::atomic_max(&m_minmax[1], scale);
    }
protected:
    typename SourceImageType::const_kernel_type m_sourceImage;
    typename FloatVectorType m_major;
    typename hpc::Buffer<float, 2>::kernel_type m_output;
    typename hpc::Buffer<int, 1>::kernel_type m_minmax;
};

class NormalizeKernel
{
public:
    NormalizeKernel(hpc::Buffer<float, 2>& image, hpc::Buffer<int, 1>& bound)
        : m_image(image)
        , m_bound(bound)
    {}

    void operator()(hpc::gpu::index<2> idx) const kernel_code
    {
        float pixel = m_image[idx] * PRECISION;
        float normalize = (pixel - m_bound[0]) / (float)(m_bound[1] - m_bound[0]);

        normalize = (hpc::kernel_math::max)(0, normalize);
        normalize = (hpc::kernel_math::min)(1.f, normalize);
        m_image[idx] = normalize;
    }


protected:
    typename hpc::Buffer<float, 2>::kernel_type m_image;
    typename hpc::Buffer<int, 1>::kernel_type m_bound;
};

namespace DebuggableAMP
{

template <typename dataType, unsigned int dim>
void
BGR2Gray<dataType, dim>::SetParameters(ParameterTable& params)
{
    if (params.find(ImageStr) != params.end())
    {
        m_rgb = *(params[ImageStr].AsPointer<cv::Mat>());
    }

    if (params.find(ResultImageStr) != params.end())
    {
        m_pOutput = params[ResultImageStr].AsPointer<cv::Mat>();
    }

    if (params.find(MajorStr) != params.end())
    {
        m_major = *(params[MajorStr].AsPointer<cv::Vec<float, (int)dim>>());
    }
}

template void BGR2Gray<int, 3>::SetParameters(ParameterTable&);


template <typename dataType, unsigned int dim>
IAlgorithm::Error
BGR2Gray<dataType, dim>::Process()
{
    hpc::DeviceContext context;
    typedef typename hpc::BasicTypeTraits<dataType, dim>::VectorType vectorType;
    std::shared_ptr<typename hpc::Buffer<BGR2GrayKernel<dataType, dim>::PixelType, 2>>
        gpuSource = context.Create2DBuffer(m_rgb.rows, m_rgb.cols, false);

    gpuSource->copy_from_cpu(reinterpret_cast<vectorType *>(m_rgb.data));

    std::shared_ptr<hpc::Buffer<float, 2>> gpuResult = context.Create2DBuffer(m_rgb.rows, m_rgb.cols, false);
    std::shared_ptr<hpc::Buffer<int, 1>> gpuMinmax = context.Create1DBuffer(2, false);

    int minmax[2] = { 9999999, -9999999 };
    gpuMinmax->copy_from_cpu(minmax);

    typename hpc::BasicTypeTraits<float, dim>::VectorType major;
    std::memcpy(&major, &m_major, sizeof(major));

    BGR2GrayKernel<dataType, dim> toGraykernel(*gpuSource, major, *gpuResult, *gpuMinmax);
    context.EnqueueTask(gpuSource->extent, toGraykernel);
    
    NormalizeKernel normalizeKernel(*gpuResult, *gpuMinmax);
    context.EnqueueTask(gpuResult->extent, normalizeKernel);

    gpuMinmax->copy_to_cpu(minmax);

    gpuResult->copy_to_cpu(reinterpret_cast<float*>(m_pOutput->data));

    return eSuccess;
}

template IAlgorithm::Error BGR2Gray<int, 3>::Process();

}