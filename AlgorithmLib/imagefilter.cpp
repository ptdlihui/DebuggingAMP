#include "Methods.h"
#include "hpcutil.h"
#include <string>


const std::string KernelStr("Kernel");

template <typename dataType, unsigned int dim, int rowTileSize, int colTileSize>
class ImageFilterKernel
{
public:
    typedef typename hpc::BasicTypeTraits<dataType, dim>::VectorType PixelType;
    typedef typename hpc::BasicTypeTraits<float, dim>::VectorType FloatType;
    typedef typename hpc::Buffer<PixelType, 2> ImageType;

    ImageFilterKernel(const hpc::Buffer<float, 2>& kernel, const ImageType& source, ImageType& result)
        : m_kernel(kernel)
        , m_image(source)
        , m_result(result)
    {}

    void operator()(block_index<rowTileSize, colTileSize> t_idx) const kernel_code
    {
        block_shared FloatType ImageBlock[rowTileSize][colTileSize];

        int halfRowKernel = m_kernel.extent[0] / 2;
        int halfColKernel = m_kernel.extent[1] / 2;

        int rowBoundStart = halfRowKernel;
        int rowBoundEnd = rowTileSize - halfRowKernel;
        int colBoundStart = halfColKernel;
        int colBoundEnd = colTileSize - halfColKernel;

        int realTileRow = rowTileSize - 2 * halfRowKernel;
        int realTIleCol = colTileSize - 2 * halfColKernel;

        int realGlobalRowBase = realTileRow * t_idx.tile[0];
        int realGlobalColBase = realTIleCol * t_idx.tile[1];

        hpc::gpu::index<2> realGlobal = hpc::gpu::index<2>(realGlobalRowBase + t_idx.local[0] - rowBoundStart
            , realGlobalColBase + t_idx.local[1] - colBoundStart);

        if (realGlobal[0] < 0)
            realGlobal[0]= 0;

        if (realGlobal[0] >= m_image.extent[0])
            realGlobal[0] = m_image.extent[0] - 1;

        if (realGlobal[1] < 0)
            realGlobal[1] = 0;

        if (realGlobal[1] >= m_image.extent[1])
            realGlobal[1] = m_image.extent[1] - 1;
        
        ImageBlock[t_idx.local[0]][t_idx.local[1]] = FloatType(m_image[realGlobal]);

        hpc::kernel_sync::wait(t_idx);

        if (t_idx.local[0] >= rowBoundStart
            && t_idx.local[0] < rowBoundEnd
            && t_idx.local[1] >= colBoundStart
            && t_idx.local[1] < colBoundEnd)
        {
            FloatType ret = { 0 };
            hpc::gpu::index<2> local = t_idx.local;

            for (int row = -halfRowKernel ; row < (halfRowKernel + 1); row ++)
                for (int col = -halfColKernel; col < (halfColKernel + 1); col++)
                {
                    hpc::gpu::index<2> kernelLocal(row + halfRowKernel, col + halfColKernel);
                    float weight = m_kernel(kernelLocal);

                    int rowInCache = local[0] + row;
                    int colInCache = local[1] + col;

                    ret = ret + weight * ImageBlock[rowInCache][colInCache];
                }


            m_result[realGlobal] = PixelType(ret);
        }

        hpc::kernel_sync::wait(t_idx);
        ImageBlock[t_idx.local[0]][t_idx.local[1]] = { 0 };


    }
protected:
    hpc::Buffer<float, 2>::const_kernel_type m_kernel;
    typename hpc::Buffer<PixelType, 2>::const_kernel_type m_image;
    typename hpc::Buffer<PixelType, 2>::kernel_type m_result;

};

namespace DebuggableAMP
{
template <typename dataType, unsigned int dim
    , unsigned int imageRowSize, unsigned int imageColSize
    , unsigned int rowBlockNumber, unsigned int colBlockNumber
    , unsigned int windowRowSize, unsigned int windowColSize>
void 
ImageFilter<dataType, dim, imageRowSize, imageColSize, rowBlockNumber, colBlockNumber, windowRowSize, windowColSize>::SetParameters(ParameterTable& params)
{
    if (params.find(ImageStr) != params.end())
    {
        m_pImage = (params[ImageStr].AsPointer<cv::Mat>());
        assert(m_pImage->rows == imageRowSize);
        assert(m_pImage->cols == imageColSize);
        assert(m_pImage->rows % rowBlockNumber == 0);
        assert(m_pImage->cols % colBlockNumber == 0);
    }

    if (params.find(KernelStr) != params.end())
    {
        m_kernel = *(params[KernelStr].AsPointer<cv::Mat>());
        assert(m_kernel.type() == CV_32FC1);
        assert(m_kernel.rows == windowRowSize);
        assert(m_kernel.cols == windowColSize);
    }
}

template void ImageFilter<int, 3, 1080, 1920, 60, 120, 5, 5>::SetParameters(ParameterTable&);
template void ImageFilter<int, 3, 640, 480, 160, 120, 5, 5>::SetParameters(ParameterTable&);

template <typename dataType, unsigned int dim
    , unsigned int imageRowSize, unsigned int imageColSize
    , unsigned int rowBlockNumber, unsigned int colBlockNumber
    , unsigned int windowRowSize, unsigned int windowColSize>
IAlgorithm::Error
ImageFilter<dataType, dim, imageRowSize, imageColSize, rowBlockNumber, colBlockNumber, windowRowSize, windowColSize>::Process()
{
    typedef typename hpc::BasicTypeTraits<dataType, dim>::VectorType PixelType;
    hpc::DeviceContext context;

    std::shared_ptr<hpc::Buffer<float, 2>> kernelBuffer = context.Create2DBuffer(windowRowSize, windowColSize, false);
    kernelBuffer->copy_from_cpu(reinterpret_cast<float*>(m_kernel.data));

    std::shared_ptr<hpc::Buffer<PixelType, 2>> source = context.Create2DBuffer(imageRowSize, imageColSize, false);
    source->copy_from_cpu(reinterpret_cast<PixelType*>(m_pImage->data));

    std::shared_ptr<hpc::Buffer<PixelType, 2>> result = context.Create2DBuffer(imageRowSize, imageColSize, false);

    constexpr int sizePerRow = imageRowSize / rowBlockNumber;
    constexpr int sizePerCol = imageColSize / colBlockNumber;

    constexpr int halfKernelRow = windowRowSize / 2;
    constexpr int halfKernelCol = windowColSize / 2;

    constexpr int realTileSizePerRow = sizePerRow + 2 * halfKernelRow;
    constexpr int realTileSizePerCol = sizePerCol + 2 * halfKernelCol;

    hpc::gpu::extent<2> realExtent(realTileSizePerRow * rowBlockNumber, realTileSizePerCol * colBlockNumber);
    ImageFilterKernel<dataType, dim, realTileSizePerRow, realTileSizePerCol> kernel(*kernelBuffer, *source, *result);

    ULONGLONG start = GetTickCount64();
    context.EnqueueTask(realExtent.tile<realTileSizePerRow, realTileSizePerCol>(), kernel);
    std::cout << "Elapse : " << GetTickCount64() - start << std::endl;

    result->copy_to_cpu(reinterpret_cast<PixelType*>(m_pImage->data));

    return eSuccess;
}
template IAlgorithm::Error ImageFilter<int, 3, 1080, 1920, 60, 120, 5, 5>::Process();
template IAlgorithm::Error ImageFilter<int, 3, 640, 480, 160, 120, 5, 5>::Process();
}