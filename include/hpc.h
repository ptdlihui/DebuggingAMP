#pragma once

#include <amp.h>
#include <amp_graphics.h>
#include <amp_math.h>
#include <amp_short_vectors.h>
#include <mutex>
#include "hpcdebug.h"

//#define GPU_SIMU

#define cpu_only restrict(cpu)
#define gpu_only restrict(amp)
#define cpu_gpu restrict(cpu, amp)

#ifndef GPU_SIMU
#define kernel_code gpu_only
#define block_shared tile_static
#define block_index hpc::gpu::tiled_index
#else
#define kernel_code cpu_only
#define block_shared static
#define block_index hpc::cpu_tiled_index
#endif


template <typename T>
class MemoryBlock
{
public:
    MemoryBlock() : m_pData(nullptr), m_number(0), m_size(0) {};
    MemoryBlock(unsigned int number)
        : m_pData(nullptr)
        , m_number(0)
        , m_size(0)
    {
        Create(number);
    }

    virtual ~MemoryBlock()
    {
        Delete();
    }
    void Create(unsigned int number)
    {
        if (number != m_number)
        {
            Delete();
            m_size = number * sizeof(T);
            m_number = number;
            m_pData = (T*)std::malloc(m_size);
        }
    }

    void Delete()
    {
        if (m_pData)
        {
            std::free(m_pData);
            m_pData = nullptr;
            m_size = 0;
            m_number = 0;
        }
    }

    size_t DataSize() const
    {
        return m_size;
    }

    unsigned int DataNumber() const
    {
        return m_number;
    }

    T* Data()
    {
        return m_pData;
    }

    const T* Data() const
    {
        return m_pData;
    }

    const T& operator[](const int i) const
    {
        return m_pData[i];
    }

    T& operator[](const int i)
    {
        return m_pData[i];
    }

protected:
    T* m_pData;
    size_t m_size;
    unsigned int m_number;
};


namespace hpc
{
namespace gpu = concurrency;
namespace fast_math = concurrency::fast_math;
namespace precise_math = concurrency::precise_math;
namespace graphics = concurrency::graphics;
namespace direct3d = concurrency::direct3d;

typedef graphics::uint_2 uint2;
typedef graphics::uint_3 uint3;
typedef graphics::uint_4 uint4;

typedef graphics::int_2 int2;
typedef graphics::int_3 int3;
typedef graphics::int_4 int4;

typedef graphics::float_2 float2;
typedef graphics::float_3 float3;
typedef graphics::float_4 float4;

typedef graphics::double_2 double2;
typedef graphics::double_3 double3;
typedef graphics::double_4 double4;

typedef gpu::accelerator Device;
typedef gpu::accelerator_view DeviceInstance;

template <typename T, int D>
class Buffer : public gpu::array<T, D>
{
public:
    typedef gpu::array<T, D> gpu_type;
    typedef gpu::array_view<T, D> gpu_view_type;
#ifndef GPU_SIMU
    typedef typename gpu_type& kernel_type;
    typedef typename const gpu_type& const_kernel_type;
#else
    typedef typename gpu_view_type kernel_type;
    typedef typename const gpu::array_view<const T, D> const_kernel_type;
#endif

    typedef MemoryBlock<T> cpumemory_type;

    Buffer(const Buffer<T, D>& src)
        : gpu_type(src)
        , m_shared(src.m_shared)
        , m_staging(src.m_staging)
    {
        init();
    }


    Buffer(int d0, DeviceInstance device, bool shared, bool staging) cpu_only
        : gpu_type(d0, device, shared ? gpu::access_type_read_write : gpu::access_type_auto)
        , m_shared(shared)
        , m_staging(staging)
    {
        init();
    }

    Buffer(int d0, int d1, DeviceInstance device, bool shared, bool staging) cpu_only
        : gpu_type(d0, d1, device, shared ? gpu::access_type_read_write : gpu::access_type_auto)
        , m_shared(shared)
        , m_staging(staging)
    {
        init();
    }

    Buffer(int d0, int d1, int d2, DeviceInstance device, bool shared, bool staging) cpu_only
        : gpu_type(d0, d1, d2, device, shared ? gpu::access_type_read_write : gpu::access_type_auto)
        , m_shared(shared)
        , m_staging(staging)
    {
        init();
    }

    T* stage_data() cpu_only
    {
        return m_shared ? data() : m_staging ? m_cpumap.Data() : nullptr;
    }

    const T* stage_data() const cpu_only
    {
        return m_shared ? data() : m_staging ? m_cpumap.Data() : nullptr;
    }

    void copy_to_cpu(T* pTarget) cpu_only
    {
        if (m_shared)
            ::memcpy_s(pTarget, sizeof(T) * extent.size() , data(), sizeof(T) * extent.size());
        else
            gpu::copy(*this, pTarget);
    }

    void copy_from_cpu(const T* pTarget) cpu_only
    {
        if (m_shared)
            ::memcpy_s(data(), sizeof(T) * extent.size(),  pTarget, sizeof(T) * extent.size());
        else
            gpu::copy_async(pTarget, *this);
    }

    void sync_stage_to_cpu() cpu_only
    {
        if (m_staging)
        {
            gpu::copy(*this, m_cpumap.Data());
        }
    }

    void sync_stage_to_gpu() cpu_only
    {
        if (m_staging)
        {
            gpu::copy_async(m_cpumap.Data(), *this);
        }
    }
protected:
    void init() cpu_only
    {
        if (m_shared) m_staging = false;
        if (m_staging)
        {
            m_cpumap.Create(extent.size());
        }
    }

protected:
    cpumemory_type m_cpumap;
    bool m_shared;
    bool m_staging;
};

class BufferParams
{
public:
    BufferParams(int dim0, int dim1, int dim2, hpc::DeviceInstance device, bool shared, bool staging)
        : m_dim0(dim0)
        , m_dim1(dim1)
        , m_dim2(dim2)
        , m_device(device)
        , m_shared(shared)
        , m_staging(staging)
    {
    }

protected:
    int m_dim0;
    int m_dim1;
    int m_dim2;
    hpc::DeviceInstance m_device;
    bool m_shared;
    bool m_staging;
};

template <int D>
class BufferCreationParams : public BufferParams
{

};

template <>
class BufferCreationParams<1> : public BufferParams
{
public:
    BufferCreationParams(int dim0, hpc::DeviceInstance device, bool shared, bool staging)
        : BufferParams(dim0, 0, 0, device, shared, staging)
    {}


    template <typename T>
    operator std::shared_ptr<Buffer<T, 1>> ()
    {
        return std::make_shared<Buffer<T, 1>>(m_dim0, m_device, m_shared, m_staging);
    }
};

template <>
class BufferCreationParams<2> : public BufferParams
{
public:
    BufferCreationParams(int dim0, int dim1, hpc::DeviceInstance device, bool shared, bool staging)
        : BufferParams(dim0, dim1, 0, device, shared, staging)

    {};

    template <typename T>
    operator std::shared_ptr<Buffer<T, 2>>()
    {
        return std::make_shared<Buffer<T, 2>>(m_dim0, m_dim1, m_device, m_shared, m_staging);
    }
};

template <>
class BufferCreationParams<3> : public BufferParams
{
public:
    BufferCreationParams(int dim0, int dim1, int dim2, hpc::DeviceInstance device, bool shared, bool staging)
        : BufferParams(dim0, dim1, dim2, device, shared, staging)
    {};

    template <typename T>
    operator std::shared_ptr<Buffer<T, 3>>()
    {
        return std::make_shared<Buffer<T, 3>>(m_dim0, m_dim1, m_dim2, m_device, m_shared, m_staging);
    }
};


class DeviceContext
{
public:
    DeviceContext()
        : m_device(Device::get_auto_selection_view())
        , m_shared(false)
    {
    }
    DeviceContext(DeviceInstance device, bool shared)
        : m_device(device)
        , m_shared(shared)
    {}

    DeviceContext(const DeviceContext& context)
        : m_device(context.m_device)
        , m_shared(context.m_shared)
    {}

    const DeviceContext& operator = (const DeviceContext& right)
    {
        if (this == &right)
            return *this;

        m_device = right.m_device;
        m_shared = right.m_shared;

        return *this;
    }

    BufferCreationParams<1> Create1DBuffer(int d0, bool staging)
    {
        return BufferCreationParams<1>(d0, m_device, m_shared, staging);
    }

    BufferCreationParams<2> Create2DBuffer(int d0, int d1, bool staging)
    {
        return BufferCreationParams<2>(d0, d1, m_device, m_shared, staging);
    }

    BufferCreationParams<3> Create3DBuffer(int d0, int d1, int d2, bool staging)
    {
        return BufferCreationParams<3>(d0, d1, d2, m_device, m_shared, staging);
    }

    template <int D, typename KernelType>
    void EnqueueTask(const gpu::extent<D>& domain, const KernelType& kernel)
    {
#ifndef GPU_SIMU
        gpu::parallel_for_each(m_device, domain, kernel);
#else
        SimulateTask(domain, kernel);
#endif
    }

    template <int D, typename KernelType>
    void SimulateTask(const gpu::extent<D>& domain, const KernelType& kernel)
    {

    }

    template <typename KernelType>
    void SimulateTask(const gpu::extent<1>& domain, const KernelType& kernel)
    {
#pragma omp parallel for
        for (int i = 0; i < domain[0]; i++)
        {
            hpc::gpu::index<1> idx(i);
            kernel(idx);
        }
    }

    template <typename KernelType>
    void SimulateTask(const gpu::extent<2>& domain, const KernelType& kernel)
    {
#pragma omp parallel for
        for (int i = 0; i < domain[0]; i++)
            for (int j = 0; j < domain[1]; j++)
            {
                hpc::gpu::index<2> idx(i, j);
                kernel(idx);
            }
    }


    template <typename KernelType>
    void SimulateTask(const gpu::extent<3>& domain, const KernelType& kernel)
    {
#pragma omp parallel for
        for (int i = 0; i < domain[0]; i++)
            for (int j = 0; j < domain[1]; j++)
                for (int k = 0; k < domain[2]; k++)
                {
                    hpc::gpu::index<3> idx(i, j, k);
                    kernel(idx);
                }
    }

    template <int t0, typename KernelType>
    void EnqueueTask(const gpu::tiled_extent<t0>& domain, const KernelType& kernel)
    {
#ifndef GPU_SIMU
        gpu::parallel_for_each(m_device, domain, kernel);
#else
        hpc::TiledProcessor<KernelType, t0> cpuTiledProcessor(domain, kernel);
        cpuTiledProcessor.run();
#endif
    }

    template <int t0, int t1, typename KernelType>
    void EnqueueTask(const gpu::tiled_extent<t0, t1>& domain, const KernelType& kernel)
    {
#ifndef GPU_SIMU
        gpu::parallel_for_each(m_device, domain, kernel);
#else
        hpc::TiledProcessor<KernelType, t0, t1> cpuTiledProcessor(domain, kernel);
        cpuTiledProcessor.run();
#endif
    }

    template <int t0, int t1, int t2, typename KernelType>
    void EnqueueTask(const gpu::tiled_extent<t0, t1, t2>& domain, const KernelType& kernel)
    {
        gpu::parallel_for_each(m_device, domain, kernel);
    }

    void sync()
    {
        m_device.flush();
        m_device.wait();
    }

protected:
    DeviceInstance m_device;
    bool m_shared;
};

struct DeviceObjects
{
    Device FirstDevice;
    Device SecondDevice;
    Device DebugDevice;
    Device CPUDevice;
    bool SecondValid;
};

inline const DeviceObjects& GlobalDeviceObjects()
{
    static DeviceObjects objects;
    static bool init = false;

    if (!init)
    {
        objects.FirstDevice = Device(Device::default_accelerator);
        objects.DebugDevice = Device(Device::direct3d_ref);
        objects.CPUDevice = Device(Device::direct3d_warp);
        objects.SecondValid = false;
        std::vector<hpc::Device> devices = Device::get_all();
        for (auto& instance : devices)
        {
            if (instance.dedicated_memory > 0 && instance.has_display == false)
            {
                objects.SecondDevice = instance;
                objects.SecondValid = true;
                break;
            }
        }
        init = true;
    }

    return objects;
}
}