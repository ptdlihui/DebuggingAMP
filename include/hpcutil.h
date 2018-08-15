#pragma once
#include "hpc.h"

#define Epsilon     0.00000001f
namespace hpc
{
    template <typename Type, unsigned int Depth>
    struct BasicTypeTraits
    {

    };

    template <typename Type>
    Type odevity(int i) cpu_gpu
    {
        if (i % 2 == 0)
            return Type(1);
        else
            return Type(-1);
    }

    template <typename Type, unsigned int Dim>
    struct Matrix
    {
        Type elem[Dim][Dim];

        Matrix() cpu_gpu
        {
            for (unsigned int i = 0; i < Dim; i++)
            {
                for (unsigned int j = 0; j < Dim; j++)
                    elem[i][j] = (Type)(0);
            }
        }

        const Matrix<Type, Dim>& operator *= (const Type right) cpu_gpu
        {
            for (unsigned int i = 0; i < Dim; i++)
            {
                for (unsigned int j = 0; j < Dim; j++)
                    elem[i][j] *= right;
            }

            return *this;
        }

        const Matrix<Type, Dim - 1> cofactor(unsigned int row, unsigned int col) const cpu_gpu
        {
            Matrix<Type, Dim - 1> ret;
            unsigned int ri = 0;
            unsigned int rj = 0;
            for (unsigned int i = 0; i < Dim; i++)
            {
                if (i == row) continue;
                for (unsigned int j = 0; j < Dim; j++)
                {
                    if (j == col) continue;
                    ret.elem[ri][rj] = elem[i][j];
                    rj++;
                }
                ri++;
                rj = 0;
            }

            return ret;
        }

        Type determinant() const cpu_gpu
        {
            Type ret = (Type)0;

            for (int j = 0; j < Dim; j++)
                ret += elem[0][j] * odevity<Type>(j) * cofactor(0, j).determinant();

            return ret;
        }

        const Matrix<Type, Dim> adjoint(void) const cpu_gpu
        {
            Matrix<Type, Dim> ret;

            for (int i = 0; i < Dim; i++)
                for (int j = 0; j < Dim; j++)
                    ret.elem[i][j] = odevity<Type>(i + j) * cofactor(j, i).determinant();

            return ret;
        }
    };


    template <typename Type>
    struct Matrix < Type, 2 >
    {
        Type elem[2][2];

        Matrix() cpu_gpu
        {
            elem[0][0] = elem[0][1] = elem[1][0] = elem[1][1] = 0;
        }

        const Matrix<Type, 2>& operator *= (const Type right) cpu_gpu
        {
            elem[0][0] *= right;
            elem[0][1] *= right;
            elem[1][0] *= right;
            elem[1][1] *= right;

            return *this;
        }

        const Matrix<Type, 1> cofactor(unsigned int row, unsigned int col) const cpu_gpu
        {
            static_assert(0 , "Not supposed to be called");
            return GPUMatrix<Type, 1>();
        }

        Type determinant() const cpu_gpu
        {
            return elem[0][0] * elem[1][1] - elem[0][1] * elem[1][0];
        }

        const Matrix<Type, 2> adjoint() const cpu_gpu
        {
            Matrix<Type, 2> ret;
            ret.elem[0][0] = elem[1][1];
            ret.elem[1][1] = elem[0][0];
            ret.elem[0][1] = -elem[0][1];
            ret.elem[1][0] = -elem[1][0];

            return ret;
        }
    };

    struct kernel_atomic
    {
        // Add
        static int add(int* dest, int value) cpu_only
        {
            int ret = InterlockedAdd(reinterpret_cast<LONG*>(dest), value);
            return ret - value;
        }

        static int add(int* dest, int value) gpu_only
        {
            return hpc::gpu::atomic_fetch_add(dest, value);
        }

        static unsigned int add(unsigned int* dest, unsigned int value) cpu_only
        {
            unsigned int ret = InterlockedAdd(reinterpret_cast<LONG*>(dest), value);
            return ret - value;
        }

        static unsigned int add(unsigned int* dest, unsigned int value) gpu_only
        {
            return hpc::gpu::atomic_fetch_add(dest, value);
        }

        // And
        static int and(int* dest, int value) cpu_only
        {
            return InterlockedAnd(reinterpret_cast<LONG*>(dest), value);
        }

        static int and(int* dest, int value) gpu_only
        {
            return hpc::gpu::atomic_fetch_and(dest, value);
        }

        static unsigned int and(unsigned int* dest, unsigned int value) cpu_only
        {
            return InterlockedAnd(reinterpret_cast<LONG*>(dest), value);
        }

        static unsigned int and(unsigned int* dest, unsigned int value) gpu_only
        {
            return hpc::gpu::atomic_fetch_and(dest, value);
        }

        //Or
        static int or(int* dest, int value) cpu_only
        {
            return InterlockedOr(reinterpret_cast<LONG*>(dest), value);
        }

        static int or (int* dest, int value) gpu_only
        {
            return hpc::gpu::atomic_fetch_or(dest, value);
        }

        static unsigned int or (unsigned int* dest, unsigned int value) cpu_only
        {
            return InterlockedOr(reinterpret_cast<LONG*>(dest), value);
        }

        static unsigned int or (unsigned int* dest, unsigned int value) gpu_only
        {
            return hpc::gpu::atomic_fetch_or(dest, value);
        }

        //Xor
        static int xor(int* dest, int value) cpu_only
        {
            return InterlockedXor(reinterpret_cast<LONG*>(dest), value);
        }

        static int xor(int* dest, int value) gpu_only
        {
            return hpc::gpu::atomic_fetch_xor(dest, value);
        }

        static unsigned int xor(unsigned int* dest, unsigned int value) cpu_only
        {
            return InterlockedXor(reinterpret_cast<LONG*>(dest), value);
        }

        static unsigned int xor(unsigned int* dest, unsigned int value) gpu_only
        {
            return hpc::gpu::atomic_fetch_xor(dest, value);
        }

        // Sub
        static int sub(int* dest, int value) cpu_only
        {
            int ret = InterlockedAdd(reinterpret_cast<LONG*>(dest), -value);
            return ret + value;
        }

        static int sub(int* dest, int value) gpu_only
        {
            return hpc::gpu::atomic_fetch_sub(dest, value);
        }

        static unsigned int sub(unsigned int* dest, unsigned int value) cpu_only
        {
            unsigned int ret = InterlockedAdd(reinterpret_cast<LONG*>(dest), (~value) + 1);
            return ret + value;
        }

        static unsigned int sub(unsigned int* dest, unsigned int value) gpu_only
        {
            return hpc::gpu::atomic_fetch_sub(dest, value);
        }

        //max
        static int atomic_max(int* dest, int value) cpu_only
        {
            int initialValue, maxValue;
            do
            {
                initialValue = *dest;
                maxValue = max(initialValue, value);
            } while (InterlockedCompareExchange(reinterpret_cast<long*>(dest), maxValue, initialValue) != initialValue);

            return initialValue;
        }

        static int atomic_max(int* dest, int value) gpu_only
        {
            return hpc::gpu::atomic_fetch_max(dest, value);
        }

        static unsigned int atomic_max(unsigned int* dest, unsigned int value) cpu_only
        {
            unsigned int initialValue, maxValue;
            do 
            {
                initialValue = *dest;
                maxValue = max(initialValue, value);
            } while (InterlockedCompareExchange(reinterpret_cast<unsigned long*>(dest), maxValue, initialValue) != initialValue);

            return initialValue;
        }

        static unsigned int atomic_max(unsigned int* dest, unsigned int value) gpu_only
        {
            return hpc::gpu::atomic_fetch_max(dest, value);
        }

        // min
        static int atomic_min(int* dest, int value) cpu_only
        {
            int initialValue, minValue;
            do
            {
                initialValue = *dest;
                minValue = min(initialValue, value);
            } while (InterlockedCompareExchange(reinterpret_cast<long*>(dest), minValue, initialValue) != initialValue);

            return initialValue;
        }

        static int atomic_min(int* dest, int value) gpu_only
        {
            return hpc::gpu::atomic_fetch_min(dest, value);
        }


        static unsigned int atomic_min(unsigned int* dest, unsigned int value) cpu_only
        {
            unsigned int initialValue, minValue;
            do
            {
                initialValue = *dest;
                minValue = min(initialValue, value);
            } while (InterlockedCompareExchange(reinterpret_cast<unsigned long*>(dest), minValue, initialValue) != initialValue);

            return initialValue;
        }

        static unsigned int atomic_min(unsigned int* dest, unsigned int value) gpu_only
        {
            return hpc::gpu::atomic_fetch_min(dest, value);
        }
    };

    struct kernel_math
    {
        static float sqrt(const float v) cpu_only
        {
            return std::sqrtf(v);
        }

        static float sqrt(const float v) gpu_only
        {
            return hpc::fast_math::sqrtf(v);
        }

        static float floor(const float v) cpu_only
        {
            return std::floorf(v);
        }

        static float floor(const float v) gpu_only
        {
            return hpc::fast_math::floorf(v);
        }

        static float ceil(const float v) cpu_only
        {
            return std::ceilf(v);
        }

        static float ceil(const float v) gpu_only
        {
            return hpc::fast_math::ceilf(v);
        }

        static float abs(const float v) cpu_only
        {
            return std::fabs(v);
        }

        static float abs(const float v) gpu_only
        {
            return hpc::fast_math::fabs(v);
        }

        static int abs(const int v) cpu_only
        {
            return std::abs(v);
        }

        static int abs(const int v) gpu_only
        {
            return hpc::direct3d::abs(v);
        }

        static float (max)(const float l, const float r) cpu_only
        {
            return (std::max)(l, r);
        }

        static float (max)(const float l, const float r) gpu_only
        {
            return hpc::fast_math::fmaxf(l, r);
        }

        static float (min)(const float l, const float r) cpu_only
        {
            return (std::min)(l, r);
        }

        static float (min)(const float l, const float r) gpu_only
        {
            return hpc::fast_math::fminf(l, r);
        }

        static float pow(const float x, const float y) cpu_only
        {
            return std::powf(x, y);
        }

        static float pow(const float x, const float y) gpu_only
        {
            return hpc::fast_math::powf(x, y);
        }

        static bool isnan(const float v) cpu_only
        {
            return std::isnan(v);
        }

        static bool isnan(const float v) gpu_only
        {
            return !!hpc::fast_math::isnan(v);
        }

        static float clamp(float low, float up, float value) cpu_gpu
        {
            return (min)((max)(low, value), up);
        }
    };

    struct kernel_sync
    {
        template <typename tile_index_type>
        static void wait(tile_index_type& t_index) cpu_only
        {
            t_index.wait();
        }

        template <typename tile_index_type>
        static void wait(tile_index_type& t_index) gpu_only
        {
            t_index.barrier.wait();
        }
    };

    template <>
    struct BasicTypeTraits< float, 1u>
    {
        typedef float VectorType;

        static void swap(float& left, float& right) cpu_gpu
        {
            float tmp = left;
            left = right;
            right = tmp;
        }

        static float (min)(const float& left, const float& right) cpu_gpu
        {
            return left < right ? left : right;
        }

        static float (max)(const float& left, const float& right) cpu_gpu
        {
            return left > right ? left : right;
        }

        static bool thresCheck(const float& value, const float& seed, const float& upperTol, const float& lowerTol) cpu_gpu
        {
            return ((value - seed) >= (-lowerTol)) && ((value - seed) <= (upperTol));
        }


    };

    template <>
    struct BasicTypeTraits<int, 1u>
    {
        typedef int VectorType;

        static void swap(int& left, int& right) cpu_gpu
        {
            int tmp = left;
            left = right;
            right = tmp;
        }

        static int (min)(const int& left, const int& right) cpu_gpu
        {
            return (left < right) ? left : right;
        }

        static int (max)(const int& left, const int& right) cpu_gpu
        {
            return (left > right) ? left : right;
        }


    };

    template <>
    struct BasicTypeTraits < float, 2u >
    {
        typedef float2 VectorType;
        static float dot(float2 left, float2 right) cpu_gpu
        {
            return left.x * right.x + left.y * right.y;
        }

        static float2 normalize(float2 v) cpu_gpu
        {
            float length = kernel_math::sqrt(dot(v, v));
            if (length < Epsilon)
                return v;

            float2 ret = v / length;
            return ret;
        }

        static float quadratic_form(float2 left, Matrix<float, 2> matrix, float2 right) cpu_gpu
        {
            float2 temp;
            temp.x = matrix.elem[0][0] * right.x + matrix.elem[0][1] * right.y;
            temp.y = matrix.elem[1][0] * right.x + matrix.elem[1][1] * right.y;

            return dot(left, temp);
        }

        static bool thresCheck(const float2& value, const float2& seed, const float& upperTol, const float& lowerTol) cpu_gpu
        {
            float2 temp = value - seed;
            return (temp.x >= -lowerTol && temp.x <= upperTol) && (temp.y >= -lowerTol && temp.y <= upperTol);
        }
    };

    template<>
    struct BasicTypeTraits < int, 3u>
    {
        typedef int3 VectorType;

        static int dot(int3 left, int3 right) cpu_gpu
        {
            return left.x * right.x + left.y * right.y + left.z * right.z;
        }

        static int quadratic_form(int3 left, Matrix<int, 3> matrix, int3 right) cpu_gpu
        {
            int3 temp;
            temp.x = matrix.elem[0][0] * right.x + matrix.elem[0][1] * right.y + matrix.elem[0][2] * right.z;
            temp.y = matrix.elem[1][0] * right.x + matrix.elem[1][1] * right.y + matrix.elem[1][2] * right.z;
            temp.z = matrix.elem[2][0] * right.x + matrix.elem[2][1] * right.y + matrix.elem[2][2] * right.z;

            return dot(left, temp);
        }
    };

    template<>
    struct BasicTypeTraits < float, 3u>
    {
        typedef float3 VectorType;

        static float dot(float3 left, float3 right) cpu_gpu
        {
            return left.x * right.x + left.y * right.y + left.z * right.z;
        }

        static float3 normalize(float3 v) cpu_gpu
        {
            float length = kernel_math::sqrt(dot(v, v));
            if (length < Epsilon)
                return v;

            float3 ret = v / length;
            return ret;
        }

        static float3 cross(float3 left, float3 right) cpu_gpu
        {
            return{ left.y * right.z - left.z * right.y
                , left.z * right.x - left.x * right.z
                , left.x * right.y - left.y * right.x };
        }

        static float distance(float3 left, float3 right) cpu_gpu
        {
            return kernel_math::sqrt(kernel_math::pow(left.x - right.x, 2.f)
                + kernel_math::pow(left.y - right.y, 2.f)
                + kernel_math::pow(left.z - right.z, 2.f));
        }

        static float maxAbs(float3 v) cpu_gpu
        {
            return (kernel_math::max)((kernel_math::max)(kernel_math::abs(v.x)
                                                    , kernel_math::abs(v.y))
                                                    , kernel_math::abs(v.z));
        }


        static float quadratic_form(float3 left, Matrix<float, 3> matrix, float3 right) cpu_gpu
        {
            float3 temp;
            temp.x = matrix.elem[0][0] * right.x + matrix.elem[0][1] * right.y + matrix.elem[0][2] * right.z;
            temp.y = matrix.elem[1][0] * right.x + matrix.elem[1][1] * right.y + matrix.elem[1][2] * right.z;
            temp.z = matrix.elem[2][0] * right.x + matrix.elem[2][1] * right.y + matrix.elem[2][2] * right.z;

            return dot(left, temp);
        }

        static bool thresCheck(const float3& value, const float3& seed, const float& upperTol, const float& lowerTol) cpu_gpu
        {
            float3 temp = value - seed;
            return (temp.x >= -lowerTol && temp.x <= upperTol)
                && (temp.y >= -lowerTol && temp.y <= upperTol)
                && (temp.z >= -lowerTol && temp.z <= upperTol);
        }

        static float3 multiply_matrix(const Matrix<float, 3>& matrix, const float3 right) cpu_gpu
        {
            float3 ret;
            ret.x = matrix.elem[0][0] * right.x + matrix.elem[0][1] * right.y + matrix.elem[0][2] * right.z;
            ret.y = matrix.elem[1][0] * right.x + matrix.elem[1][1] * right.y + matrix.elem[1][2] * right.z;
            ret.z = matrix.elem[2][0] * right.x + matrix.elem[2][1] * right.y + matrix.elem[2][2] * right.z;

            return ret;
        }
    };

    template<>
    struct BasicTypeTraits <float, 4u>
    {
        typedef float4 VectorType;
        static float dot(float4 left, float4 right) cpu_gpu
        {
            return left.x * right.x + left.y * right.y + left.z * right.z + left.w * right.w;
        }

        static float4 normalize(float4 v) cpu_gpu
        {
            float length = kernel_math::sqrt(dot(v, v));
            if (length < Epsilon)
                return v;

            float4 ret = v / length;
            return ret;
        }

        static float4 multiply_matrix(const Matrix<float, 4>& matrix, const float4 right) cpu_gpu
        {
            float4 ret;
            ret.x = matrix.elem[0][0] * right.x + matrix.elem[0][1] * right.y + matrix.elem[0][2] * right.z + matrix.elem[0][3] * right.w;
            ret.y = matrix.elem[1][0] * right.x + matrix.elem[1][1] * right.y + matrix.elem[1][2] * right.z + matrix.elem[1][3] * right.w;
            ret.z = matrix.elem[2][0] * right.x + matrix.elem[2][1] * right.y + matrix.elem[2][2] * right.z + matrix.elem[2][3] * right.w;
            ret.w = matrix.elem[3][0] * right.x + matrix.elem[3][1] * right.y + matrix.elem[3][2] * right.z + matrix.elem[3][3] * right.w;

            return ret;
        }


    };

    template<>
    struct BasicTypeTraits < double, 3u>
    {
        typedef double3 VectorType;

        static double dot(double3 left, double3 right) cpu_gpu
        {
            return left.x * right.x + left.y * right.y + left.z * right.z;
        }

        static double distance(double3 left, double3 right) gpu_only
        {
            return precise_math::sqrt(precise_math::pow(left.x - right.x, 2.0)
                + precise_math::pow(left.y - right.y, 2.0)
                + precise_math::pow(left.z - right.z, 2.0));
        }

        static double quadratic_form(double3 left, Matrix<double, 3> matrix, double3 right) cpu_gpu
        {
            double3 temp;
            temp.x = matrix.elem[0][0] * right.x + matrix.elem[0][1] * right.y + matrix.elem[0][2] * right.z;
            temp.y = matrix.elem[1][0] * right.x + matrix.elem[1][1] * right.y + matrix.elem[1][2] * right.z;
            temp.z = matrix.elem[2][0] * right.x + matrix.elem[2][1] * right.y + matrix.elem[2][2] * right.z;

            return dot(left, temp);
        }
    };
}


#define GET_CHAR_COMP(value, i) (((value) & (0xff << (i * 8))) >> (i * 8))
#define GET_R(value) GET_CHAR_COMP(value, 0)
#define GET_G(value) GET_CHAR_COMP(value, 1)
#define GET_B(value) GET_CHAR_COMP(value, 2)
#define GET_A(value) GET_CHAR_COMP(value, 3)

#define SET_CHAR_COMP(R, G, B, A) (((A) << 24) | ((B) << 16) | ((G) << 8) | R) 

#define GET_USHORT_COMP(value, i) (((value) & (0xffff << (i * 16))) >> (i * 16))
#define GET_LOW(value) GET_USHORT_COMP(value, 0)
#define GET_HIGH(value) GET_USHORT_COMP(value, 1)

#define SHORT_SIGN_MASK 0x8000

#define GET_SIGNED_SHORT(value) (((value) & SHORT_SIGN_MASK) == 0 ? (value) : ((0xffff << 16) | (value)))

#define SET_SHORT_COMP(L, H) (((H) << 16) | ((L) & 0xffff))

#define READ_CHAR_FROM_UINT_ARRAY(ARRAY, idx) ((((ARRAY)((idx) >> 2) & (0xFF << (((idx) & 0x3) << 3)))) >> (((idx) & 0x3) << 3))

#define WRITE_CHAR_TO_UINT_ARRAY(ARRAY, idx, val) \
hpc::kernel_atomic::and(&(ARRAY)[idx >> 2], ~(0xFF << ((idx & 0x3) << 3))); \
hpc::kernel_atomic::or(&(ARRAY)[idx >> 2], (val & 0xFF) << ((idx & 0x3) << 3));

#define READ_CHAR_FROM_UINT_ARRAY2(ARRAY, idx) (((ARRAY)(idx[0], idx[1] >> 2) & (0xFF << (((idx[1]) & 0x3) << 3))) >> (((idx[1]) & 0x3) << 3))


#define WRITE_CHAR_TO_UINT_ARRAY2(ARRAY, idx, val) \
hpc::kernel_atomic::and(&(ARRAY)(idx[0], idx[1] >> 2), ~(0xFF << ((idx[1] & 0x3) << 3))); \
hpc::kernel_atomic::or(&(ARRAY)(idx[0], idx[1] >> 2), (val & 0xFF) << ((idx[1] & 0x3) << 3));
