#pragma once

#ifndef EIGEN_NO_CUDA
#define EIGEN_NO_CUDA
#endif

#include <limits>
#include <thrust/device_vector.h>

#include "Frame.h"

constexpr float TRUNCATION = 0.06f;

namespace cuda
{
    class Volume
    {
    private:
        dim3 m_dimensions;
        float3 m_dd;
        float3 m_ddd;
        float3 m_min;
        float3 m_max;
        uint m_dim;

        // GPU Arrays
        thrust::device_vector<float> dev_tsdf;
        thrust::device_vector<float> dev_weights;
        thrust::device_vector<uchar4> dev_color;

    public:
        Volume(float3 min, float3 max, int x_dim, int y_dim, int z_dim, uint dim);

        float *GetTSDFRawPointer() const;
        float *GetWeightRawPointer() const;
        uchar4 *GetColorRawPointer() const;

        void ZeroOutMemory();

        void integrate(const Frame &frame);
        Frame extract();
    };
};