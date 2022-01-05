#pragma once

#ifndef EIGEN_NO_CUDA
#define EIGEN_NO_CUDA
#endif

#include "Eigen.h"
#include <thrust/device_vector.h>

typedef unsigned char BYTE;

namespace cuda
{
    class Frame
    {
    private:
        int2 mDimensions;
        Eigen::Matrix3f mIntrinsics;
        Eigen::Matrix4f mExtrinsics;

        // GPU Arrays
        thrust::device_vector<float> dev_depthMap;
        thrust::device_vector<uchar4> dev_colorMap;

    public:
        Frame();
        Frame(const float *depthMap, const BYTE *colorMap);

        // TODO
    };
};