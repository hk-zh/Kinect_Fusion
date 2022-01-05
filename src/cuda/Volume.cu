#include "Volume.h"

#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <cuda/std/cfloat>

#include "cudaMath.h"

inline __device__ float3 rotateVector(const float3 &vector, const float *rotMat)
{
    float x = rotMat[0] * vector.x + rotMat[1] + vector.y + rotMat[2] + vector.z;
    float y = rotMat[3] * vector.x + rotMat[4] + vector.y + rotMat[5] + vector.z;
    float z = rotMat[6] * vector.x + rotMat[7] + vector.y + rotMat[8] + vector.z;

    return make_float3(x, y, z);
}

inline __device__ float3 translateVector(const float3 &vector, const float3 &translation)
{
    return make_float3(vector.x + translation.x, vector.y + translation.y, vector.z + translation.z);
}

inline __device__ float3 transformVector(const float3 &point, const float *matrix4f)
{
    float x = matrix4f[0] * point.x + matrix4f[3] * point.y + matrix4f[6] * point.z + matrix4f[9];
    float y = matrix4f[1] * point.x + matrix4f[4] * point.y + matrix4f[7] * point.z + matrix4f[10];
    float z = matrix4f[2] * point.x + matrix4f[5] * point.y + matrix4f[8] * point.z + matrix4f[11];

    return make_float3(x, y, z);
}

inline __device__ float3 gridToWorld(const int3 &point, const float3 &min, const float3 &max, const float3 &dd)
{
    float3 coord;
    coord.x = min.x + (max.x - min.x) * (point.x * dd.x);
    coord.y = min.y + (max.y - min.y) * (point.y * dd.y);
    coord.z = min.z + (max.z - min.z) * (point.z * dd.z);

    return coord;
}

struct ApplyTransform
{
    const float *mTransformation;

    ApplyTransform(const float *transformation) : mTransformation(transformation)
    {
    }

    __device__ __host__
        float3
        operator()(float3 point)
    {
        float x = mTransformation[0] * point.x + mTransformation[3] * point.y + mTransformation[6] * point.z + mTransformation[9];
        float y = mTransformation[1] * point.x + mTransformation[4] * point.y + mTransformation[7] * point.z + mTransformation[10];
        float z = mTransformation[2] * point.x + mTransformation[5] * point.y + mTransformation[8] * point.z + mTransformation[11];

        return make_float3(x, y, z);
    }
};

__device__ bool
FrameContainsImagePoint(int2 point, int2 frameDim)
{
    return point.x <= frameDim.x && point.y <= frameDim.y;
}

__global__ void
ZeroOutMemoryKernel(float *tsdf, float *weights, uint4 *color, int N)
{
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (N < idx)
    {
        tsdf[idx] = FLT_MAX;
        weights[idx] = 0.0f;
        color[idx] = make_uint4(0, 0, 0, 0);
    }
}

__global__ void
IntegrationKernel(float *depthMap,
                  float3 *normalMap,
                  uchar4 *colorMap,
                  float *worldToCamera,
                  float *cameraToWorld,
                  float *intrinsics,
                  int2 frameDimensions,
                  float *tsdfValues,
                  float *tsdfWeights,
                  float3 min,
                  float3 max,
                  float3 volDimension)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    const int j = blockDim.y * blockIdx.y + threadIdx.y;
    const int k = blockDim.z * blockDim.z + threadIdx.z;

    const int globalIdx = (i * volDimension.x) * (j * volDimension.y) + k;

    // TODO: Check memory boundaries

    float3 Pg = gridToWorld(make_int3(i, j, k), min, max, volDimension);
    float3 Pc = transformVector(Pg, worldToCamera);
    float3 Pi_ = rotateVector(Pg, intrinsics);
    int2 Pi = make_int2(Pi_.x, Pi_.y);

    if (FrameContainsImagePoint(Pi, frameDimensions))
    {
        int index = Pi.y * frameDimensions.y + Pi.x;
        float depth = depthMap[index];

        if (depth == MINF)
        {
            return;
        }

        float lambda = length(Pc / Pc.z);
        float3 translation = make_float3(cameraToWorld[9], cameraToWorld[10], cameraToWorld[11]);
        float sdf = length(depth - ((Pg - translation) / lambda));
        float3 ray = normalize(Pg - translation);
        float3 normal = normalMap[index];

        float cos_angle = dot(ray, normal) / length(ray) / length(normal);
        float tsdf_weight = 1; // TODO

        float tsdfValue = tsdfValues[globalIdx];
        float weight = tsdfWeights[globalIdx];
        uchar4 color = colorMap[globalIdx];

        if (tsdfValue == FLT_MAX)
        {
            tsdfValue = 0;
            weight = 0;
            color = make_uchar4(0, 0, 0, 0);
        }

        float tsdf;
        if (sdf > 0)
        {
            tsdf = fminf(1.0f, sdf / TRUNCATION);
        }
        else
        {
            tsdf = fmaxf(1.0f, sdf / TRUNCATION);
        }

        tsdfValues[globalIdx] = (tsdfValue * weight + tsdf * tsdf_weight) / (weight + tsdf_weight);
        tsdfWeights[globalIdx] = weight + tsdf_weight;

        // TODO update color
    }
}

cuda::Volume::Volume(float3 min, float3 max, int x_dim, int y_dim, int z_dim, uint dim) : m_min(min), m_max(max), m_dimensions(x_dim, y_dim, z_dim), m_dim(dim)
{
    auto numElements = x_dim * y_dim * z_dim;

    dev_tsdf.reserve(numElements);
    dev_weights.reserve(numElements);
    dev_color.reserve(numElements);
}

float *cuda::Volume::GetTSDFRawPointer() const
{
    return (float *)thrust::raw_pointer_cast(dev_tsdf.data());
}

float *cuda::Volume::GetWeightRawPointer() const
{
    return (float *)thrust::raw_pointer_cast(dev_weights.data());
}

uchar4 *cuda::Volume::GetColorRawPointer() const
{
    return (uchar4 *)thrust::raw_pointer_cast(dev_color.data());
}

void cuda::Volume::ZeroOutMemory()
{
    thrust::fill(dev_tsdf.begin(), dev_tsdf.end(), FLT_MAX);
    thrust::fill(dev_weights.begin(), dev_weights.end(), 0.0f);
    thrust::fill(dev_color.begin(), dev_color.end(), make_uchar4(0, 0, 0, 0));
}

void cuda::Volume::integrate(const cuda::Frame &frame)
{
    // TODO
}

cuda::Frame cuda::Volume::extract()
{
    return cuda::Frame();
}