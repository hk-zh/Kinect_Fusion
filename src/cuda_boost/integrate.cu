#include <cmath>
#include "cuda.h"
#include "cuda_runtime.h"
#include "Volume.cuh"
#include "Eigen.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#define TRUNCATION 0.06f
__global__ void integrate_cuda(const float* depthMap, const BYTE* colorMap,
                               int width, int height, uint dx, uint dy, uint dz,
                               Matrix4f extrinsicMatrix, Matrix3f intrinsicMatrix, Vector3f* mNormalsGlobal,
                               Voxel* vol, Vector3f min, Vector3f max){
    Vector3f Pg, Pc, ray, normal;
    Vector2i Pi;
    Vector4uc color;
    float depth, lambda, sdf, tsdf, tsdf_weight, value, weight, cos_angle;
    uint index;


    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dx || y >= dy)
        return;
    for (int z = 0; z < dz; z++) {

        // project the grid point into image space
        Pg  = Vector3f(
                min.x() + (max.x() - min.x()) * (float(x) * (1.0f / (dx - 1))),
                min.y() + (max.y() - min.y()) * (float(y) * (1.0f / (dy - 1))),
                min.z() + (max.z() - min.z()) * (float(z) * (1.0f / (dz - 1))));


        //Pc = frame.projectPointIntoFrame(Pg);
        const auto rotation = extrinsicMatrix.block(0, 0, 3, 3);
        const auto translation = extrinsicMatrix.block(0, 3, 3, 1);
        Pc =  rotation * Pg + translation;
        //Pi = frame.projectOntoImgPlane(Pc);
        Eigen::Vector3f projected = intrinsicMatrix * Pc;
        if (projected.z() == 0) {
            Pi =  Eigen::Vector2i(-INFINITY, -INFINITY);
        }
        else{
            projected /= projected.z();
            Pi = Eigen::Vector2i((int)round(projected.x()), (int)round(projected.y()));
        }

        //std::cout << Pg << std::endl << Pc << std::endl << Pi << std::endl;

        //Pg = gridToWorld(i, j, k);
        //Pc = Frame::transformPoint(Pg, worldToCamera);
        //Pi = Frame::perspectiveProjection(Pc, intrinsic);

        //std::cout << Pg << std::endl << Pc << std::endl << Pi << std::endl;

        //if (frame.containsImgPoint(Pi)) {
        if(Pi.x() >= 0 && Pi.x() < width && Pi.y() >= 0 &&
           Pi.y() < height){
            // get the depth of the point
            index = Pi.y() * width + Pi.x();
            depth = depthMap[index];

            if (depth == -INFINITY)
                continue;

            //std::cout << "Odbok!!\n";

            // calculate the sdf value
            lambda = (Pc / Pc.z()).norm();

            sdf = depth - ((Pg - translation) / lambda).norm();

            // compute the weight as the angle between the ray from the voxel point and normal of the associated frame point devided by depth
            ray = (Pg - translation).normalized();
            //normal = frame.getNormalGlobal(index);
            normal = *(mNormalsGlobal+index);

            cos_angle = - ray.dot(normal) / ray.norm() / normal.norm();

            tsdf_weight = 1.0f; //-cos_angle / depth; // 1; // 1 / depth;

            // get the previous value and weight

            uint prevIdx = x * dy * dz + y * dz + z;

            value = vol[prevIdx].getValue();
            weight = vol[prevIdx].getWeight();
            color = vol[prevIdx].getColor();

            // if we are doing the integration for the first time
            if (value == INFINITY) {
                value = 0;
                weight = 0;
                color = Vector4uc{ 0, 0, 0, 0 };
            }

            // truncation of the sdf
            if (sdf > 0) {
                tsdf = 1.0f < sdf / TRUNCATION ? 1.0f : sdf / TRUNCATION;
            }
            else {
                tsdf = -1.0f > sdf / TRUNCATION ? -1.0f : sdf/ TRUNCATION;
            }
            if (tsdf < -0.7f) {
                continue;
            }

            // the new value and weight is the running average
            vol[prevIdx].setValue((value * weight + tsdf * tsdf_weight) / (weight + tsdf_weight));
            vol[prevIdx].setWeight(weight + tsdf_weight);

            if (sdf <= TRUNCATION / 2 && sdf>= - TRUNCATION / 2 && colorMap[4 * index +3] == 255) {
                vol[prevIdx].setColor(
                        Vector4uc{
                                (const unsigned char)(((float)color[0] * weight + (float)colorMap[4 * index + 0] * tsdf_weight) / (weight + tsdf_weight)),
                                (const unsigned char)(((float)color[1] * weight + (float)colorMap[4 * index + 1] * tsdf_weight) / (weight + tsdf_weight)),
                                (const unsigned char)(((float)color[2] * weight + (float)colorMap[4 * index + 2] * tsdf_weight) / (weight + tsdf_weight)),
                                (const unsigned char)(((float)color[3] * weight + (float)colorMap[4 * index + 3] * tsdf_weight) / (weight + tsdf_weight))
                        }
                );
            }

            //std::cout << vol[getPosFromTuple(i, j, k)].getValue() << std::endl;
        }

    }
}


extern "C" void start(Frame& frame, Volume& volume){
    const Matrix4f worldToCamera = frame.getExtrinsicMatrix();
    const Matrix4f cameraToWorld = worldToCamera.inverse();
    const Matrix3f intrinsicMatrix = frame.getIntrinsicMatrix();
    Vector3f translation = cameraToWorld.block(0, 3, 3, 1);
    const float* depthMap = frame.getDepthMap();
    const BYTE* colorMap = frame.getColorMap();
    int width = frame.getFrameWidth();
    int height = frame.getFrameHeight();

    uint dx = volume.getDimX();
    uint dy = volume.getDimY();
    uint dz = volume.getDimZ();

    Vector3f min = volume.getmin();
    Vector3f max = volume.getmax();


    //std::cout << intrinsic << std::endl;

    // subscripts: g - global coordinate system | c - camera coordinate system | i - image space
    // short notations: V - vector | P - point | sdf - signed distance field value | tsdf - truncated sdf




    //TODO copy all the params into cuda memory

    float *dDepthMap, *dValues, *dWeights;
    Vector3f *dmNormalsGlobal;
    Vector4uc *dColors;
    BYTE* dColorMap;
    Voxel* vol_cuda;



    std::vector<Eigen::Vector3f> mNormalsGlobal = frame.getNormalMapGlobal();

    uint size = mNormalsGlobal.size();

    uint colormap_size = 4*640*480;

    thrust::host_vector<Vector3f> mNormalsGlobal_host = mNormalsGlobal;

    //thrust::device_vector<Eigen::Vector3f> mNormalsGlobal_device = mNormalsGlobal_host;
    Vector3f* mNormalsGlobal_cuda = thrust::raw_pointer_cast(mNormalsGlobal_host.data());

    cudaMalloc(&dDepthMap, width * height * sizeof(float));
    cudaMalloc(&dColorMap, colormap_size * sizeof(BYTE));
    cudaMalloc(&dValues, dx * dy * dz * sizeof(float));
    cudaMalloc(&dWeights, dx * dy * dz * sizeof(float));
    cudaMalloc(&dColors, dx * dy * dz * sizeof(Vector4uc));
    cudaMalloc(&dmNormalsGlobal, size * sizeof(Vector3f));
    cudaMalloc(&vol_cuda, dx * dy * dz * sizeof(Voxel));

    // copy data to device
    cudaMemcpy(
            dDepthMap, depthMap, width * height * sizeof(float),
            cudaMemcpyHostToDevice);
    cudaMemcpy(
            dColorMap, colorMap, colormap_size * sizeof(BYTE),
            cudaMemcpyHostToDevice);
    cudaMemcpy(
            vol_cuda, volume.get_vol(), dx * dy * dz * sizeof(Voxel),
            cudaMemcpyHostToDevice);
    cudaMemcpy(
            dmNormalsGlobal, mNormalsGlobal_cuda, size * sizeof(Vector3f),
            cudaMemcpyHostToDevice);



    std::cout << "Integrate starting..." << std::endl;


    const dim3 threads(32, 32);
    const dim3 blocks(
            (dx + threads.x - 1) / threads.x,
            (dy + threads.y - 1) / threads.y);

    integrate_cuda<<<blocks,threads>>>(dDepthMap, dColorMap,
                                       width, height, dx, dy, dz,
                                       worldToCamera, intrinsicMatrix, dmNormalsGlobal,
                                       vol_cuda, min, max);

    cudaDeviceSynchronize();





    auto err = cudaGetErrorString(cudaMemcpy(
            volume.get_vol(), vol_cuda, dx * dy * dz * sizeof(Voxel),
            cudaMemcpyDeviceToHost));
    std::cout<<err<<std::endl;




    cudaFree(dDepthMap);
    cudaFree(vol_cuda);
    cudaFree(dmNormalsGlobal);
    cudaFree(dColorMap);
    std::cout << "Integrate done!" << std::endl;
}