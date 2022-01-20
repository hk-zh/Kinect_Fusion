#include <cmath>
#include "cuda.h"
#include "cuda_runtime.h"
#include "Volume.cuh"
//#include "RayCaster.cuh"
#include "Eigen.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <cmath>
#include <thrust/sequence.h>
#include <fstream>
#include <iostream>

class Ray_cuda{
private:
    Vector3f start;
    Vector3f direction;
    Vector3f currentPoint;
    Vector3f previousPoint;

public:
    __device__ Ray_cuda(){}
    __device__ Ray_cuda(Vector3f& start_, Vector3f& direction_) : start(start_), direction(direction_) {
        currentPoint = start_;
        direction.normalize();
    }

    __device__ Vector3f& next(){
        previousPoint = currentPoint;
        currentPoint += direction;
        return currentPoint;
    }
};

__device__ inline Vector3f worldToGrid(Vector3f& p, Vector3f& min, Vector3f& max, float ddx, float ddy, float ddz) {
    Vector3f coord(0.0f, 0.0f, 0.0f);

    coord[0] = (p[0] - min[0]) / (max[0] - min[0]) / ddx;
    coord[1] = (p[1] - min[1]) / (max[1] - min[1]) / ddy;
    coord[2] = (p[2] - min[2]) / (max[2] - min[2]) / ddz;

    return coord;
}

__device__ inline Vector3f gridToWorld(Vector3f& p, Vector3f& min, Vector3f& max, float ddx, float ddy, float ddz)
{
    Vector3f coord(0.0f, 0.0f, 0.0f);

    coord[0] = min[0] + (max[0] - min[0]) * (p[0] * ddx);
    coord[1] = min[1] + (max[1] - min[1]) * (p[1] * ddy);
    coord[2] = min[2] + (max[2] - min[2]) * (p[2] * ddz);

    return coord;
}

__device__ 	bool isInterpolationPossible(Vector3f& point, uint dx, uint dy, uint dz) {
    return!(
                    point[0] > (float)dx - 3 ||
                    point[1] > (float)dy - 3 ||
                    point[2] > (float)dz - 3 ||
                    point[0] < 2 ||
                    point[1] < 2 ||
                    point[2] < 2
            );
}

__device__ 	inline Vector3i intCoords(const Vector3f& p) {
    Vector3i coord{ 0, 0, 0 };

    coord[0] = int(p[0]);
    coord[1] = int(p[1]);
    coord[2] = int(p[2]);

    return coord;
}

__device__ 	bool isPointInVolume(Vector3f& point, uint dx, uint dy, uint dz){
    return
            !(
                    point[0] > (float)dx - 1 ||
                    point[1] > (float)dy - 1 ||
                    point[2] > (float)dz - 1 ||
                    point[0] < 0 ||
                    point[1] < 0 ||
                    point[2] < 0
            );
}

__device__ float trilinearInterpolation(const Vector3f& p, Voxel* voxels, uint dx, uint dy, uint dz){
    Vector3i start = intCoords(p);
    float c000, c001, c010, c011, c100, c101, c110, c111;

    c000 = voxels[(start[0] + 0)*dy*dz + ( start[1] + 0)*dz + ( start[2] + 0)].getValue();
    c100 = voxels[(start[0] + 1)*dy*dz + ( start[1] + 0)*dz + ( start[2] + 0)].getValue();
    c001 = voxels[(start[0] + 0)*dy*dz + ( start[1] + 0)*dz + ( start[2] + 1)].getValue();
    c101 = voxels[(start[0] + 1)*dy*dz + ( start[1] + 0)*dz + ( start[2] + 1)].getValue();
    c010 = voxels[(start[0] + 0)*dy*dz + ( start[1] + 1)*dz + ( start[2] + 0)].getValue();
    c110 = voxels[(start[0] + 1)*dy*dz + ( start[1] + 1)*dz + ( start[2] + 0)].getValue();
    c011 = voxels[(start[0] + 0)*dy*dz + ( start[1] + 1)*dz + ( start[2] + 1)].getValue();
    c111 = voxels[(start[0] + 1)*dy*dz + ( start[1] + 1)*dz + ( start[2] + 1)].getValue();

    if (
            c000 == INFINITY ||
            c001 == INFINITY ||
            c010 == INFINITY ||
            c011 == INFINITY ||
            c100 == INFINITY ||
            c101 == INFINITY ||
            c110 == INFINITY ||
            c111 == INFINITY
            )
        return INFINITY;

    float xd, yd, zd;

    xd = p[0] - (float)start[0]; //(p[0] - start[0]) / (start[0] + 1 - start[0]);
    yd = p[1] - (float)start[1]; //(p[1] - start[1]) / (start[1] + 1 - start[1]);
    zd = p[2] - (float)start[2]; //(p[1] - start[2]) / (start[2] + 1 - start[2]);

    float c00, c01, c10, c11;

    c00 = c000 * (1 - xd) + c100 * xd;
    c01 = c001 * (1 - xd) + c101 * xd;
    c10 = c010 * (1 - xd) + c110 * xd;
    c11 = c011 * (1 - xd) + c111 * xd;

    float c0, c1;

    c0 = c00 * (1 - yd) + c10 * yd;
    c1 = c01 * (1 - yd) + c11 * yd;

    float c;

    c = c0 * (1 - zd) + c1 * zd;

    return c;
}

__device__ 	void setVisitedSingle(int x, int y, int z, uint dx, uint dy, uint dz, bool* visitedVoxels) {
    unsigned int index = x * dy * dz + y * dz + z;
    if (index < dx * dy * dz) {
        visitedVoxels[index] = true;
    }

}

__device__ void set_visited(Vector3i& voxCoords, uint dx, uint dy, uint dz, bool* visitedVoxels){

    Vector3i starting_points[8];
    Vector3i p_int = voxCoords;

    starting_points[0] = (Vector3i{ p_int[0] + 0, p_int[1] + 0, p_int[2] + 0 });
    starting_points[0] = (Vector3i{ p_int[0] - 1, p_int[1] + 0, p_int[2] + 0 });
    starting_points[0] = (Vector3i{ p_int[0] + 0, p_int[1] - 1, p_int[2] + 0 });
    starting_points[0] = (Vector3i{ p_int[0] - 1, p_int[1] - 1, p_int[2] + 0 });
    starting_points[0] = (Vector3i{ p_int[0] + 0, p_int[1] + 0, p_int[2] - 1 });
    starting_points[0] = (Vector3i{ p_int[0] - 1, p_int[1] + 0, p_int[2] - 1 });
    starting_points[0] = (Vector3i{ p_int[0] + 0, p_int[1] - 1, p_int[2] - 1 });
    starting_points[0] = (Vector3i{ p_int[0] - 1, p_int[1] - 1, p_int[2] - 1 });

    for (Matrix<int, 3, 1> p_int : starting_points) {
        setVisitedSingle( p_int[0] + 0, p_int[1] + 0, p_int[2] + 0, dx, dy, dz, visitedVoxels);
        setVisitedSingle( p_int[0] + 1, p_int[1] + 0, p_int[2] + 0, dx, dy, dz, visitedVoxels);
        setVisitedSingle( p_int[0] + 0, p_int[1] + 1, p_int[2] + 0, dx, dy, dz, visitedVoxels);
        setVisitedSingle( p_int[0] + 1, p_int[1] + 1, p_int[2] + 0, dx, dy, dz, visitedVoxels);
        setVisitedSingle( p_int[0] + 0, p_int[1] + 0, p_int[2] + 1, dx, dy, dz, visitedVoxels);
        setVisitedSingle( p_int[0] + 1, p_int[1] + 0, p_int[2] + 1, dx, dy, dz, visitedVoxels);
        setVisitedSingle( p_int[0] + 0, p_int[1] + 1, p_int[2] + 1, dx, dy, dz, visitedVoxels);
        setVisitedSingle( p_int[0] + 1, p_int[1] + 1, p_int[2] + 1, dx, dy, dz, visitedVoxels);

    }
}
__device__ 	bool voxelVisited(int x, int y, int z, uint dx, uint dy, uint dz, bool* visitedVoxels) {
    Vector3i pos = Vector3i{ x, y, z };
    unsigned int index = x * dy * dz + y * dz + z;
    if (index >= dx * dy * dz) {
        return false;
    } else {
        return visitedVoxels[index];
    }
}


__device__ 	bool voxelVisited(Vector3f& p, uint dx, uint dy, uint dz, bool* visitedVoxels) {
    Vector3i pi = intCoords(p);
    return voxelVisited(pi[0], pi[1], pi[2], dx, dy, dz, visitedVoxels);
}

__global__ void raycast_parallel(int width, int height, uint dx, uint dy, uint dz,
                                 Matrix3f intrinsic_inverse,
                                 Vector3f min, Vector3f max, Voxel* voxels,
                                 float ddx, float ddy, float ddz, Vector3f translation, Matrix3f rotationMatrix,
                                 Vector3f* output_vertices_global_cuda, Vector4uc* output_colors_global_cuda, bool* vistedVoxels){


    // Calculate the column index of the Pd element, denote by x
    uint j = blockIdx.y * blockDim.y + threadIdx.y;
    // Calculate the row index of the Pd element, denote by y
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    uint index = i * width + j;

    if (i >= height || j >= width)
        return;


    //std::cout << i << " " << j << std::endl;

    // starting point is the position of the camera (translation) in grid coordinates
    Vector3f ray_start, ray_dir, ray_current, ray_previous, ray_next;
    Vector3i ray_current_int, ray_previous_int;

    float sdf_1, sdf_2;
    Vector3f p, v, n;



    ray_start = worldToGrid(translation, min, max, ddx, ddy, ddz);

    // calculate the direction vector as vector from camera position to the pixel(i, j)s world coordinates
    //index = i * width + j;

    ray_next = Vector3f{ float(j), float(i), 1.0f };
    ray_next = intrinsic_inverse * ray_next;
    ray_next = rotationMatrix * ray_next + translation;
    ray_next = worldToGrid(ray_next, min, max, ddx, ddy, ddz);

    ray_dir = ray_next - ray_start;
    ray_dir = ray_dir.normalized();

    if (!ray_dir.derived().array().isFinite().all() || ray_dir == Vector3f{ 0.0f, 0.0f, 0.0f }) {
        output_vertices_global_cuda[index] = Vector3f(-INFINITY, -INFINITY, -INFINITY);
        output_colors_global_cuda[index]= Vector4uc(0, 0, 0, 0);
        return;
    }

    Ray_cuda ray = Ray_cuda(ray_start, ray_dir);

    ray_current = ray_start;
    // forward until the ray in range
    int cnt = 0;
    while (!isInterpolationPossible(ray_current, dx, dy, dz) && cnt++ < 1000) {
        ray_current = ray.next();
    }
    ray_current_int = intCoords(ray_current);

    if (!isPointInVolume(ray_current, dx, dy, dz)) {
        output_vertices_global_cuda[index] = Vector3f(-INFINITY, -INFINITY, -INFINITY);
        output_colors_global_cuda[index]= Vector4uc(0, 0, 0, 0);
        return;
    }

    while (true) {//vol.isPointInVolume(ray_current)) {

        ray_previous = ray_current;
        ray_previous_int = ray_current_int;

        // until reach the next grid
        do {
            //std::cout << ray_current << std::endl;
            ray_current = ray.next();
            ray_current_int = intCoords(ray_current);
        } while (ray_previous_int == ray_current_int);


        if (!isInterpolationPossible(ray_previous, dx, dy, dz) || !isInterpolationPossible(ray_current, dx, dy, dz)) {
            output_vertices_global_cuda[index] = Vector3f(-INFINITY, -INFINITY, -INFINITY);
            output_colors_global_cuda[index]= Vector4uc(0, 0, 0, 0);
            break;
        } else if (voxels[ray_previous_int.x()*dy*dz + ray_previous_int.y()*dz + ray_previous_int.z()].getValue() == 0 ) {
            v = gridToWorld(ray_previous, min, max, ddx, ddy, ddz);
            output_vertices_global_cuda[index] = v;
            output_colors_global_cuda[index]=voxels[ray_previous_int.x()*dy*dz + ray_previous_int.y()*dz + ray_previous_int.z()].getColor();
            //vistedVoxels[]
            if (!voxelVisited(ray_previous, dx, dy, dz, vistedVoxels)) {
                set_visited(ray_previous_int, dx, dy, dz, vistedVoxels);
            }

            break;
        } else if (voxels[ray_current_int.x()*dy*dz + ray_current_int.y()*dz + ray_current_int.z()].getValue() == 0) {
            v = gridToWorld(ray_current, min, max, ddx, ddy, ddz);
            output_vertices_global_cuda[index] = v;
            output_colors_global_cuda[index] = voxels[ray_current_int.x()*dy*dz + ray_current_int.y()*dz + ray_current_int.z()].getColor();

            if (!voxelVisited(ray_current, dx, dy, dz, vistedVoxels)) {
                set_visited(ray_current_int, dx, dy, dz, vistedVoxels);
            }

            break;
        } else if (
                voxels[ray_previous_int.x()*dy*dz + ray_previous_int.y()*dz + ray_previous_int.z()].getValue() != INFINITY &&
                voxels[ray_previous_int.x()*dy*dz + ray_previous_int.y()*dz + ray_previous_int.z()].getValue() > 0 &&
                voxels[ray_current_int.x()*dy*dz + ray_current_int.y()*dz + ray_current_int.z()].getValue() != INFINITY &&
                voxels[ray_current_int.x()*dy*dz + ray_current_int.y()*dz + ray_current_int.z()].getValue() < 0
                ) {
            sdf_1 = trilinearInterpolation(ray_previous, voxels, dx, dy, dz);
            sdf_2 = trilinearInterpolation(ray_current, voxels, dx, dy, dz);

            if (sdf_1 == INFINITY || sdf_2 == INFINITY || sdf_2 == sdf_1) {
                output_vertices_global_cuda[index] = Vector3f(-INFINITY, -INFINITY, -INFINITY);
                output_colors_global_cuda[index]= Vector4uc(0, 0, 0, 0);
                break;
            }

            p = ray_previous - (ray_dir * sdf_1) / (sdf_2 - sdf_1);

            if (!isInterpolationPossible(p, dx, dy, dz)) {
                output_vertices_global_cuda[index] = Vector3f(-INFINITY, -INFINITY, -INFINITY);
                output_colors_global_cuda[index]= Vector4uc(0, 0, 0, 0);
                break;
            }
            if (voxels[ray_previous_int.x()*dy*dz + ray_previous_int.y()*dz + ray_previous_int.z()].isValidColor()
                && voxels[ray_current_int.x()*dy*dz + ray_current_int.y()*dz + ray_current_int.z()].isValidColor()) {
                Vector4uc color1 = voxels[ray_previous_int.x()*dy*dz + ray_previous_int.y()*dz + ray_previous_int.z()].getColor();
                Vector4uc color2 = voxels[ray_current_int.x()*dy*dz + ray_current_int.y()*dz + ray_current_int.z()].getColor();
                Vector4uc color = Vector4uc{
                        (const unsigned char) (
                                ((float) color1[0] * abs(sdf_2) + (float) color2[0] * abs(sdf_1)) /
                                (abs(sdf_1) + abs(sdf_2))),
                        (const unsigned char) (
                                ((float) color1[1] * abs(sdf_2) + (float) color2[1] * abs(sdf_1)) /
                                (abs(sdf_1) + abs(sdf_2))),
                        (const unsigned char) (
                                ((float) color1[2] * abs(sdf_2) + (float) color2[2] * abs(sdf_1)) /
                                (abs(sdf_1) + abs(sdf_2))),
                        (const unsigned char) (
                                ((float) color1[3] * abs(sdf_2) + (float) color2[3] * abs(sdf_1)) /
                                (abs(sdf_1) + abs(sdf_2))),
                };
                output_colors_global_cuda[index] = (color);
            } else if (voxels[ray_previous_int.x()*dy*dz + ray_previous_int.y()*dz + ray_previous_int.z()].isValidColor()) {
                Vector4uc color1 = voxels[ray_previous_int.x()*dy*dz + ray_previous_int.y()*dz + ray_previous_int.z()].getColor();
                output_colors_global_cuda[index] = (color1);
            } else if (voxels[ray_current_int.x()*dy*dz + ray_current_int.y()*dz + ray_current_int.z()].isValidColor()) {
                Vector4uc color2 = voxels[ray_current_int.x()*dy*dz + ray_current_int.y()*dz + ray_current_int.z()].getColor();
                output_colors_global_cuda[index] = (color2);
            } else {
                output_colors_global_cuda[index] = (Vector4uc {0,0,0,0});
            }
            //std::cout << ray_previous << std::endl << ray_current << std::endl << ray_dir << std::endl << sdf_1 << " " << sdf_2 << std::endl << p << std::endl;
            v = gridToWorld(p, min, max, ddx, ddy, ddz);
            output_vertices_global_cuda[index] = v;

            if (!voxelVisited(ray_previous, dx, dy, dz, vistedVoxels)) {
                set_visited(ray_previous_int, dx, dy, dz, vistedVoxels);
            }
//
            if (!voxelVisited(ray_current, dx, dy, dz, vistedVoxels)) {
                set_visited(ray_current_int, dx, dy, dz, vistedVoxels);
            }
            break;
        }
    }
}


extern "C" void start_raycast(Frame& frame, Volume& volume){
    const Matrix4f worldToCamera = frame.getExtrinsicMatrix();
    const Matrix4f cameraToWorld = worldToCamera.inverse();
    const Matrix3f intrinsic_inverse = frame.getIntrinsicMatrix().inverse();
    Vector3f translation = cameraToWorld.block(0, 3, 3, 1);
    Matrix3f rotationMatrix = cameraToWorld.block(0, 0, 3, 3);
    Vector4uc color;
    int width = frame.getFrameWidth();
    int height = frame.getFrameHeight();


    uint dx = volume.getDimX();
    uint dy = volume.getDimY();
    uint dz = volume.getDimZ();

    Vector3f min = volume.getmin();
    Vector3f max = volume.getmax();


    Vector3f *output_vertices_global_cuda;
    Vector4uc *output_colors_global_cuda;
    bool* visitedVoxels;



    std::vector<Eigen::Vector3f> mNormalsGlobal = frame.getNormalMapGlobal();



    thrust::host_vector<Vector3f> mNormalsGlobal_host = mNormalsGlobal;



    Voxel* vol_cuda;


    cudaMalloc(&visitedVoxels, dx*dy*dz*sizeof(bool));
    cudaMalloc(&output_vertices_global_cuda, width*height* sizeof(Vector3f));
    cudaMalloc(&output_colors_global_cuda, width*height* sizeof(Vector4uc));
    cudaMalloc(&vol_cuda, dx * dy * dz * sizeof(Voxel));


    // copy data to device

    cudaMemcpy(
            vol_cuda, volume.get_vol(), dx * dy * dz * sizeof(Voxel),
            cudaMemcpyHostToDevice);
    cudaMemcpy(
            visitedVoxels, volume.getVisitedVoxels(), dx * dy * dz * sizeof(bool),
            cudaMemcpyHostToDevice);




    std::cout << "RayCast starting..." << std::endl;
    std::cout << "Height: "<<height<<" Width: "<<width << std::endl;

    const dim3 threads(32, 32);
    const dim3 blocks(
            (height + threads.x - 1) / threads.x,
            (width + threads.y - 1) / threads.y);

    raycast_parallel<<<blocks,threads>>>(width, height, dx, dy, dz,
                                       intrinsic_inverse,
                                       min, max, vol_cuda,
                                       volume.getddx(), volume.getddy(), volume.getddz(),
                                       translation, rotationMatrix, output_vertices_global_cuda, output_colors_global_cuda, visitedVoxels);

    Vector3f* output_vertices_global_raw = (Vector3f*)std::malloc(width*height* sizeof(Vector3f));
    Vector4uc* output_colors_global_raw = (Vector4uc*)std::malloc(width*height* sizeof(Vector4uc));
    //bool* visitedVoxels_host = (bool*)std::malloc(dx * dy * dz * sizeof(bool));

    auto err = cudaGetErrorString(cudaMemcpy(
            output_vertices_global_raw, output_vertices_global_cuda, width*height* sizeof(Vector3f),
            cudaMemcpyDeviceToHost));
    std::cout<<err<<std::endl;
    cudaMemcpy(
            volume.getVisitedVoxels(), visitedVoxels, dx * dy * dz * sizeof(bool),
            cudaMemcpyDeviceToHost);








    cudaMemcpy(
            output_colors_global_raw, output_colors_global_cuda, width*height* sizeof(Vector4uc),
            cudaMemcpyDeviceToHost);
    std::shared_ptr<std::vector<Vector3f>> output_vertices_global = std::make_shared<std::vector<Vector3f>>(
            std::vector<Vector3f>(output_vertices_global_raw, output_vertices_global_raw+width*height
            ));
    std::shared_ptr<std::vector<Vector4uc>> output_colors_global = std::make_shared<std::vector<Vector4uc>>(
            std::vector<Vector4uc>(output_colors_global_raw, output_colors_global_raw+width*height
            ));


    std::cout << "output_vertices_global: " << output_vertices_global->size() << std::endl;
    frame.mVerticesGlobal = output_vertices_global;
    //frame.mNormalsGlobal = output_normals_global;
    frame.mVertices = std::make_shared<std::vector<Vector3f>>(frame.transformPoints(*output_vertices_global, worldToCamera));

    frame.computeNormalMap(width, height);
    frame.mNormalsGlobal = std::make_shared<std::vector<Vector3f>>(frame.rotatePoints(frame.getNormalMap(), rotationMatrix));


    //std::ofstream myfile;
    //myfile.open ("./globalvertex_cuda.txt");
    std::cout<<output_colors_global->size()<<std::endl;
    for (int i = 0; i < output_colors_global->size(); i++) {
        frame.colorMap[4*i] =(*output_colors_global)[i][0];
        frame.colorMap[4*i+1] =(*output_colors_global)[i][1];
        frame.colorMap[4*i+2] =(*output_colors_global)[i][2];
        frame.colorMap[4*i+3] =(*output_colors_global)[i][3];

        //myfile << (*output_vertices_global)[i]<< std::endl;
    }
    //myfile.close();
    std::cout << "RayCast done!" << std::endl;


    cudaFree(vol_cuda);
    cudaFree(output_vertices_global_cuda);
    cudaFree(output_colors_global_cuda);
    cudaFree(visitedVoxels);

}