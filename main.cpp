#include <array>
#include <fstream>
#include <iostream>

#include "ICP.h"
#include "RayCaster.h"
#include "Frame.h"
#include "Volume.h"
#include "VirtualSensor.h"
#include "Eigen.h"
#include "SimpleMesh.h"
#include "MarchingCubes.h"
#include "ICPOptimizer.h"

#define DISTANCE_THRESHOLD 0.05
#define EDGE_THRESHOLD 0.02
#define ANGLE_THRESHOLD 1.05
#define MAX_FRAME_NUM 2000
#define MIN_POINT -0.7f, -0.7f, -0.5f
#define MAX_POINT 0.7f, 0.7f, 1.0f
#define RESOLUTION 256, 256, 256
#define ICP_ITERATIONS 20
#define USE_ICP_FROM_CLASS false
#define SAMPLE_FREQUENCY 10



void sample(std::vector<Vector3f> &a1, std::vector<Vector3f> &b1, std::vector<Vector3f> &a2, std::vector<Vector3f> &b2,
            unsigned int num_pixels) {
    a1.clear();
    a2.clear();
    for (int i = 0; i < num_pixels; i = i + SAMPLE_FREQUENCY) {
        if (b1[i].allFinite() && b2[i].allFinite()) {
            a1.push_back(b1[i]);
            a2.push_back(b2[i]);
        }
    }
}


int main() {
    // Make sure this path points to the data folder
    std::string filenameIn = "../data/rgbd_dataset_freiburg1_plant/";
    std::string filenameBaseOut = std::string("../output_plant_128/mesh_");
    std::string filenameBaseOutMC = std::string("../output_plant_128/MCmesh_");

    // load video
    std::cout << "Initialize virtual sensor..." << std::endl;
    VirtualSensor sensor;
    if (!sensor.Init(filenameIn)) {
        std::cout << "Failed to initialize the sensor!" << std::endl;
        return -1;
    }

    int frameCount = 0;
    Frame curFrame, prevFrame, curFrameFiltered;
    Vector3f min_point = Vector3f{MIN_POINT};
    Vector3f max_point = Vector3f{MAX_POINT};

    Volume volume = Volume(min_point, max_point, RESOLUTION, 3);
    RayCaster rc = RayCaster(volume);
    Matrix4f identity = Matrix4f::Identity(4, 4);  // initial estimate
    Matrix4f pose_cur = identity;
    Matrix4f pose_prev = identity;
    unsigned int num_pixels = sensor.GetDepthImageWidth() * sensor.GetDepthImageHeight();
    std::vector<Vector3f> vertex_current = std::vector<Vector3f>(num_pixels);
    std::vector<Vector3f> normal_current = std::vector<Vector3f>(num_pixels);
    std::vector<Vector3f> vertex_prediction = std::vector<Vector3f>(num_pixels);
    std::vector<Vector3f> normal_prediction = std::vector<Vector3f>(num_pixels);

    auto *linearIcp = new LinearICPOptimizer();
    auto *ceresIcp = new CeresICPOptimizer();
    while (frameCount < MAX_FRAME_NUM && sensor.ProcessNextFrame()) {
        float *depthMap = sensor.GetDepth();
        float *depthMapFiltered = sensor.GetDepthFiltered();
        BYTE *colorMap = sensor.GetColorRGBX();
        Matrix3f depthIntrinsics = sensor.GetDepthIntrinsics();
        Matrix4f depthExtrinsics = sensor.GetDepthExtrinsics();
        Matrix4f trajectory = sensor.GetTrajectory();
        Matrix4f trajectoryInv = sensor.GetTrajectory().inverse();
        int depthHeight = (int) sensor.GetDepthImageHeight();
        int depthWidth = (int) sensor.GetDepthImageWidth();

        //std::cout << trajectory;

        curFrame =
                Frame(depthMap, colorMap, depthIntrinsics, depthExtrinsics,
                      trajectoryInv, depthWidth, depthHeight);

        curFrameFiltered =
                Frame(depthMapFiltered, colorMap, depthIntrinsics, depthExtrinsics,
                      trajectoryInv, depthWidth, depthHeight);


        if (frameCount == 0) {
            volume.integrate(curFrameFiltered);
            prevFrame = curFrame;
        } else {
            /* ==============  ICP: old version  ============== */
            bool succ = true;
            if (USE_ICP_FROM_CLASS) {
                sample(vertex_current, curFrame.getVertexMapGlobal(), normal_current, curFrame.getNormalMapGlobal(), num_pixels);
                sample(vertex_prediction, prevFrame.getVertexMapGlobal(), normal_prediction, prevFrame.getNormalMapGlobal(), num_pixels);
                pose_cur = pose_prev;
                succ = linearIcp -> estimatePose(
                        vertex_current, normal_current, vertex_prediction,normal_prediction, pose_cur
                );
            } else {
                ICP icp(prevFrame, curFrame, DISTANCE_THRESHOLD, ANGLE_THRESHOLD);
                pose_cur = pose_prev;
                succ = icp.estimatePose(pose_cur, ICP_ITERATIONS);
            }


            if (succ) {
                std::cout << pose_cur << std::endl;
                curFrame.setExtrinsicMatrix(pose_cur.inverse());
                curFrameFiltered.setExtrinsicMatrix(pose_cur.inverse());
                volume.integrate(curFrameFiltered);
                rc.changeFrame(curFrame);
                curFrame = rc.rayCast();
                prevFrame = curFrame;
                pose_prev = pose_cur;
            } else {
                std::cout << "ICP failed, use previous frame and pose!" << std::endl;
                std::cout << pose_prev << std::endl;
            }







            // output
            if (frameCount % 20 == 1) {
                std::stringstream ss;
                ss << filenameBaseOut << frameCount << ".off";

                if (!curFrame.writeMesh(ss.str(), EDGE_THRESHOLD)) {
                    std::cout << "Failed to write mesh!\nCheck file path!" << std::endl;
                    return -1;
                }
            }

            if (frameCount % 100 == 1) {
                std::stringstream ss;
                ss << filenameBaseOutMC << frameCount << ".off";

                std::cout << "Marching Cubes started..." << std::endl;
                SimpleMesh mesh;

                std::unordered_map<Vector3i, bool, matrix_hash<Vector3i>> visitedVoxels = volume.getVisitedVoxels();

                for (auto &visitedVoxel : visitedVoxels) {
                    Vector3i voxelCoords = visitedVoxel.first;
                    ProcessVolumeCell(&volume, voxelCoords[0], voxelCoords[1], voxelCoords[2], 0.00f, &mesh);
                }
                /*
                for (unsigned int x = 0; x < volume.getDimX() - 1; x++)
                {
                    //std::cerr << "Marching Cubes on slice " << x << " of " << volume.getDimX() << std::endl;

                    for (unsigned int y = 0; y < volume.getDimY() - 1; y++)
                    {
                        for (unsigned int z = 0; z < volume.getDimZ() - 1; z++)
                        {
                            ProcessVolumeCell(&volume, x, y, z, 0.00f, &mesh);
                        }
                    }
                }
                */
                std::cout << "Marching Cubes done! " << mesh.getVertices().size() << " " << mesh.getTriangles().size()
                          << std::endl;

                // write mesh to file
                if (!mesh.writeMesh(ss.str())) {
                    std::cout << "ERROR: unable to write output file!" << std::endl;
                    return -1;
                }
            }
        }


        frameCount++;
    }

    return 0;
}
