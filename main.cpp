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
#define MAX_FRAME_NUM 1000
#define MIN_POINT -0.7f, -0.5f, -0.5f
#define MAX_POINT 0.7f, 0.5f, 1.2f
#define RESOLUTION 512, 512, 512
#define SAMPLE_FREQUENCY 10
#define ICP_ITERATIONS 30
#define ICP_VERSION 1
#define OUTPUT_FILE "../output_plant_512/"
#define INPUT_FILE "../data/rgbd_dataset_freiburg1_plant/"
/*
 * We have two icp versions.
 * Version 1: the version from Kinect-Fusion paper
 * Version 2: the version using knn to find corresponding pairs
 */




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
    std::string filenameIn = INPUT_FILE;
    std::string filenameBaseOut = std::string(OUTPUT_FILE) + "mesh_";
    std::string filenameBaseOutMC = std::string(OUTPUT_FILE) + "MCmesh_";
    std::string filenameBaseOutNormal = std::string(OUTPUT_FILE) + "normal_";
    std::string filenameBaseOutColor = std::string(OUTPUT_FILE) + "color_";
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
    std::vector<Vector3f> vertex_previous = std::vector<Vector3f>(num_pixels);
    std::vector<Vector3f> normal_previous = std::vector<Vector3f>(num_pixels);

    auto *linearIcp = new LinearICPOptimizer(ICP_ITERATIONS);
    auto *ceresICP = new CeresICPOptimizer(ICP_ITERATIONS);
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
            bool succ;
            if (ICP_VERSION == 2) {
                sample(vertex_current, curFrame.getVertexMapGlobal(), normal_current, curFrame.getNormalMapGlobal(), num_pixels);
                sample(vertex_previous, prevFrame.getVertexMapGlobal(), normal_previous, prevFrame.getNormalMapGlobal(), num_pixels);
                pose_cur = pose_prev;
                succ = ceresICP -> estimatePose(
                        vertex_previous, normal_previous, vertex_current,normal_current, pose_cur
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
            std::stringstream ss;
            ss << filenameBaseOutNormal << frameCount << ".png";
            FreeImageB::SaveImageToFile(curFrame.getNormalMap(), ss.str(),sensor.GetColorImageWidth(),  sensor.GetColorImageHeight(), false);

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

//               bool* visitedVoxels = volume.getVisitedVoxels();

//                for (auto &visitedVoxel : visitedVoxels) {
//                    Vector3i voxelCoords = visitedVoxel.first;
//                    ProcessVolumeCell(&volume, voxelCoords[0], voxelCoords[1], voxelCoords[2], 0.00f, &mesh);
//                }

                for (int x = 0; x < volume.getDimX() - 1; x++)
                {
                    for (int y = 0; y < volume.getDimY() - 1; y++)
                    {
                        for (int z = 0; z < volume.getDimZ() - 1; z++)
                        {
                            if (volume.voxelVisited(x, y, z)) {
                                ProcessVolumeCell(&volume, x, y, z, 0.00f, &mesh);
                            }

                        }
                    }
                }

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
