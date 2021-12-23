#include "ICP.h"

#include <iostream>
#include <memory>
#include <utility>

ICP::ICP(Frame &_prevFrame, Frame &_curFrame, const double distanceThreshold,
         const double normalThreshold)
        : prevFrame(_prevFrame),
          curFrame(_curFrame),
          distanceThreshold(distanceThreshold),
          normalThreshold(normalThreshold) {}

bool ICP::estimatePose(
        Eigen::Matrix4f &estimatedPose,
        int iterationsNum
) {
    bool succ = true;
    for (size_t iteration = 0; iteration < iterationsNum; iteration++) {
        const std::vector<std::pair<size_t, size_t>> correspondenceIds = findIndicesOfCorrespondingPoints(
                estimatedPose);
        if (correspondenceIds.size() < 100) {
            succ = false;
            break;
        }
        std::cout << "# corresponding points: " << correspondenceIds.size()
                  << std::endl;
        std::cout << "# total number of points: "
                  << curFrame.getVertexMap().size() << std::endl;

        int nPoints = (int) correspondenceIds.size();
        Eigen::Matrix3f rotationEP = estimatedPose.block(0, 0, 3, 3);
        Eigen::Vector3f translationEP = estimatedPose.block(0, 3, 3, 1);

        MatrixXf A = MatrixXf::Zero(4 * nPoints, 6);
        VectorXf b = VectorXf::Zero(4 * nPoints);

        for (size_t i = 0; i < nPoints; i++) {
            auto pair = correspondenceIds[i];
            Eigen::Vector3f s = rotationEP * curFrame.getVertexGlobal(pair.second) + translationEP;
            Eigen::Vector3f d = prevFrame.getVertexGlobal(pair.first);
            Eigen::Vector3f n = prevFrame.getNormalGlobal(pair.first);

            A(4 * i, 0) = n.z() * s.y() - n.y() * s.z();
            A(4 * i, 1) = n.x() * s.z() - n.z() * s.x();
            A(4 * i, 2) = n.y() * s.x() - n.x() * s.y();
            A.block<1, 3>(4 * i, 3) = n;
            b(4 * i) = (d - s).dot(n);


            //Add the point-to-point constraints to the system
            A.block<3,3>(4 * i + 1, 0) <<
                                       0.0f, s.z(), -s.y(),
                    -s.z(), 0.0f, s.x(),
                    s.y(), -s.x(), 0.0f;
            A.block<3, 3>(4 * i + 1, 3).setIdentity();
            b.segment<3>(4 * i +1) = d - s;

            // Optionally, apply a higher weight to point-to-plane correspondences
            float pointToPlaneWeight = 1.0f;
            float pointToPointWeight = 0.1f;
            A.block<1, 6>(4 * i, 0) *= pointToPlaneWeight;
            b(4 * i) *= pointToPlaneWeight;
            A.block<3, 6>(4 * i + 1, 0) *= pointToPointWeight;
            b.segment<3>(4 * i + 1) *= pointToPointWeight;
        }

        VectorXf x(6);
        x = A.bdcSvd(ComputeThinU | ComputeThinV).solve(b);

        const float alpha = x(0), beta = x(1), gamma = x(2);
        Matrix3f rotation =
                AngleAxisf(alpha, Vector3f::UnitX()).toRotationMatrix() *
                AngleAxisf(beta, Vector3f::UnitY()).toRotationMatrix() *
                AngleAxisf(gamma, Vector3f::UnitZ()).toRotationMatrix();
        Vector3f translation = x.tail(3);

        Matrix4f curentPose = Matrix4f::Identity();
        curentPose.block<3, 3>(0, 0) = rotation;
        curentPose.block<3, 1>(0, 3) = translation;
        estimatedPose = curentPose * estimatedPose;
    }
    return succ;
}

// Helper method to find corresponding points between curent frame and
// previous frame Reference Paper:
// https://www.cvl.iis.u-tokyo.ac.jp/~oishi/Papers/Alignment/Blais_Registering_multiview_PAMI1995.pdf
// Input: curent frame, previous frame, estimated pose of previous
// frame Output: indices of corresponding vertices in curent and
// previous frame Simple version: only take euclidean distance between
// points into consideration without normals Advanced version: Euclidean
// distance between points + difference in normal angles

std::vector<std::pair<size_t, size_t>> ICP::findIndicesOfCorrespondingPoints(
        const Eigen::Matrix4f &estPose) {
    Eigen::Matrix4f estimatedPose = estPose;
    std::vector<std::pair<size_t, size_t>> indicesOfCorrespondingPoints;

    std::vector<Eigen::Vector3f> prevFrameVertexMapGlobal = prevFrame.getVertexMapGlobal();
    std::vector<Eigen::Vector3f> prevFrameNormalMapGlobal = prevFrame.getNormalMapGlobal();

    std::vector<Eigen::Vector3f> curFrameVertexMapGlobal =
            curFrame.getVertexMapGlobal();
    std::vector<Eigen::Vector3f> curFrameNormalMapGlobal =
            curFrame.getNormalMapGlobal();

    Eigen::Matrix3f rotation = estimatedPose.block(0, 0, 3, 3);
    Eigen::Vector3f translation = estimatedPose.block(0, 3, 3, 1);

    Eigen::Matrix4f estimatedPoseInv = estimatedPose.inverse();

    Eigen::Matrix3f rotationInv = estimatedPoseInv.block(0, 0, 3, 3);
    Eigen::Vector3f translationInv = estimatedPoseInv.block(0, 3, 3, 1);

    // GPU implementation: use a separate thread for every run of the for
    // loop
    for (size_t idx = 0; idx < prevFrameVertexMapGlobal.size(); idx++) {
        Eigen::Vector3f prevPointGlobal = prevFrameVertexMapGlobal[idx];
        Eigen::Vector3f prevNormalGlobal = prevFrameNormalMapGlobal[idx];
        // std::cout << "Curent Point (Camera): " << curPoint[0] << " " <<
        // curPoint[1] << " " << curPoint[2] << std::endl;
        if (prevPointGlobal.allFinite() && prevNormalGlobal.allFinite()) {

            Eigen::Vector3f prevPointCurCamera = rotationInv * prevPointGlobal + translationInv;
            Eigen::Vector3f prevNormalCurCamera = rotationInv * prevFrameNormalMapGlobal[idx];

            // project point from global camera system into camera system of
            // the current frame
            Eigen::Vector3f prevPointCurFrame =
                    curFrame.projectPointIntoFrame(prevPointCurCamera);
            // project point from camera system of the previous frame onto the
            // image plane of the current frame
            Eigen::Vector2i prevPointImgCoordCurFrame =
                    curFrame.projectOntoImgPlane(prevPointCurFrame);

            if (curFrame.containsImgPoint(prevPointImgCoordCurFrame)) {
                size_t curIdx =
                        prevPointImgCoordCurFrame[1] * curFrame.getFrameWidth() +
                        prevPointImgCoordCurFrame[0];

                Eigen::Vector3f curFramePointGlobal = rotation * curFrameVertexMapGlobal[curIdx] + translation;
                Eigen::Vector3f curFrameNormalGlobal = rotation * curFrameNormalMapGlobal[curIdx];

                if (curFramePointGlobal.allFinite() &&
                    (curFramePointGlobal - prevPointGlobal).norm() <
                    distanceThreshold &&
                    curFrameNormalGlobal.allFinite() &&
                    (std::abs(curFrameNormalGlobal.dot(prevNormalGlobal)) / curFrameNormalGlobal.norm() /
                     prevNormalGlobal.norm() <
                     normalThreshold)) {
                    indicesOfCorrespondingPoints.emplace_back(idx, curIdx);
                }
            }
        }
    }
    return indicesOfCorrespondingPoints;
}