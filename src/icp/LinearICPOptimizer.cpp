//
// Created by Hongkuan Zhou on 18.12.21.
//
#include "ICPOptimizer.h"


LinearICPOptimizer::LinearICPOptimizer() = default;
LinearICPOptimizer::~LinearICPOptimizer()
{
    std::cout << "LinearICPOptimizer deleted!" << std::endl;
}

bool LinearICPOptimizer::estimatePose(std::vector<Vector3f> vertex_current, std::vector<Vector3f> normal_current,
                                          std::vector<Vector3f> vertex_prediction,
                                          std::vector<Vector3f> normal_prediction, Matrix4f &initialPose) {
    // Build the index of the FLANN tree (for fast nearest neighbor lookup).
    m_nearestNeighborSearch->buildIndex(vertex_current);

    // The initial estimate can be given as an argument.
    Matrix4f estimatedPose = initialPose;
    bool success = true;
    for (int i = 0; i < (int)m_nIterations; ++i)
    {
        // Compute the matches.
        std::cout << "Matching points ..." << std::endl;
        clock_t begin = clock();

        std::vector<Vector3f> transformedPoints = transformPoints(vertex_prediction, estimatedPose);
        std::vector<Vector3f> transformedNormals = transformNormals(normal_prediction, estimatedPose);

        auto matches = m_nearestNeighborSearch->queryMatches(transformedPoints);
        pruneCorrespondences(transformedNormals, normal_current, matches);

        clock_t end = clock();
        double elapsedSecs = double(end - begin) / CLOCKS_PER_SEC;
        std::cout << "Completed in " << elapsedSecs << " seconds." << std::endl;

        std::vector<Vector3f> sourcePoints;
        std::vector<Vector3f> targetPoints;

        // Add all matches to the sourcePoints and targetPoints vectors,
        // so that sourcePoints[i] matches targetPoints[i].
        for (int j = 0; j < (int)transformedPoints.size(); j++)
        {
            const auto &match = matches[j];
            if (match.idx >= 0)
            {
                sourcePoints.push_back(transformedPoints[j]);
                targetPoints.push_back(vertex_current[match.idx]);
            }
        }
        if (sourcePoints.size() < MINIMUM_MATCHING_NUMBER) {
            success = false;
            break;
        }
        std:: cout << "sourcePoints" << sourcePoints.size() << std:: endl;
        std:: cout << "targetPoint" << targetPoints.size() << std::endl;
        // Estimate the new pose
        if (m_bUsePointToPlaneConstraints)
        {
            estimatedPose = estimatePosePointToPlane(sourcePoints, targetPoints, normal_current) * estimatedPose;
        }
        else
        {
            estimatedPose = estimatePosePointToPoint(sourcePoints, targetPoints) * estimatedPose;
        }

        std::cout << "Optimization iteration done." << std::endl;
    }
    initialPose = estimatedPose;
    return success;
}


Matrix4f LinearICPOptimizer::estimatePosePointToPoint(const std::vector<Vector3f> &sourcePoints, const std::vector<Vector3f> &targetPoints)
{
    ProcrustesAligner procrustAligner;
    Matrix4f estimatedPose = procrustAligner.estimatePose(sourcePoints, targetPoints);
    // estimatedPose = Matrix4f::Identity();
    return estimatedPose;
}

Matrix4f LinearICPOptimizer::estimatePosePointToPlane(const std::vector<Vector3f> &sourcePoints, const std::vector<Vector3f> &targetPoints, const std::vector<Vector3f> &targetNormals)
{
    const unsigned nPoints = sourcePoints.size();

    // Build the system
    MatrixXf A = MatrixXf::Zero(4 * nPoints, 6);
    VectorXf b = VectorXf::Zero(4 * nPoints);

    for (unsigned i = 0; i < nPoints; i++)
    {
        Vector3f s = sourcePoints[i];
        Vector3f d = targetPoints[i];
        Vector3f n = targetNormals[i];

        //Add the point-to-plane constraints to the system
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

    // Solve the system
    VectorXf x(6);
    JacobiSVD<MatrixXf> svd(A, ComputeThinU | ComputeThinV);
    x = svd.solve(b);
    float alpha = x(0), beta = x(1), gamma = x(2);

    // Build the pose matrix
    Matrix3f rotation = AngleAxisf(alpha, Vector3f::UnitX()).toRotationMatrix() *
                        AngleAxisf(beta, Vector3f::UnitY()).toRotationMatrix() *
                        AngleAxisf(gamma, Vector3f::UnitZ()).toRotationMatrix();

    Vector3f translation = x.tail(3);


    Matrix4f estimatedPose = Matrix4f::Identity();
    estimatedPose.block<3, 3>(0, 0) = rotation;
    estimatedPose.block<3, 1>(0, 3) = translation;
    // estimatedPose(0,0) = 1; estimatedPose(0,1) = -x[2]; estimatedPose(0,2) = x[1]; estimatedPose(0,3) = x[3];
    // estimatedPose(1,0) = x[2]; estimatedPose(1,1) = 1; estimatedPose(1,2) = -x[0]; estimatedPose(1,3) = x[4];
    // estimatedPose(2,0) = -x[1]; estimatedPose(2,1) = x[0]; estimatedPose(2,2) = 1; estimatedPose(2,3) = x[5];
    // estimatedPose(3,0) = 0; estimatedPose(3,1) = 0; estimatedPose(3,2) = 0; estimatedPose(3,3) = 1;

    // std::cout << estimatedPose << std::endl;
    return estimatedPose;
}