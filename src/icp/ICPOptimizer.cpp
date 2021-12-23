//
// Created by 周泓宽 on 17.12.21.
//

#include "ICPOptimizer.h"

ICPOptimizer::ICPOptimizer() : m_bUsePointToPlaneConstraints{true},
                                   m_nIterations{10},
                                   m_nearestNeighborSearch{std::make_unique<NearestNeighborSearchFlann>()}{}


void ICPOptimizer::setMatchingMaxDistance(float maxDistance) {
    m_nearestNeighborSearch->setMatchingMaxDistance(maxDistance);
}

void ICPOptimizer::usePointToPlaneConstraints(bool bUsePointToPlaneConstraints) {
    m_bUsePointToPlaneConstraints = bUsePointToPlaneConstraints;
}

void ICPOptimizer::setNbOfIterations(unsigned int nIterations) {
    m_nIterations = nIterations;
}

//virtual bool ICPOptimizer::estimatePose(std::vector<Vector3f> vertex_current, std::vector<Vector3f> normal_current,
//                                    std::vector<Vector3f> vertex_prediction, std::vector<Vector3f> normal_prediction,
//                                    Matrix4f &initialPose);

ICPOptimizer:: ~ICPOptimizer()
{
    std::cout << "ICPOptimizer deleted!" << std::endl;
};

std::vector<Vector3f> ICPOptimizer::transformPoints(const std::vector<Vector3f> &sourcePoints, const Matrix4f &pose)
{
    std::vector<Vector3f> transformedPoints;
    transformedPoints.reserve(sourcePoints.size());

    const auto rotation = pose.block(0, 0, 3, 3);
    const auto translation = pose.block(0, 3, 3, 1);

    for (const auto &point : sourcePoints)
    {
        transformedPoints.emplace_back(rotation * point + translation);
    }

    return transformedPoints;
}

std::vector<Vector3f> ICPOptimizer::transformNormals(const std::vector<Vector3f> &sourceNormals, const Matrix4f &pose)
{
    std::vector<Vector3f> transformedNormals;
    transformedNormals.reserve(sourceNormals.size());

    const auto rotation = pose.block(0, 0, 3, 3);

    for (const auto &normal : sourceNormals)
    {
        transformedNormals.emplace_back(rotation.inverse().transpose() * normal);
    }

    return transformedNormals;
}

void ICPOptimizer::pruneCorrespondences(const std::vector<Vector3f> &sourceNormals, const std::vector<Vector3f> &targetNormals, std::vector<Match> &matches)
{
    const unsigned nPoints = sourceNormals.size();

    for (unsigned i = 0; i < nPoints; i++)
    {
        Match &match = matches[i];
        if (match.idx >= 0)
        {
            const auto &sourceNormal = sourceNormals[i];
            const auto &targetNormal = targetNormals[match.idx];

            // TODO: Invalidate the match (set it to -1) if the angle between the normals is greater than 60
            double cosin = sourceNormal.dot(targetNormal) / (sourceNormal.norm() * targetNormal.norm());
            if (cosin < 0.5)
            {
                match.idx = -1;
            }
        }
    }
}

bool ICPOptimizer::estimatePose(std::vector<Vector3f> vertex_current, std::vector<Vector3f> normal_current,
                                std::vector<Vector3f> vertex_prediction, std::vector<Vector3f> normal_prediction,
                                Matrix4f &initialPose) {
    return false;
}
