//
// Created by Hongkuan Zhou on 18.12.21.
//

#include "ICPOptimizer.h"

CeresICPOptimizer::CeresICPOptimizer(unsigned int m_nIterations) : ICPOptimizer(m_nIterations) {

}

bool CeresICPOptimizer::estimatePose(std::vector<Vector3f> &vertex_previous, std::vector<Vector3f> &normal_previous,
                                         std::vector<Vector3f> &vertex_current,
                                         std::vector<Vector3f> &normal_current, Matrix4f &initialPose)
{
    // Build the index of the FLANN tree (for fast nearest neighbor lookup).
    m_nearestNeighborSearch->buildIndex(vertex_previous);

    // The initial estimate can be given as an argument.
    Matrix4f estimatedPose = initialPose;

    // We optimize on the transformation in SE3 notation: 3 parameters for the axis-angle vector of the rotation (its length presents
    // the rotation angle) and 3 parameters for the translation vector.
    double incrementArray[6];
    auto poseIncrement = PoseIncrement<double>(incrementArray);
    poseIncrement.setZero();
    bool success = true;
    for (int i = 0; i < (int)m_nIterations; ++i)
    {
        // Compute the matches.
        std::cout << "Matching points ..." << std::endl;
        clock_t begin = clock();

        auto transformedPoints = transformPoints(vertex_current, estimatedPose);
        auto transformedNormals = transformNormals(normal_current, estimatedPose);

        auto matches = m_nearestNeighborSearch->queryMatches(transformedPoints);
        pruneCorrespondences(transformedNormals, normal_previous, matches);
        if (matches.size() < MINIMUM_MATCHING_NUMBER) {
            success = false;
            break;
        }
        clock_t end = clock();
        double elapsedSecs = double(end - begin) / CLOCKS_PER_SEC;
        std::cout << "Completed in " << elapsedSecs << " seconds." << std::endl;

        // Prepare point-to-point and point-to-plane constraints.
        ceres::Problem problem;
        prepareConstraints(transformedPoints, vertex_previous, normal_previous, matches, poseIncrement, problem);

        // Configure options for the solver.
        ceres::Solver::Options options;
        configureSolver(options);

        // Run the solver (for one iteration).
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        std::cout << summary.BriefReport() << std::endl;

        // Update the current pose estimate (we always update the pose from the left, using left-increment notation).
        Matrix4f matrix = PoseIncrement<double>::convertToMatrix(poseIncrement);
        estimatedPose = matrix * estimatedPose;
        poseIncrement.setZero();

        std::cout << "Optimization iteration done." << std::endl;
    }
    initialPose = estimatedPose;
    return success;
}

void CeresICPOptimizer::configureSolver(ceres::Solver::Options &options)
{
    // Ceres options.
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.use_nonmonotonic_steps = false;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 1;
    options.num_threads = 8;
}

void CeresICPOptimizer::prepareConstraints(const std::vector<Vector3f> &sourcePoints, const std::vector<Vector3f> &targetPoints, const std::vector<Vector3f> &targetNormals, const std::vector<Match> &matches, const PoseIncrement<double> &poseIncrement, ceres::Problem &problem) const
{
    const unsigned nPoints = sourcePoints.size();

    for (unsigned i = 0; i < nPoints; ++i)
    {
        const auto match = matches[i];
        if (match.idx >= 0)
        {
            const auto &sourcePoint = sourcePoints[i];
            const auto &targetPoint = targetPoints[match.idx];
            const auto &weight = match.weight;
            if (!sourcePoint.allFinite() || !targetPoint.allFinite())
                continue;

            // TODO: Create a new point-to-point cost function and add it as constraint (i.e. residual block)
            // to the Ceres problem.
            problem.AddResidualBlock(
                    PointToPointConstraint::create(sourcePoint, targetPoint, weight), nullptr, poseIncrement.getData());

            if (m_bUsePointToPlaneConstraints)
            {
                const auto &targetNormal = targetNormals[match.idx];

                if (!targetNormal.allFinite())
                    continue;

                // TODO: Create a new point-to-plane cost function and add it as constraint (i.e. residual block)
                // to the Ceres problem.
                problem.AddResidualBlock(
                        PointToPlaneConstraint::create(sourcePoint, targetPoint, targetNormal, weight), nullptr, poseIncrement.getData());
            }
        }
    }
}

CeresICPOptimizer::~CeresICPOptimizer() {
    std::cout << "CeresICPOptimizer deleted!" << std::endl;
}


