//
// Created by Hongkuan Zhou on 18.12.21.
//

#include "ICPOptimizer.h"

CeresICPOptimizer::CeresICPOptimizer() = default;

Matrix4f CeresICPOptimizer::estimatePose(std::vector<Vector3f> vertex_current, std::vector<Vector3f> normal_current,
                                         std::vector<Vector3f> vertex_prediction,
                                         std::vector<Vector3f> normal_prediction, Matrix4f initialPose)
{
    // Build the index of the FLANN tree (for fast nearest neighbor lookup).
    m_nearestNeighborSearch->buildIndex(vertex_current);

    // The initial estimate can be given as an argument.
    Matrix4f estimatedPose = initialPose;

    // We optimize on the transformation in SE3 notation: 3 parameters for the axis-angle vector of the rotation (its length presents
    // the rotation angle) and 3 parameters for the translation vector.
    double incrementArray[6];
    auto poseIncrement = PoseIncrement<double>(incrementArray);
    poseIncrement.setZero();

    for (int i = 0; i < (int)m_nIterations; ++i)
    {
        // Compute the matches.
        std::cout << "Matching points ..." << std::endl;
        clock_t begin = clock();

        auto transformedPoints = transformPoints(vertex_prediction, estimatedPose);
        auto transformedNormals = transformNormals(normal_prediction, estimatedPose);

        auto matches = m_nearestNeighborSearch->queryMatches(transformedPoints);
        pruneCorrespondences(transformedNormals, normal_current, matches);

        clock_t end = clock();
        double elapsedSecs = double(end - begin) / CLOCKS_PER_SEC;
        std::cout << "Completed in " << elapsedSecs << " seconds." << std::endl;

        // Prepare point-to-point and point-to-plane constraints.
        ceres::Problem problem;
        prepareConstraints(transformedPoints, vertex_current, normal_current, matches, poseIncrement, problem);

        // Configure options for the solver.
        ceres::Solver::Options options;
        configureSolver(options);

        // Run the solver (for one iteration).
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        std::cout << summary.BriefReport() << std::endl;
        //std::cout << summary.FullReport() << std::endl;

        // Update the current pose estimate (we always update the pose from the left, using left-increment notation).
        Matrix4f matrix = PoseIncrement<double>::convertToMatrix(poseIncrement);
        estimatedPose = matrix * estimatedPose;
        poseIncrement.setZero();

        std::cout << "Optimization iteration done." << std::endl;
    }

    return estimatedPose;
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



LinearICPOptimizer::LinearICPOptimizer() = default;
LinearICPOptimizer::~LinearICPOptimizer()
{
    std::cout << "LinearICPOptimizer deleted!" << std::endl;
}

Matrix4f LinearICPOptimizer::estimatePose(std::vector<Vector3f> vertex_current, std::vector<Vector3f> normal_current,
                                          std::vector<Vector3f> vertex_prediction,
                                          std::vector<Vector3f> normal_prediction, Matrix4f initialPose) {
    // Build the index of the FLANN tree (for fast nearest neighbor lookup).
    m_nearestNeighborSearch->buildIndex(vertex_current);

    // The initial estimate can be given as an argument.
    Matrix4f estimatedPose = initialPose;

    for (int i = 0; i < (int)m_nIterations; ++i)
    {
        // Compute the matches.
        std::cout << "Matching points ..." << std::endl;
        clock_t begin = clock();

        auto transformedPoints = transformPoints(vertex_prediction, estimatedPose);
        auto transformedNormals = transformNormals(normal_prediction, estimatedPose);

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

    return estimatedPose;
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
        const auto &s = sourcePoints[i];
        const auto &d = targetPoints[i];
        const auto &n = targetNormals[i];

        //Add the point-to-plane constraints to the system
        A(4 * i, 0) = n[2] * s[1] - n[1] * s[2];
        A(4 * i, 1) = n[0] * s[2] - n[2] * s[0];
        A(4 * i, 2) = n[1] * s[0] - n[0] * s[1];
        A(4 * i, 3) = n[0];
        A(4 * i, 4) = n[1];
        A(4 * i, 5) = n[2];
        b[4 * i] = n[0] * d[0] + n[1] * d[1] + n[2] * d[2] - n[0] * s[0] - n[1] * s[1] - n[2] * s[2];
        //Add the point-to-point constraints to the system

        A(4 * i + 1, 0) = 0;
        A(4 * i + 1, 1) = s[2];
        A(4 * i + 1, 2) = -s[1];
        A(4 * i + 1, 3) = 1;
        A(4 * i + 1, 4) = 0;
        A(4 * i + 1, 5) = 0;
        b[4 * i + 1] = d[0] - s[0];

        A(4 * i + 2, 0) = -s[2];
        A(4 * i + 2, 1) = 0;
        A(4 * i + 2, 2) = s[0];
        A(4 * i + 2, 3) = 0;
        A(4 * i + 2, 4) = 1;
        A(4 * i + 2, 5) = 0;
        b[4 * i + 2] = d[1] - s[1];

        A(4 * i + 3, 0) = s[1];
        A(4 * i + 3, 1) = -s[0];
        A(4 * i + 3, 2) = 0;
        A(4 * i + 3, 3) = 0;
        A(4 * i + 3, 4) = 0;
        A(4 * i + 3, 5) = 1;
        b[4 * i + 3] = d[2] - s[2];
        // Optionally, apply a higher weight to point-to-plane correspondences
        A(4 * i) = 10 * A(4 * i);
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