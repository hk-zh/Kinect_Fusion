//
// Created by 周泓宽 on 17.12.21.
//

#ifndef KINECT_FUSION_ICPOPTIMIZER_H
#define KINECT_FUSION_ICPOPTIMIZER_H
#endif //KINECT_FUSION_ICPOPTIMIZER_H

#include <Eigen.h>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <flann/flann.hpp>
#include "NearestNeighbour.h"
#include "ProcrustesAligner.h"

#define MINIMUM_MATCHING_NUMBER 50

/**
 * Helper methods for writing Ceres cost functions.
 */
template <typename T>
static inline void fillVector(const Vector3f &input, T *output)
{
    output[0] = T(input[0]);
    output[1] = T(input[1]);
    output[2] = T(input[2]);
}

/**
 * Pose increment is only an interface to the underlying array (in constructor, no copy
 * of the input array is made).
 * Important: Input array needs to have a size of at least 6.
 */
template <typename T>
class PoseIncrement
{
public:
    explicit PoseIncrement(T *const array) : m_array{array} {}

    void setZero()
    {
        for (int i = 0; i < 6; ++i)
            m_array[i] = T(0);
    }

    T *getData() const
    {
        return m_array;
    }

    /**
     * Applies the pose increment onto the input point and produces transformed output point.
     * Important: The memory for both 3D points (input and output) needs to be reserved (i.e. on the stack)
     * beforehand).
     */
    void apply(T *inputPoint, T *outputPoint) const
    {
        // pose[0,1,2] is angle-axis rotation.
        // pose[3,4,5] is translation.
        const T *rotation = m_array;
        const T *translation = m_array + 3;

        T temp[3];
        ceres::AngleAxisRotatePoint(rotation, inputPoint, temp);

        outputPoint[0] = temp[0] + translation[0];
        outputPoint[1] = temp[1] + translation[1];
        outputPoint[2] = temp[2] + translation[2];
    }

    /**
     * Converts the pose increment with rotation in SO3 notation and translation as 3D vector into
     * transformation 4x4 matrix.
     */
    static Matrix4f convertToMatrix(const PoseIncrement<double> &poseIncrement)
    {
        // pose[0,1,2] is angle-axis rotation.
        // pose[3,4,5] is translation.
        double *pose = poseIncrement.getData();
        double *rotation = pose;
        double *translation = pose + 3;

        // Convert the rotation from SO3 to matrix notation (with column-major storage).
        double rotationMatrix[9];
        ceres::AngleAxisToRotationMatrix(rotation, rotationMatrix);

        // Create the 4x4 transformation matrix.
        Matrix4f matrix;
        matrix.setIdentity();
        matrix(0, 0) = float(rotationMatrix[0]);
        matrix(0, 1) = float(rotationMatrix[3]);
        matrix(0, 2) = float(rotationMatrix[6]);
        matrix(0, 3) = float(translation[0]);
        matrix(1, 0) = float(rotationMatrix[1]);
        matrix(1, 1) = float(rotationMatrix[4]);
        matrix(1, 2) = float(rotationMatrix[7]);
        matrix(1, 3) = float(translation[1]);
        matrix(2, 0) = float(rotationMatrix[2]);
        matrix(2, 1) = float(rotationMatrix[5]);
        matrix(2, 2) = float(rotationMatrix[8]);
        matrix(2, 3) = float(translation[2]);

        return matrix;
    }

private:
    T *m_array;
};

/**
 * Optimization constraints.
 */
class PointToPointConstraint
{
public:
    PointToPointConstraint(const Vector3f &sourcePoint, const Vector3f &targetPoint, const float weight) : m_sourcePoint{sourcePoint},
                                                                                                           m_targetPoint{targetPoint},
                                                                                                           m_weight{weight}
    {
    }

    template <typename T>
    bool operator()(const T *const pose, T *residuals) const
    {

        PoseIncrement<T> poseIncrement = PoseIncrement<T>(const_cast<T *const>(pose));

        // TODO: Implement the point-to-point cost function.
        // The resulting 3D residual should be stored in the residuals array. To apply the pose
        // increment (pose parameters) to the source point, you can use the PoseIncrement class.
        // Important: Ceres automatically squares the cost function.
        residuals[0] = T(0);
        residuals[1] = T(0);
        residuals[2] = T(0);
        T output[3];
        T input[3] = {T(m_sourcePoint[0]), T(m_sourcePoint[1]), T(m_sourcePoint[2])};
        poseIncrement.apply(input, output);
        residuals[0] = (output[0] - T(m_targetPoint[0]));
        residuals[1] = (output[1] - T(m_targetPoint[1]));
        residuals[2] = (output[2] - T(m_targetPoint[2]));

        return true;
    }

    static ceres::CostFunction *create(const Vector3f &sourcePoint, const Vector3f &targetPoint, const float weight)
    {
        return new ceres::AutoDiffCostFunction<PointToPointConstraint, 3, 6>(
                new PointToPointConstraint(sourcePoint, targetPoint, weight));
    }

protected:
    const Vector3f m_sourcePoint;
    const Vector3f m_targetPoint;
    const float m_weight;
    const float LAMBDA = 0.1f;
};

class PointToPlaneConstraint
{
public:
    PointToPlaneConstraint(const Vector3f &sourcePoint, const Vector3f &targetPoint, const Vector3f &targetNormal, const float weight) : m_sourcePoint{sourcePoint},
                                                                                                                                         m_targetPoint{targetPoint},
                                                                                                                                         m_targetNormal{targetNormal},
                                                                                                                                         m_weight{weight}
    {
    }

    template <typename T>
    bool operator()(const T *const pose, T *residuals) const
    {

        PoseIncrement<T> poseIncrement = PoseIncrement<T>(const_cast<T *const>(pose));

        // TODO: Implement the point-to-plane cost function.
        // The resulting 1D residual should be stored in the residuals array. To apply the pose
        // increment (pose parameters) to the source point, you can use the PoseIncrement class.
        // Important: Ceres automatically squares the cost function.
        residuals[0] = T(0);
        T output[3];
        T input[3] = {T(m_sourcePoint[0]), T(m_sourcePoint[1]), T(m_sourcePoint[2])};
        poseIncrement.apply(input, output);
        residuals[0] = (T(m_targetNormal[0]) * (T(m_targetPoint[0]) - output[0]) + T(m_targetNormal[1]) * (T(m_targetPoint[1]) - output[1]) + T(m_targetNormal[2]) * (T(m_targetPoint[2]) - output[2]));
        return true;
    }

    static ceres::CostFunction *create(const Vector3f &sourcePoint, const Vector3f &targetPoint, const Vector3f &targetNormal, const float weight)
    {
        return new ceres::AutoDiffCostFunction<PointToPlaneConstraint, 1, 6>(
                new PointToPlaneConstraint(sourcePoint, targetPoint, targetNormal, weight));
    }

protected:
    const Vector3f m_sourcePoint;
    const Vector3f m_targetPoint;
    const Vector3f m_targetNormal;
    const float m_weight;
    const float LAMBDA = 1.0f;
};

class ICPOptimizer {
public:
    ICPOptimizer();
    void setMatchingMaxDistance(float maxDistance);
    void usePointToPlaneConstraints(bool bUsePointToPlaneConstraints);
    void setNbOfIterations(unsigned nIterations);
    virtual bool estimatePose(std::vector<Vector3f> vertex_current, std::vector<Vector3f> normal_current, std::vector<Vector3f> vertex_prediction, std::vector<Vector3f> normal_prediction, Matrix4f &initialPose);
    ~ICPOptimizer();
protected:
    bool m_bUsePointToPlaneConstraints;
    unsigned m_nIterations;
    std::unique_ptr<NearestNeighborSearch> m_nearestNeighborSearch;
    static std::vector<Vector3f> transformPoints(const std::vector<Vector3f> &sourcePoints, const Matrix4f &pose);
    static std::vector<Vector3f> transformNormals(const std::vector<Vector3f> &sourceNormals, const Matrix4f &pose);
    static void pruneCorrespondences(const std::vector<Vector3f> &sourceNormals, const std::vector<Vector3f> &targetNormals, std::vector<Match> &matches);
};



class CeresICPOptimizer : public ICPOptimizer
{
public:
    CeresICPOptimizer();
    bool estimatePose(std::vector<Vector3f> vertex_current, std::vector<Vector3f> normal_current, std::vector<Vector3f> vertex_prediction, std::vector<Vector3f> normal_prediction, Matrix4f &initialPose ) override;
    ~CeresICPOptimizer();

private:
    static void configureSolver(ceres::Solver::Options &options);
    void prepareConstraints(const std::vector<Vector3f> &sourcePoints, const std::vector<Vector3f> &targetPoints, const std::vector<Vector3f> &targetNormals, const std::vector<Match> &matches, const PoseIncrement<double> &poseIncrement, ceres::Problem &problem) const;
};


class LinearICPOptimizer : public ICPOptimizer
{
public:
    LinearICPOptimizer();
    ~LinearICPOptimizer();
    bool estimatePose(std::vector<Vector3f> vertex_current, std::vector<Vector3f> normal_current, std::vector<Vector3f> vertex_prediction, std::vector<Vector3f> normal_prediction, Matrix4f &initialPose) override;

private:
    static Matrix4f estimatePosePointToPoint(const std::vector<Vector3f> &sourcePoints, const std::vector<Vector3f> &targetPoints);
    static Matrix4f estimatePosePointToPlane(const std::vector<Vector3f> &sourcePoints, const std::vector<Vector3f> &targetPoints, const std::vector<Vector3f> &targetNormals);
};