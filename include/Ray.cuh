#pragma once

#ifndef RAY_H
#define RAY_H

#include "Eigen.h"

class Ray {
private:
	Vector3f start;
	Vector3f direction;
	Vector3f currentPoint;
	Vector3f previousPoint;

public:
    __device__ __host__ Ray();
    __device__ __host__ Ray(Vector3f& start_, Vector3f& direction_);

    __device__ __host__ Vector3f& next();

    __device__ __host__ Vector3f& getStartingPoint();
    __device__ __host__ void setStartingPoint(Vector3f& start_);

    __device__ __host__ Vector3f& getDirection();
    __device__ __host__ void setDirection(Vector3f& direction_);

    __device__ __host__ Vector3f& getCurrentPosition();
    __device__ __host__ Vector3f& getPreviousPosition();
};

#endif //RAY_H