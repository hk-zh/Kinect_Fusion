#include "Ray.cuh"
#include "Eigen.h"

__device__ __host__ Ray::Ray() {}

__device__ __host__ Ray::Ray(Vector3f& start_, Vector3f& direction_) : start(start_), direction(direction_) {
	currentPoint = start_;
	direction.normalize();
}

__device__ __host__ Vector3f& Ray::next() {
	previousPoint = currentPoint;
	currentPoint += direction;

	return currentPoint;
}

__device__ __host__ Vector3f& Ray::getStartingPoint() {
	return start;
}
__device__ __host__ void Ray::setStartingPoint(Vector3f& start_) {
	start = start_;
	currentPoint = start_;
}

__device__ __host__ Vector3f& Ray::getDirection() {
	return direction;
}
__device__ __host__ void Ray::setDirection(Vector3f& direction_) {
	direction = direction_;
}

__device__ __host__ Vector3f& Ray::getCurrentPosition() {
	return currentPoint;
}
__device__ __host__ Vector3f& Ray::getPreviousPosition() {
	return previousPoint;
}