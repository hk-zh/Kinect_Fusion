#pragma once

#include <vector>
#include <iostream>
#include <cstring>
#include <fstream>

#include "Eigen.h"
#include "FreeImageHelper.h"
#define M_SIGMA_S 5
#define M_SIGMA_R 1.0f
typedef unsigned char BYTE;

// reads sensor files according to https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
class VirtualSensor
{
public:

	VirtualSensor() : m_currentIdx(-1), m_increment(1) {}

	~VirtualSensor()
	{
		SAFE_DELETE_ARRAY(m_depthFrame)
		SAFE_DELETE_ARRAY(m_colorFrame)
        SAFE_DELETE_ARRAY(m_depthFrame_filtered)
	}

	bool Init(const std::string& datasetDir)
	{
		m_baseDir = datasetDir;

		// read filename lists
		if (!ReadFileList(datasetDir + "depth.txt", m_filenameDepthImages, m_depthImagesTimeStamps)) return false;
		if (!ReadFileList(datasetDir + "rgb.txt", m_filenameColorImages, m_colorImagesTimeStamps)) return false;

		// read tracking
		if (!ReadTrajectoryFile(datasetDir + "groundtruth.txt", m_trajectory, m_trajectoryTimeStamps)) return false;

        //if (m_filenameDepthImages.size() != m_filenameColorImages.size()) return false;

		// image resolutions
		m_colorImageWidth = 640;
		m_colorImageHeight = 480;
		m_depthImageWidth = 640;
		m_depthImageHeight = 480;

		// intrinsics
		m_colorIntrinsics <<	525.0f, 0.0f, 319.5f,
								0.0f, 525.0f, 239.5f,
								0.0f, 0.0f, 1.0f;

		m_depthIntrinsics = m_colorIntrinsics;

		m_colorExtrinsics.setIdentity();
		m_depthExtrinsics.setIdentity();

		m_depthFrame = new float[m_depthImageWidth*m_depthImageHeight];
        m_depthFrame_filtered = new float[m_depthImageWidth*m_depthImageHeight];
		for (unsigned int i = 0; i < m_depthImageWidth*m_depthImageHeight; ++i) m_depthFrame[i] = 0.5f;

		m_colorFrame = new BYTE[4* m_colorImageWidth*m_colorImageHeight];
		for (unsigned int i = 0; i < 4*m_colorImageWidth*m_colorImageHeight; ++i) m_colorFrame[i] = 255;

		m_currentIdx = -1;
		return true;
	}

	bool ProcessNextFrame()
	{
		if (m_currentIdx == -1)	m_currentIdx = 0;
		else m_currentIdx += m_increment;

		if ((unsigned int)m_currentIdx >= (unsigned int)m_filenameColorImages.size()) return false;

		std::cout << "ProcessNextFrame [" << m_currentIdx << " | " << m_filenameColorImages.size() << "]" << std::endl;

		FreeImageB rgbImage;
		rgbImage.LoadImageFromFile(m_baseDir + m_filenameColorImages[m_currentIdx]);
		memcpy(m_colorFrame, rgbImage.data, 4 * 640 * 480);

		// depth images are scaled by 5000 (see https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats)
		FreeImageU16F dImage;
		dImage.LoadImageFromFile(m_baseDir + m_filenameDepthImages[m_currentIdx]);

		for (unsigned int i = 0; i < m_depthImageWidth*m_depthImageHeight; ++i)
		{
			if (dImage.data[i] == 0)
				m_depthFrame[i] = MINF;
			else
				m_depthFrame[i] = dImage.data[i] * 1.0f / 5000.0f;
		}

		// find transformation (simple nearest neighbor, linear search)
		double timestamp = m_depthImagesTimeStamps[m_currentIdx];
		double min = std::numeric_limits<double>::max();
		unsigned idx = 0;
		for (unsigned int i = 0; i < m_trajectory.size(); ++i)
		{
			double d = fabs(m_trajectoryTimeStamps[i] - timestamp);
			if (min > d)
			{
				min = d;
				idx = i;
			}
		}
		m_currentTrajectory = m_trajectory[idx];
        filter_depth_map();

		return true;
	}

	unsigned int GetCurrentFrameCnt() const
	{
		return (unsigned int)m_currentIdx;
	}

	// get current color data
	BYTE* GetColorRGBX()
	{
		return m_colorFrame;
	}
	// get current depth data
	float* GetDepth()
	{
		return m_depthFrame;
	}
    // get current filtered depth data
    float* GetDepthFiltered()
    {
        return m_depthFrame_filtered;
    }

	// color camera info
	Eigen::Matrix3f GetColorIntrinsics()
	{
		return m_colorIntrinsics;
	}

	Eigen::Matrix4f GetColorExtrinsics()
	{
		return m_colorExtrinsics;
	}

	unsigned int GetColorImageWidth() const
	{
		return m_colorImageWidth;
	}

	unsigned int GetColorImageHeight() const
	{
		return m_colorImageHeight;
	}

	// depth (ir) camera info
	Eigen::Matrix3f GetDepthIntrinsics()
	{
		return m_depthIntrinsics;
	}

	Eigen::Matrix4f GetDepthExtrinsics()
	{
		return m_depthExtrinsics;
	}

	unsigned int GetDepthImageWidth() const
	{
		return m_colorImageWidth;
	}

	unsigned int GetDepthImageHeight() const
	{
		return m_colorImageHeight;
	}

	// get current trajectory transformation
	Eigen::Matrix4f GetTrajectory()
	{
		return m_currentTrajectory;
	}

    void filter_depth_map()
    {
        auto n_sigma = [](float x, float sigma)
        {
            //exp(−t^2 * σ^−2 )
            return (float) exp(-pow(x, 2) * pow(sigma, -2));
        };

        auto depth_locator = [&](float *base, size_t x, size_t y)
        {
            return base + x * m_depthImageWidth + y;
        };

        for (size_t ux = 0; ux < m_depthImageHeight; ++ux)
        {
            for (size_t uy = 0; uy < m_depthImageWidth; ++uy)
            {
                if (*depth_locator(m_depthFrame, ux, uy) == MINF)
                {
                    *depth_locator(m_depthFrame_filtered, ux, uy) = MINF;
                    continue;
                }

                float sum_weights = 0;
                float sum_values = 0;

                size_t qx_st = ux > M_SIGMA_S ? ux - M_SIGMA_S : 0;
                size_t qy_st = uy > M_SIGMA_S ? uy - M_SIGMA_S : 0;
                size_t qx_ed = ux + M_SIGMA_S + 1 < m_depthImageHeight ? ux + M_SIGMA_S + 1 : m_depthImageHeight;
                size_t qy_ed = uy + M_SIGMA_S + 1 < m_depthImageWidth ? uy + M_SIGMA_S + 1 : m_depthImageWidth;

                for (size_t qx = qx_st; qx < qx_ed; ++qx)
                {
                    for (size_t qy = qy_st; qy < qy_ed; ++qy)
                    {
                        if (*depth_locator(m_depthFrame, qx, qy) == MINF)
                        {
                            continue;
                        }

                        float loc_diff_norm = (Vector2f(ux, uy) - Vector2f(qx, qy)).norm();
                        float depth_diff_norm = abs(*depth_locator(m_depthFrame, ux, uy) - *depth_locator(m_depthFrame, qx, qy));
                        if (depth_diff_norm > 3.0f * static_cast<float>(M_SIGMA_R))
                        {
                            continue;
                        }
                        float tmp_w = n_sigma(loc_diff_norm, static_cast<float>(M_SIGMA_S)) * n_sigma(depth_diff_norm, M_SIGMA_R);

                        sum_weights += tmp_w;
                        sum_values += tmp_w * (*depth_locator(m_depthFrame, qx, qy));
                    }
                }

                *depth_locator(m_depthFrame_filtered, ux, uy) = sum_values / (sum_weights);
            }
        }
    }
protected:

	static bool ReadFileList(const std::string& filename, std::vector<std::string>& result, std::vector<double>& timestamps)
	{
		std::ifstream fileDepthList(filename, std::ios::in);
		if (!fileDepthList.is_open()) return false;
		result.clear();
		timestamps.clear();
		std::string dump;
		std::getline(fileDepthList, dump);
		std::getline(fileDepthList, dump);
		std::getline(fileDepthList, dump);
		while (fileDepthList.good())
		{

			double timestamp;
			fileDepthList >> timestamp;
			std::string file_name;
			fileDepthList >> file_name;
			if (file_name.empty()) break;
			timestamps.push_back(timestamp);
			result.push_back(file_name);
		}
		fileDepthList.close();
		return true;
	}

	static bool ReadTrajectoryFile(const std::string& filename, std::vector<Eigen::Matrix4f>& result, std::vector<double>& timestamps)
	{
		std::ifstream file(filename, std::ios::in);
		if (!file.is_open()) return false;
		result.clear();
		std::string dump;
		std::getline(file, dump);
		std::getline(file, dump);
		std::getline(file, dump);

		while (file.good())
		{
			double timestamp;
			file >> timestamp;
			Eigen::Vector3f translation;
			file >> translation.x() >> translation.y() >> translation.z();
			Eigen::Quaternionf rot;
			file >> rot;

			Eigen::Matrix4f transf;
			transf.setIdentity();
			transf.block<3, 3>(0, 0) = rot.toRotationMatrix();
			transf.block<3, 1>(0, 3) = translation;

			if (rot.norm() == 0) break;

			transf = transf.inverse().eval();

			timestamps.push_back(timestamp);
			result.push_back(transf);
		}
		file.close();
		return true;
	}

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	// current frame index
	int m_currentIdx;

	int m_increment;

	// frame data
	float* m_depthFrame;
	float* m_depthFrame_filtered;
	BYTE* m_colorFrame;
	Eigen::Matrix4f m_currentTrajectory;

	// color camera info
	Eigen::Matrix3f m_colorIntrinsics;
	Eigen::Matrix4f m_colorExtrinsics;
	unsigned int m_colorImageWidth;
	unsigned int m_colorImageHeight;

	// depth (ir) camera info
	Eigen::Matrix3f m_depthIntrinsics;
	Eigen::Matrix4f m_depthExtrinsics;
	unsigned int m_depthImageWidth;
	unsigned int m_depthImageHeight;

	// base dir
	std::string m_baseDir;
	// filenameList depth
	std::vector<std::string> m_filenameDepthImages;
	std::vector<double> m_depthImagesTimeStamps;
	// filenameList color
	std::vector<std::string> m_filenameColorImages;
	std::vector<double> m_colorImagesTimeStamps;

	// trajectory
	std::vector<Eigen::Matrix4f> m_trajectory;
	std::vector<double> m_trajectoryTimeStamps;
	bool filter;
};


