#include "RayCaster.h"

//RayCaster::RayCaster() {}

RayCaster::RayCaster(Volume& vol_) : vol(vol_) {}

void RayCaster::changeFrame(Frame& frame_) {
	frame = frame_;
}

// a function that writes down the invalid results
void mistake(
	std::vector<Vector3f>& ovg, std::vector<Vector4uc>& ocg
) {
	ovg.emplace_back(Vector3f(MINF, MINF, MINF));
	ocg.emplace_back(Vector4uc(0, 0, 0, 0));
}

Frame& RayCaster::rayCast() {
	const Matrix4f worldToCamera = frame.getExtrinsicMatrix();
	const Matrix4f cameraToWorld = worldToCamera.inverse();
	const Matrix3f intrinsic_inverse = frame.getIntrinsicMatrix().inverse();
	Vector3f translation = cameraToWorld.block(0, 3, 3, 1);
	Matrix3f rotationMatrix = cameraToWorld.block(0, 0, 3, 3);
	Vector4uc color;

	int width = frame.getFrameWidth();
	int height = frame.getFrameHeight();

	Vector3f ray_start, ray_dir, ray_current, ray_previous, ray_next;
	Vector3i ray_current_int, ray_previous_int;

	std::shared_ptr<std::vector<Vector3f>> output_vertices_global = std::make_shared<std::vector<Vector3f>>(std::vector<Vector3f>());
	output_vertices_global->reserve(width * height);

//	std::shared_ptr<std::vector<Vector3f>> output_normals_global = std::make_shared<std::vector<Vector3f>>(std::vector<Vector3f>());
//	output_normals_global->reserve(width * height);

    std::shared_ptr<std::vector<Vector4uc>> output_colors_global = std::make_shared<std::vector<Vector4uc>>(std::vector<Vector4uc>());
    output_colors_global->reserve(width * height);

	std::shared_ptr<std::vector<Vector3f>> output_vertices, output_normals, output_colors;

	float sdf_1, sdf_2;
	Vector3f p, v, n;
	uint index;

	std::cout << "RayCast starting..." << std::endl;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			// starting point is the position of the camera (translation) in grid coordinates
			ray_start = vol.worldToGrid(translation);

			// calculate the direction vector as vector from camera position to the pixel(i, j)s world coordinates

			ray_next = Vector3f{ float(j), float(i), 1.0f };
			ray_next = intrinsic_inverse * ray_next;
			ray_next = rotationMatrix * ray_next + translation;
			ray_next = vol.worldToGrid(ray_next);
			
			ray_dir = ray_next - ray_start;
			ray_dir = ray_dir.normalized();
			
			if (!ray_dir.allFinite() || ray_dir == Vector3f{ 0.0f, 0.0f, 0.0f }) {
				mistake(*output_vertices_global, *output_colors_global);
				continue;
			}

			Ray ray = Ray(ray_start, ray_dir);

			ray_current = ray_start;
            // forward until the ray in range
            int cnt = 0;
            while (!vol.isInterpolationPossible(ray_current) && cnt++ < 2000) {
                ray_current = ray.next();
            }
            ray_current_int = Volume::intCoords(ray_current);

			if (!vol.isPointInVolume(ray_current)) {
				mistake(*output_vertices_global, *output_colors_global);
				continue;
			}

			while (true) {//vol.isPointInVolume(ray_current)) {

				ray_previous = ray_current;
				ray_previous_int = ray_current_int;

                // until reach the next grid
                int k = 0;
				do {
					ray_current = ray.next();
					ray_current_int = Volume::intCoords(ray_current);
					k++;
				} while (ray_previous_int == ray_current_int);

					
				if (!vol.isInterpolationPossible(ray_previous) || !vol.isInterpolationPossible(ray_current)) {
                    mistake(*output_vertices_global, *output_colors_global);
                    break;
                } else if (vol.get(ray_previous_int).getValue() == 0) {
					v = vol.gridToWorld(ray_previous);
					output_vertices_global->emplace_back(v);
                    output_colors_global->emplace_back(vol.get(ray_previous_int).getColor());
					if (!vol.voxelVisited(ray_previous)) {
						vol.setVisited(ray_previous_int);
					}

					break;
				} else if (vol.get(ray_current_int).getValue() == 0) {
                    v = vol.gridToWorld(ray_current);
                    output_vertices_global->emplace_back(v);
                    output_colors_global->emplace_back(vol.get(ray_current_int).getColor());

                    if (!vol.voxelVisited(ray_current)) {
                        vol.setVisited(ray_current_int);
                    }

                    break;
                } else if (
					vol.get(ray_previous_int).getValue() != std::numeric_limits<float>::max() && 
					vol.get(ray_previous_int).getValue() > 0  &&
					vol.get(ray_current_int).getValue() != std::numeric_limits<float>::max() &&
					vol.get(ray_current_int).getValue() < 0
				) {
					sdf_1 = vol.trilinearInterpolation(ray_previous);
					sdf_2 = vol.trilinearInterpolation(ray_current);

//					sdf_1 = vol.get(ray_previous_int).getValue();
//					sdf_2 = vol.get(ray_current_int).getValue();

					if (sdf_1 == std::numeric_limits<float>::max() || sdf_2 == std::numeric_limits<float>::max()) {
                        mistake(*output_vertices_global, *output_colors_global);
                        break;
					}

					p = ray_previous - (ray_dir.normalized() * ray.forwardLength() * k * sdf_1) / (sdf_2 - sdf_1);

					if (!vol.isInterpolationPossible(p)) {
						mistake(*output_vertices_global, *output_colors_global);
						break;
					}
                    if (vol.get(ray_previous_int).isValidColor() && vol.get(ray_current_int).isValidColor()) {
                        Vector4uc color1 = vol.get(ray_previous_int).getColor();
                        Vector4uc color2 = vol.get(ray_current_int).getColor();
                        Vector4uc color = Vector4uc{
                                (const unsigned char) (
                                        ((float) color1[0] * abs(sdf_2) + (float) color2[0] * abs(sdf_1)) /
                                        (abs(sdf_1) + abs(sdf_2))),
                                (const unsigned char) (
                                        ((float) color1[1] * abs(sdf_2) + (float) color2[1] * abs(sdf_1)) /
                                        (abs(sdf_1) + abs(sdf_2))),
                                (const unsigned char) (
                                        ((float) color1[2] * abs(sdf_2) + (float) color2[2] * abs(sdf_1)) /
                                        (abs(sdf_1) + abs(sdf_2))),
                                (const unsigned char) (
                                        ((float) color1[3] * abs(sdf_2) + (float) color2[3] * abs(sdf_1)) /
                                        (abs(sdf_1) + abs(sdf_2))),
                        };
                        output_colors_global->emplace_back(color);
                    } else if (vol.get(ray_previous_int).isValidColor()) {
                        Vector4uc color1 = vol.get(ray_previous_int).getColor();
                        output_colors_global ->emplace_back(color1);
                    } else if (vol.get(ray_current_int).isValidColor()) {
                        Vector4uc color2 = vol.get(ray_current_int).getColor();
                        output_colors_global ->emplace_back(color2);
                    } else {
                        output_colors_global ->emplace_back(Vector4uc {0,0,0,0});
                    }
					v = vol.gridToWorld(p);
					output_vertices_global->emplace_back(v);

					if (!vol.voxelVisited(ray_previous)) {
						vol.setVisited(ray_previous_int);
					}

					if (!vol.voxelVisited(ray_current))
						vol.setVisited(ray_current_int);
					break;
				}
			}
		}			
	}
    std::cout << "output_vertices_global: " << output_vertices_global->size() << std::endl;
	frame.mVerticesGlobal = output_vertices_global;
	frame.mVertices = std::make_shared<std::vector<Vector3f>>(frame.transformPoints(*output_vertices_global, worldToCamera));
	frame.computeNormalMap(width, height);
	frame.mNormalsGlobal = std::make_shared<std::vector<Vector3f>>(frame.rotatePoints(frame.getNormalMap(), rotationMatrix));
    for (int i = 0; i < output_colors_global->size(); i++) {
        frame.colorMap[4*i] =(*output_colors_global)[i][0];
        frame.colorMap[4*i+1] =(*output_colors_global)[i][1];
        frame.colorMap[4*i+2] =(*output_colors_global)[i][2];
        frame.colorMap[4*i+3] =(*output_colors_global)[i][3];
    }
	std::cout << "RayCast done!" << std::endl;

	return frame;
}

