# Kinect_Fusion
## Project Overview
###Introduction
We realize the kinect-fusion based on the paper [KinectFusion: Real-Time Dense Surface Mapping and Tracking](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ismar2011.pdf) from Microsoft Research.

###Result:
<p float="left">
    <img width="500" alt="teddy_bear1" src="https://user-images.githubusercontent.com/57254021/149985195-1454b386-c6cb-4a60-8347-fb0584c90924.png">
    <img width="500" alt="teddy_bear2" src="https://user-images.githubusercontent.com/57254021/149985205-7ed9dcf8-8fde-46d2-ad32-d1c801f89d57.png">
</p>

<p float="left">
    <img width="500" alt="teddy_bear3" src="https://user-images.githubusercontent.com/57254021/149985222-df7c0929-272a-4ce4-b377-b0c69e3b09c2.png">
    <img width="500" alt="teddy_bear4" src="https://user-images.githubusercontent.com/57254021/149985228-78a858ff-7fb6-4ca0-a8c4-0797f96da662.png">
</p>

****

<p float="left">
    <img width="500" alt="teddy_bear1" src="https://user-images.githubusercontent.com/57254021/149985248-adf90b33-43fe-47ec-8bad-20572d5a27c7.png">
    <img width="500" alt="teddy_bear2" src="https://user-images.githubusercontent.com/57254021/149985254-c9e15afe-44bc-4bc0-8361-adbd76828466.png">
</p>

<p float="left">
    <img width="500" alt="teddy_bear3" src="https://user-images.githubusercontent.com/57254021/149986719-a58b2cc9-1a22-4594-815b-83ecb45680e4.png">
    <img width="500" alt="teddy_bear4" src="https://user-images.githubusercontent.com/57254021/149985267-573094a9-747e-4383-b380-1570d750722d.png">
</p>


## Run this code

### Linux

- Install `cmake` if it's not installed yet\
  `sudo apt-get install cmake`

- Run `sudo apt-get install libfreeimage3 libfreeimage-dev` in order to be able to process images from the dataset by simulating the virtal sensor


- Download header only Eigen library and put it into `libs` folder by runnning:\
  `cd KinectFusion/libs`\
  `git clone https://gitlab.com/libeigen/eigen.git`
  
- Install ceres and also flann

  
### MacOS

- `brew install eigen`
- `brew install freeimage`
- `brew install ceres-solver`
- `brew install brew install flann -v`

Note that checkout your CMakeLists.txt so that it links to your installed libraries.

1. Download dataset from https://vision.in.tum.de/data/datasets/rgbd-dataset/download (we used teddy bear and plant datasets)



## CUDA
You could also checkout release-cuda branch to run on CUDA.
These things are implemented on CUDA
* Volume Integration
* Raycasting
* ICP pose estimation (to be continued...)
