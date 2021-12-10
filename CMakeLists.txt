cmake_minimum_required(VERSION 3.6)

SET(EXTRA_WARNING_FLAGS "-Wsign-compare")
SET(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
SET(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)

project(kinect_fusion)

# Mac only stuff
if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(MACOSX TRUE)
endif()

# Options
set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src CACHE PATH "Path to source folder")
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include CACHE PATH "Path to source folder")
set(HELPERS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/helpers CACHE PATH "Path to helpers folder")
set(ICP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/icp CACHE PATH "Path to icp folder")
set(RAYCASTER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/raycaster CACHE PATH "Path to RayCaster folder")
set(MODELS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/models CACHE PATH "Path to models folder")
set(LIBRARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/libs CACHE PATH "Path to lib folder")
set(CUDA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda_boost CACHE PATH "Path to cuda booster source folder")
set(Eigen3_INCLUDE_DIR ${LIBRARY_DIR}/eigen/ CACHE PATH "Path to Eigen source folder")


# Use pre-compiled libraries on Windows
if(WIN32)
    set(FreeImage_INCLUDE_DIR ${LIBRARY_DIR}/FreeImage/x64/ CACHE PATH "Path to FreeImage header file")
    set(FreeImage_LIBRARY_DIR ${LIBRARY_DIR}/FreeImage/x64/ CACHE PATH "Path to FreeImage .lib/.dll folder")
endif(WIN32)

# Use 
if(MACOSX)  
    set(FreeImage_INCLUDE_DIR "/usr/local/Cellar/freeimage/3.18.0/include")
    set(FreeImage_LIBRARY_DIR "/usr/local/Cellar/freeimage/3.18.0/lib")
    set(Eigen3_INCLUDE_DIR /usr/local/Cellar/eigen/3.3.9/include/eigen3)
endif(MACOSX)
set(Eigen3_DIR ${LIBRARY_DIR}/Eigen/share/eigen3/cmake CACHE PATH "Path to installed Eigen")
find_package(Eigen3 REQUIRED)

message(${Eigen3_INCLUDE_DIR})

# Include directories
set(INCLUDE 
    ${SOURCE_DIR}
    ${INCLUDE_DIR}
    ${HELPERS_DIR}
    ${ICP_DIR}
    ${MODELS_DIR}
    ${RAYCASTER_DIR}
    ${LIBRARY_DIR}
    ${Eigen3_INCLUDE_DIR})

include_directories(${INCLUDE})

set(HEADERS 
    ${INCLUDE_DIR}/Eigen.h
    ${INCLUDE_DIR}/FreeImageHelper.h
    ${INCLUDE_DIR}/VirtualSensor.h
    ${INCLUDE_DIR}/ICP.h
    ${INCLUDE_DIR}/Frame.h
    ${INCLUDE_DIR}/Volume.h
    ${INCLUDE_DIR}/SimpleMesh.h
    ${INCLUDE_DIR}/MarchingCubes.h
    ${INCLUDE_DIR}/RayCaster.h
    ${INCLUDE_DIR}/Ray.h)

set(SOURCES
    main.cpp
    ${HELPERS_DIR}/FreeImageHelper.cpp
    ${MODELS_DIR}/Frame.cpp
    ${MODELS_DIR}/Volume.cpp
    ${ICP_DIR}/ICP.cpp
    ${RAYCASTER_DIR}/RayCaster.cpp
    ${RAYCASTER_DIR}/Ray.cpp
    )

add_library(kernels
        ${CUDA_DIR}/volume.cu
        )
enable_language(CUDA)


target_compile_features(kernels PUBLIC cxx_std_11)
set_target_properties(kernels
        PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(kernels)

link_directories(${FreeImage_LIBRARY_DIR})
add_executable(kinect_fusion ${SOURCES} ${HEADERS})
target_include_directories(kinect_fusion PUBLIC ${Eigen3_INCLUDE_DIR} ${FreeImage_INCLUDE_DIR} ${INCLUDE})
target_link_libraries(kinect_fusion general freeimage kernels)

# On Windows copy dll to executable folder
if(WIN32)
    add_custom_command(TARGET kinect_fusion POST_BUILD        # Adds a post-build event to MyTest
        COMMAND ${CMAKE_COMMAND} -E copy_if_different  # which executes "cmake - E copy_if_different..."
            "${LIBRARY_DIR}/FreeImage/x64/FreeImage.dll"      # <--this is in-file
            $<TARGET_FILE_DIR:kinect_fusion>)                 # <--this is out-file path
endif(WIN32)
