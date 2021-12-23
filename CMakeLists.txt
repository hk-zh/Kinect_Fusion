cmake_minimum_required(VERSION 3.6)

set(CMAKE_CONFIGURATION_TYPES Debug Release CACHE TYPE INTERNAL FORCE)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_SYSTEM_FLANN "Use system pre-installed flann" ON)

project(kinect_fusion)

# Mac only stuff
if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(MACOSX TRUE)
endif ()

# Options
set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src CACHE PATH "Path to source folder")
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include CACHE PATH "Path to source folder")
set(HELPERS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/helpers CACHE PATH "Path to helpers folder")
set(ICP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/icp CACHE PATH "Path to icp folder")
set(RAYCASTER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/raycaster CACHE PATH "Path to RayCaster folder")
set(MODELS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/models CACHE PATH "Path to models folder")
set(LIBRARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/libs CACHE PATH "Path to lib folder")
set(Eigen3_INCLUDE_DIR ${LIBRARY_DIR}/eigen/ CACHE PATH "Path to Eigen source folder")
set(Ceres_DIR ${LIBRARY_DIR}/Ceres/CMake CACHE PATH "Path to Ceres build folder")
set(Ceres_INCLUDE_DIR ${Ceres_DIR}/../include)


# Use pre-compiled libraries on Windows
if (WIN32)
    set(FreeImage_INCLUDE_DIR ${LIBRARY_DIR}/FreeImage/x64/ CACHE PATH "Path to FreeImage header file")
    set(FreeImage_LIBRARY_DIR ${LIBRARY_DIR}/FreeImage/x64/ CACHE PATH "Path to FreeImage .lib/.dll folder")
endif (WIN32)

# Use 
if (MACOSX)
    find_package(Ceres REQUIRED)
    set(FreeImage_INCLUDE_DIR "/usr/local/Cellar/freeimage/3.18.0/include")
    set(FreeImage_LIBRARY_DIR "/usr/local/Cellar/freeimage/3.18.0/lib")
    set(Eigen3_INCLUDE_DIR "/usr/local/Cellar/eigen/3.4.0_1/include/eigen3")
    set(Flann_INCLUDE_DIR "/usr/local/Cellar/flann/1.9.1_11/include/flann")
    set(Flann_LIBRARY_DIR "/usr/local/Cellar/flann/1.9.1_11/lib")
endif (MACOSX)

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
        ${INCLUDE_DIR}/Ray.h
        ${INCLUDE_DIR}/NearestNeighbour.h
        ${INCLUDE_DIR}/ProcrustesAligner.h
        ${INCLUDE_DIR}/ICPOptimizer.h
        ${INCLUDE_DIR}/NearestNeighbour.h
        ${INCLUDE_DIR}/ProcrustesAligner.h )
set(SOURCES
        main.cpp
        ${HELPERS_DIR}/FreeImageHelper.cpp
        ${MODELS_DIR}/Frame.cpp
        ${MODELS_DIR}/Volume.cpp
        ${ICP_DIR}/ICP.cpp
        ${RAYCASTER_DIR}/RayCaster.cpp
        ${RAYCASTER_DIR}/Ray.cpp
        ${ICP_DIR}/ICPOptimizer.cpp
        ${ICP_DIR}/CeresICPOptimizer.cpp
        ${ICP_DIR}/LinearICPOptimizer.cpp)

link_directories(${FreeImage_LIBRARY_DIR} ${Flann_LIBRARY_DIR})
add_executable(kinect_fusion ${SOURCES} ${HEADERS})
target_include_directories(kinect_fusion PUBLIC ${Flann_INCLUDE_DIR} ${Eigen3_INCLUDE_DIR} ${FreeImage_INCLUDE_DIR} ${INCLUDE})
target_link_libraries(kinect_fusion general freeimage ceres flann)

# On Windows copy dll to executable folder
if (WIN32)
    add_custom_command(TARGET kinect_fusion POST_BUILD        # Adds a post-build event to MyTest
            COMMAND ${CMAKE_COMMAND} -E copy_if_different  # which executes "cmake - E copy_if_different..."
            "${LIBRARY_DIR}/FreeImage/x64/FreeImage.dll"      # <--this is in-file
            $<TARGET_FILE_DIR:kinect_fusion>)                 # <--this is out-file path
endif (WIN32)
