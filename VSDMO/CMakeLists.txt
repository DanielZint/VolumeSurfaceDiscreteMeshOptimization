# CMake 3.8 is nescessary for setting up the working directory
cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(VSDMO)

#cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
set(VSDMO VSDMO)
project(${VSDMO} LANGUAGES CXX CUDA)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")
set(LIBRARY_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/lib")

find_package(CUDA REQUIRED)
# Find GLM
find_package(GLM REQUIRED)
include_directories(${GLM_INCLUDE_DIR})

# Find CUDA
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# Find OpenMesh
find_package(OpenMesh REQUIRED)
include_directories(${OPENMESH_INCLUDE_DIRS})
set(LIBS ${LIBS} ${OPENMESH_LIBRARIES})
add_definitions(-D_USE_MATH_DEFINES)

# Find OpenVolumeMesh
#find_package(OpenVolumeMesh REQUIRED)
#include_directories(${OPENVOLUMEMESH_INCLUDE_DIR})
#set(LIBS ${LIBS} ${OPENVOLUMEMESH_LIBRARY})
#add_definitions(-D_USE_MATH_DEFINES)

# Find SDL2
find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIR})
set(LIBS ${LIBS} ${SDL2_LIBRARY})

# Find OpenGL
find_package(OpenGL REQUIRED)
include_directories(${OPENGL_INCLUDE_DIR})
set(LIBS ${LIBS} ${OPENGL_LIBRARIES})

# Find GLEW
find_package(GLEW REQUIRED)
include_directories(${GLEW_INCLUDE_DIRS})
set(LIBS ${LIBS} ${GLEW_LIBRARIES})

set(LIBS ${LIBS} cublas)
set(LIBS ${LIBS} cusparse)

include_directories(vendor/include/cub)

#find_package(Filesystem REQUIRED)
#set(LIBS ${LIBS} stdc++fs)

# Find Eigen
#find_package(Eigen3 REQUIRED)
#include_directories(${EIGEN3_INCLUDE_DIR})

# add files in src
add_subdirectory(src)
include_directories(src)

# group files in folder structure
macro(GroupSources curdir)
    file(GLOB children RELATIVE ${PROJECT_SOURCE_DIR}/${curdir} ${PROJECT_SOURCE_DIR}/${curdir}/*)
    foreach(child ${children})
        if(IS_DIRECTORY ${PROJECT_SOURCE_DIR}/${curdir}/${child})
            GroupSources(${curdir}/${child})
        else()
            string(REPLACE "/" "\\" groupname ${curdir})
            source_group(${groupname} FILES ${PROJECT_SOURCE_DIR}/${curdir}/${child})
        endif()
    endforeach()
endmacro()

GroupSources(src)


####
# From AGPhys CMakeLists
####
OPTION(CUDA_COMPATIBILITY_MODE "Add CUDA compile flags that make the code compatible with older GPUs but takes longer to compile" OFF)
OPTION(CUDA_PROFILING "Enables the PROFILING #define" OFF)
OPTION(CUDA_DEBUG "Enables CUDA Debugging in Nsight" OFF)
OPTION(THRUST_DEBUG "Test" OFF)
OPTION(FULL_SPEED "Enables All Optimizations" OFF)

if (CUDA_COMPATIBILITY_MODE) 
	message("Compatibility Mode: -gencode=arch=compute_30,code=compute_30")
	SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_30,code=compute_30")
else()
	message("Compatibility Mode Off")
endif()

if(CUDA_PROFILING)
        add_definitions(-DCUDA_PROFILING)
        SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo --ptxas-options=-v")
        SET(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

if(CUDA_DEBUG)
        SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")
        SET(CMAKE_BUILD_TYPE DEBUG)
endif()

if(THRUST_DEBUG)
        SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DTHRUST_DEBUG")
        SET(CMAKE_BUILD_TYPE DEBUG)
endif()

if(FULL_SPEED)
        add_definitions(-DCUDA_NDEBUG)
        SET(CMAKE_BUILD_TYPE RELEASE)
    else()
        add_definitions(-DCUDA_DEBUG)
endif()


#set(CMAKE_CUDA_SEPARABLE_COMPILATION ON) # compile error
#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
#SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler --extended-lambda") # does not compile without the --extended-lambda option, although it is unkown option???

SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_52,code=sm_52") # Maxwell

#if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL "9")
#    SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_61,code=sm_61") # Pascal
#endif()

#SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_70,code=sm_70") # Volta

#if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL "10")
#	#SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_75,code=sm_75") # Turing
#endif()

#SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math --expt-relaxed-constexpr --expt-extended-lambda") # doesnt work with -G ??






SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored --ptxas-options -v")

message(STATUS "CUDA FLAGS: ${CMAKE_CUDA_FLAGS}")
#SET(CMAKE_LD_FLAGS "${CMAKE_LD_FLAGS} -lcusparsea")


if(MSVC)
	#multiprocessor compilation for visual studio
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP") 
else()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-strict-aliasing")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sign-compare")
        if(FULL_SPEED)
            SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
        endif()
endif()

####
# end AGPhys
####




# OpenMP
find_package(OpenMP REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda") #TODO

# create executable and link libraries
add_executable(${VSDMO} ${SRC} ${HEADER})
target_link_libraries(${VSDMO} ${LIBS})


# Set OpenMeshExample as startup project in visual studio
if(MSVC)
    add_definitions(-DWINDOWS)
    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${VSDMO})
    set_property(TARGET ${VSDMO} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
endif()

# copy dll-files after building into build directory
if(MSVC)
    add_custom_command(TARGET ${VSDMO} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${PROJECT_SOURCE_DIR}/vendor/bin/" $<TARGET_FILE_DIR:${VSDMO}>)
endif()