file(GLOB_RECURSE src_SRC *.cpp)
file(GLOB_RECURSE src_CUDA *.cu)
file(GLOB_RECURSE src_HEADER *.hpp *.h)

set(SRC ${SRC} ${src_SRC} ${src_CUDA} PARENT_SCOPE)
set(HEADER ${HEADER} ${src_HEADER} PARENT_SCOPE)
