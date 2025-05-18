cmake_minimum_required(VERSION 3.18)
project(gedlibpy LANGUAGES CXX)

# Find Python and dependencies
find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)
find_package(Cython REQUIRED)
include(UseCython)

# Set Cython language level
set(CYTHON_LANGUAGE_LEVEL "3")

# Include directories
set(GEDLIB_INCLUDE_DIRS
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/lsape.5/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/eigen.3.3.4/Eigen"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/nomad.3.8.1/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/nomad.3.8.1/ext/sgtelib/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/libsvm.3.22"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/fann.2.2.0/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/boost.1.69.0"
    ${Python_NumPy_INCLUDE_DIRS}
)

# Library directories
set(GEDLIB_LIBRARY_DIRS
    "${CMAKE_CURRENT_SOURCE_DIR}/lib/fann.2.2.0"
    "${CMAKE_CURRENT_SOURCE_DIR}/lib/libsvm.3.22"
    "${CMAKE_CURRENT_SOURCE_DIR}/lib/nomad.3.8.1"
)

# Libraries to link
set(GEDLIB_LIBRARIES doublefann sgtelib svm nomad)

# Create the Cython extension module
add_cython_target(gedlibpy CXX)
add_library(gedlibpy MODULE ${gedlibpy})

# Set include directories for the module
target_include_directories(gedlibpy PUBLIC ${GEDLIB_INCLUDE_DIRS})

# Set library directories for the module
target_link_directories(gedlibpy PUBLIC ${GEDLIB_LIBRARY_DIRS})

# Link libraries
target_link_libraries(gedlibpy PUBLIC ${GEDLIB_LIBRARIES} Python::Python)

# Set C++ standard
target_compile_features(gedlibpy PRIVATE cxx_std_11)

# Install the Cython extension
install(TARGETS gedlibpy
        LIBRARY DESTINATION ${SKBUILD_PLATLIB_DIR}/gklearn/gedlib
        RUNTIME DESTINATION ${SKBUILD_PLATLIB_DIR}/gklearn/gedlib)