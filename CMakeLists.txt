cmake_minimum_required(VERSION 3.18)
project(graphkit_learn LANGUAGES CXX)

# Set Python version requirements
find_package(Python REQUIRED COMPONENTS Interpreter Development)

# Add the gedlib subdirectory if it exists
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/gklearn/gedlib/CMakeLists.txt")
    add_subdirectory(gklearn/gedlib)
endif()

# Install Python packages
install(DIRECTORY gklearn 
        DESTINATION ${SKBUILD_PLATLIB_DIR}
        PATTERN "*.egg-info" EXCLUDE
        PATTERN "*.git*" EXCLUDE
        PATTERN "__pycache__" EXCLUDE
        PATTERN "*.cpp" EXCLUDE
        PATTERN "*.o" EXCLUDE
        PATTERN "*.so" EXCLUDE
        PATTERN "*.pyc" EXCLUDE)