cmake_minimum_required(VERSION 3.15)
project(quantumfdtd CXX)

# Set C++ standard (C++17 required for OpenVDB)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Determine Homebrew prefix for OpenVDB on macOS
if(APPLE)
    execute_process(
        COMMAND brew --prefix openvdb
        OUTPUT_VARIABLE OPENVDB_BREW_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(OPENVDB_BREW_PREFIX)
        list(APPEND CMAKE_MODULE_PATH "${OPENVDB_BREW_PREFIX}/lib/cmake/OpenVDB")
        list(APPEND CMAKE_PREFIX_PATH "${OPENVDB_BREW_PREFIX}")
    endif()
endif()

# Find dependencies
find_package(MPI REQUIRED)
find_package(GSL REQUIRED)
find_package(OpenVDB REQUIRED)
find_package(TBB REQUIRED)

# --- Find FFTW3 Libraries & Headers ---
find_library(FFTW3_LIB NAMES fftw3 REQUIRED)
find_library(FFTW3_MPI_LIB NAMES fftw3_mpi REQUIRED)

find_path(FFTW3_INCLUDE_DIR 
    NAMES fftw3-mpi.h fftw3.h
    PATH_SUFFIXES include
    REQUIRED
)

message(STATUS "Found FFTW3 Include: ${FFTW3_INCLUDE_DIR}")
message(STATUS "Found FFTW3 Lib:     ${FFTW3_LIB}")
message(STATUS "Found FFTW3 MPI Lib: ${FFTW3_MPI_LIB}")

# Gather all source files
file(GLOB SOURCES "build/src/*.cpp")

# Define the executable once
add_executable(mpisolve ${SOURCES})

# Include directories
target_include_directories(mpisolve PRIVATE 
    build/include
    ${FFTW3_INCLUDE_DIR}
)

# Compiler optimization flags
target_compile_options(mpisolve PRIVATE 
    -O2 
    -funroll-loops 
    -finline-functions
)

# Link all dependencies
target_link_libraries(mpisolve PRIVATE
    OpenVDB::openvdb
    TBB::tbb
    MPI::MPI_CXX
    GSL::gsl
    GSL::gslcblas
    ${FFTW3_MPI_LIB}
    ${FFTW3_LIB}
)

# Replicate the `make run` target
add_custom_target(run
    COMMAND mpirun -np 8 --use-hwthread-cpus $<TARGET_FILE:mpisolve>
    DEPENDS mpisolve
    COMMENT "Running mpisolve with MPI..."
)