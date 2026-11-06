cmake_minimum_required(VERSION 3.22)
project(cuERI VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(CUEST_ENABLE_CUDA "Build the CUDA backend when a CUDA toolkit is available" ON)
option(CUEST_ENABLE_OPENMP "Use OpenMP for the CPU backend" ON)
option(CUEST_BUILD_TESTS "Build the CTest suite" ON)
option(CUEST_BUILD_BENCH "Build the benchmark driver" ON)

# ---------------------------------------------------------------------------
# Hardware tier detection.  Tier is decided at configure time for the build
# configuration and at run time (cuestGetBackend) for the actual device.
#   Tier A: CUDA toolkit + usable GPU   Tier B: CUDA toolkit, no GPU
#   Tier C: no CUDA toolkit
# ---------------------------------------------------------------------------
include(CheckLanguage)
set(CUEST_CUDA_AVAILABLE OFF)
if(CUEST_ENABLE_CUDA)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CUEST_CUDA_AVAILABLE ON)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    message(STATUS "cuERI: CUDA toolkit found (${CMAKE_CUDA_COMPILER}); CUDA backend enabled")
  else()
    message(STATUS "cuERI: no CUDA toolkit found - building the CPU backend only (Tier C)")
    set(CUEST_ENABLE_CUDA OFF)
  endif()
else()
  message(STATUS "cuERI: CUDA disabled by option - building the CPU backend only (Tier C)")
endif()

# ---------------------------------------------------------------------------
# OpenMP (with an explicit Homebrew libomp fallback for Apple clang)
# ---------------------------------------------------------------------------
set(CUEST_OPENMP_FOUND OFF)
if(CUEST_ENABLE_OPENMP)
  find_package(OpenMP QUIET)
  if(OpenMP_CXX_FOUND)
    set(CUEST_OPENMP_FOUND ON)
    message(STATUS "cuERI: OpenMP found via find_package(OpenMP)")
  else()
    find_path(CUEST_LIBOMP_INCLUDE_DIR omp.h
              PATHS /opt/homebrew/opt/libomp/include /usr/local/opt/libomp/include)
    find_library(CUEST_LIBOMP_LIBRARY omp
                 PATHS /opt/homebrew/opt/libomp/lib /usr/local/opt/libomp/lib)
    if(CUEST_LIBOMP_INCLUDE_DIR AND CUEST_LIBOMP_LIBRARY)
      set(CUEST_OPENMP_FOUND ON)
      message(STATUS "cuERI: OpenMP via libomp at ${CUEST_LIBOMP_LIBRARY}")
    endif()
  endif()
  if(NOT CUEST_OPENMP_FOUND)
    message(STATUS "cuERI: OpenMP not found - CPU backend will run single-threaded")
  endif()
endif()

# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------
set(CUEST_SOURCES
    src/cuest_api.cpp
    src/basis_loader.cpp
    src/one_electron.cpp
    src/eri_cpu.cpp
    src/reference/ref_one_electron.cpp)

if(CUEST_CUDA_AVAILABLE)
  list(APPEND CUEST_SOURCES src/cuda/cuda_backend.cpp src/cuda/cuest_kernels.cu)
endif()

add_library(cuest_eri SHARED ${CUEST_SOURCES})
add_library(cuERI::cuest_eri ALIAS cuest_eri)
target_include_directories(cuest_eri
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_definitions(cuest_eri PRIVATE
    CUEST_DATA_DIR_DEFAULT="${CMAKE_CURRENT_SOURCE_DIR}/data")
target_compile_options(cuest_eri PRIVATE -Wall -Wextra)
set_target_properties(cuest_eri PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    OUTPUT_NAME cuest_eri)

if(CUEST_OPENMP_FOUND)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(cuest_eri PRIVATE OpenMP::OpenMP_CXX)
  else()
    target_compile_options(cuest_eri PRIVATE -Xpreprocessor -fopenmp
                           -I${CUEST_LIBOMP_INCLUDE_DIR})
    target_link_libraries(cuest_eri PRIVATE ${CUEST_LIBOMP_LIBRARY})
  endif()
  target_compile_definitions(cuest_eri PRIVATE CUEST_USE_OPENMP=1)
endif()

if(CUEST_CUDA_AVAILABLE)
  # The host-side CUDA backend (.cpp files) includes cuda_runtime.h and calls
  # the runtime API, so it needs the toolkit headers and cudart, not just nvcc.
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(cuest_eri PRIVATE CUDA::cudart)
  target_compile_definitions(cuest_eri PRIVATE CUEST_WITH_CUDA=1)
  set_target_properties(cuest_eri PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      # Shared library: run the device link here, otherwise every consumer
      # fails with an undefined __cudaRegisterLinkedBinary_* symbol.
      CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  target_compile_options(cuest_eri PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;-Xptxas;-v>)
  # Capture real ptxas register/spill output at build time (Tier A/B).
  add_custom_target(cuest_ptxas_verbose ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_SOURCE_DIR}/results
    COMMAND ${CMAKE_CUDA_COMPILER} -std=c++17 -lineinfo -Xptxas -v -c
            ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/cuest_kernels.cu
            -I${CMAKE_CURRENT_SOURCE_DIR}/src
            -I${CMAKE_CURRENT_SOURCE_DIR}/include
            -o ${CMAKE_CURRENT_BINARY_DIR}/cuest_kernels_ptxas.o
            2> ${CMAKE_CURRENT_SOURCE_DIR}/results/ptxas_verbose.txt
    COMMENT "Capturing ptxas -v register/spill report to results/ptxas_verbose.txt"
    VERBATIM)
endif()

# ---------------------------------------------------------------------------
# Tests / benchmarks / examples
# ---------------------------------------------------------------------------
if(CUEST_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
if(CUEST_BUILD_BENCH)
  add_subdirectory(bench)
endif()
add_subdirectory(examples)

# ---------------------------------------------------------------------------
# Install rules (a downstream CMake project uses find_package(cuERI))
# ---------------------------------------------------------------------------
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
install(TARGETS cuest_eri EXPORT cuERITargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/cuest DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY data/basis DESTINATION ${CMAKE_INSTALL_DATADIR}/cuest
        PATTERN "unsupported_l.json" EXCLUDE)
install(EXPORT cuERITargets NAMESPACE cuERI:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cuERI)
configure_package_config_file(cmake/cuERI.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/cuERIConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cuERI)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/cuERIConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/cuERIConfig.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/cuERIConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cuERI)

message(STATUS "cuERI: build type ${CMAKE_BUILD_TYPE}, CUDA ${CUEST_CUDA_AVAILABLE}, OpenMP ${CUEST_OPENMP_FOUND}")
