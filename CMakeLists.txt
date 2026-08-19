cmake_minimum_required(VERSION 3.24)

project(ac_rsvd LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(AC_RSVD_BUILD_TORCH "Build the PyTorch binding" ON)
option(AC_RSVD_BUILD_EXAMPLES "Build the C++ examples" ON)

set(BLA_SIZEOF_INTEGER 4)
find_package(BLAS QUIET)
if(BLAS_FOUND)
    set(AC_RSVD_BLAS_TARGET BLAS::BLAS)
else()
    find_file(AC_RSVD_BLAS_LIBRARY
        NAMES libopenblas.so.0
        PATHS /usr/lib /usr/lib64 /usr/local/lib
        PATH_SUFFIXES x86_64-linux-gnu aarch64-linux-gnu
        REQUIRED)
    add_library(ac_rsvd_blas UNKNOWN IMPORTED)
    set_target_properties(ac_rsvd_blas PROPERTIES
        IMPORTED_LOCATION "${AC_RSVD_BLAS_LIBRARY}")
    set(AC_RSVD_BLAS_TARGET ac_rsvd_blas)
endif()

find_package(LAPACK QUIET)
if(LAPACK_FOUND)
    set(AC_RSVD_LAPACK_TARGET LAPACK::LAPACK)
else()
    find_file(AC_RSVD_LAPACK_LIBRARY
        NAMES liblapack.so.3
        PATHS /usr/lib /usr/lib64 /usr/local/lib
        PATH_SUFFIXES x86_64-linux-gnu aarch64-linux-gnu
        REQUIRED)
    add_library(ac_rsvd_lapack UNKNOWN IMPORTED)
    set_target_properties(ac_rsvd_lapack PROPERTIES
        IMPORTED_LOCATION "${AC_RSVD_LAPACK_LIBRARY}")
    set(AC_RSVD_LAPACK_TARGET ac_rsvd_lapack)
endif()

file(GLOB_RECURSE AC_RSVD_CORE_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/algorithms/*.cpp")

add_library(ac_rsvd_core STATIC ${AC_RSVD_CORE_SOURCES})
set_target_properties(ac_rsvd_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(ac_rsvd_core
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src")

target_link_libraries(ac_rsvd_core PUBLIC
    ${AC_RSVD_LAPACK_TARGET}
    ${AC_RSVD_BLAS_TARGET})

target_compile_options(ac_rsvd_core PRIVATE
    -Wall
    -Wextra
    -Wpedantic)

if(AC_RSVD_BUILD_TORCH)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)

    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c
                "from torch.utils.cpp_extension import include_paths; print(';'.join(include_paths()))"
        OUTPUT_VARIABLE TORCH_INCLUDE_DIRS
        OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c
                "from torch.utils.cpp_extension import library_paths; print(library_paths()[0])"
        OUTPUT_VARIABLE TORCH_LIBRARY_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c
                "import torch; print(int(torch.compiled_with_cxx11_abi()))"
        OUTPUT_VARIABLE TORCH_CXX11_ABI
        OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY)

    find_library(TORCH_CPU_LIBRARY
        NAMES torch_cpu
        PATHS "${TORCH_LIBRARY_DIR}"
        NO_DEFAULT_PATH
        REQUIRED)
    find_library(TORCH_C10_LIBRARY
        NAMES c10
        PATHS "${TORCH_LIBRARY_DIR}"
        NO_DEFAULT_PATH
        REQUIRED)

    file(GLOB AC_RSVD_TORCH_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/bindings/torch/*.cpp")

    add_library(ac_rsvd_torch MODULE ${AC_RSVD_TORCH_SOURCES})
    target_include_directories(ac_rsvd_torch PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        ${TORCH_INCLUDE_DIRS})
    target_compile_definitions(ac_rsvd_torch PRIVATE
        _GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI})
    target_link_libraries(ac_rsvd_torch PRIVATE
        ac_rsvd_core
        "${TORCH_CPU_LIBRARY}"
        "${TORCH_C10_LIBRARY}")
    set_target_properties(ac_rsvd_torch PROPERTIES
        PREFIX ""
        BUILD_RPATH "${TORCH_LIBRARY_DIR}")
endif()

if(AC_RSVD_BUILD_EXAMPLES AND
   EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/cpp/CMakeLists.txt")
    add_subdirectory(examples/cpp)
endif()

include(CTest)
if(BUILD_TESTING AND
   EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp/CMakeLists.txt")
    add_subdirectory(tests/cpp)
    if(AC_RSVD_BUILD_TORCH AND
       EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/python/CMakeLists.txt")
        add_subdirectory(tests/python)
    endif()
endif()
