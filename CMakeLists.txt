# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.16)
project(etree VERSION 0.1.0 LANGUAGES CXX)

add_library(etree INTERFACE)
add_library(etree::etree ALIAS etree)
target_compile_features(etree INTERFACE cxx_std_17)
target_include_directories(etree INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

find_package(Threads REQUIRED)
target_link_libraries(etree INTERFACE Threads::Threads)

# Eigen: use an installed copy if available, otherwise download a pinned release.
find_package(Eigen3 3.3 QUIET NO_MODULE)
if(NOT TARGET Eigen3::Eigen)
    option(ETREE_FETCH_EIGEN "Download Eigen if no installed copy is found" ON)
    if(NOT ETREE_FETCH_EIGEN)
        message(FATAL_ERROR
            "etree requires Eigen3. Install it (e.g. libeigen3-dev) or enable ETREE_FETCH_EIGEN.")
    endif()
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
        cmake_policy(SET CMP0135 NEW)
    endif()
    include(FetchContent)
    FetchContent_Declare(eigen3_src
        URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
        URL_HASH SHA256=8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72)
    FetchContent_GetProperties(eigen3_src)
    if(NOT eigen3_src_POPULATED)
        FetchContent_Populate(eigen3_src)
    endif()
    # Eigen is header-only: point an imported target at the extracted sources
    # instead of running Eigen's own build system.
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    set_target_properties(Eigen3::Eigen PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${eigen3_src_SOURCE_DIR}")
    message(STATUS "etree: using downloaded Eigen 3.4.0 (${eigen3_src_SOURCE_DIR})")
endif()
target_link_libraries(etree INTERFACE Eigen3::Eigen)

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    option(ETREE_BUILD_TESTS "Build etree tests" ON)
else()
    option(ETREE_BUILD_TESTS "Build etree tests" OFF)
endif()

if(ETREE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
