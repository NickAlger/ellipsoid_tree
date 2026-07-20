# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.16)

# Version is single-sourced from the public header (see include/ellipsoid_tree/ellipsoid_tree.hpp).
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/include/ellipsoid_tree/ellipsoid_tree.hpp" _ellipsoid_tree_hpp)
string(REGEX MATCH "ELLIPSOID_TREE_VERSION_MAJOR ([0-9]+)" _ "${_ellipsoid_tree_hpp}")
set(_ellipsoid_tree_major "${CMAKE_MATCH_1}")
string(REGEX MATCH "ELLIPSOID_TREE_VERSION_MINOR ([0-9]+)" _ "${_ellipsoid_tree_hpp}")
set(_ellipsoid_tree_minor "${CMAKE_MATCH_1}")
string(REGEX MATCH "ELLIPSOID_TREE_VERSION_PATCH ([0-9]+)" _ "${_ellipsoid_tree_hpp}")
set(_ellipsoid_tree_patch "${CMAKE_MATCH_1}")
project(ellipsoid_tree VERSION ${_ellipsoid_tree_major}.${_ellipsoid_tree_minor}.${_ellipsoid_tree_patch} LANGUAGES CXX)

add_library(ellipsoid_tree INTERFACE)
add_library(ellipsoid_tree::ellipsoid_tree ALIAS ellipsoid_tree)
target_compile_features(ellipsoid_tree INTERFACE cxx_std_17)
target_include_directories(ellipsoid_tree INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

find_package(Threads REQUIRED)
target_link_libraries(ellipsoid_tree INTERFACE Threads::Threads)

# Eigen: use an installed copy if available, otherwise download a pinned release.
find_package(Eigen3 3.3 QUIET NO_MODULE)
if(NOT TARGET Eigen3::Eigen)
    option(ELLIPSOID_TREE_FETCH_EIGEN "Download Eigen if no installed copy is found" ON)
    if(NOT ELLIPSOID_TREE_FETCH_EIGEN)
        message(FATAL_ERROR
            "ellipsoid_tree requires Eigen3. Install it (e.g. libeigen3-dev) or enable ELLIPSOID_TREE_FETCH_EIGEN.")
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
    message(STATUS "ellipsoid_tree: using downloaded Eigen 3.4.0 (${eigen3_src_SOURCE_DIR})")
endif()
target_link_libraries(ellipsoid_tree INTERFACE Eigen3::Eigen)

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    option(ELLIPSOID_TREE_BUILD_TESTS "Build ellipsoid_tree tests" ON)
    option(ELLIPSOID_TREE_BUILD_EXAMPLES "Build ellipsoid_tree examples" ON)
else()
    option(ELLIPSOID_TREE_BUILD_TESTS "Build ellipsoid_tree tests" OFF)
    option(ELLIPSOID_TREE_BUILD_EXAMPLES "Build ellipsoid_tree examples" OFF)
endif()

if(ELLIPSOID_TREE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(ELLIPSOID_TREE_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

option(ELLIPSOID_TREE_BUILD_PYTHON "Build the ellipsoid_tree Python bindings (pybind11 via FetchContent)" OFF)
if(ELLIPSOID_TREE_BUILD_PYTHON)
    add_subdirectory(bindings)
endif()

# ------------------------- Installation / CMake package export -------------------------
# Default ON when built standalone, OFF as a subproject and OFF inside pip
# builds (scikit-build-core would otherwise install the headers into the wheel).
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR AND NOT SKBUILD)
    set(ELLIPSOID_TREE_INSTALL_DEFAULT ON)
else()
    set(ELLIPSOID_TREE_INSTALL_DEFAULT OFF)
endif()
option(ELLIPSOID_TREE_INSTALL "Generate install and CMake package export rules" ${ELLIPSOID_TREE_INSTALL_DEFAULT})

if(ELLIPSOID_TREE_INSTALL)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)

    install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    install(TARGETS ellipsoid_tree EXPORT ellipsoid_treeTargets)
    install(EXPORT ellipsoid_treeTargets
        FILE ellipsoid_treeTargets.cmake
        NAMESPACE ellipsoid_tree::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ellipsoid_tree)

    configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/ellipsoid_treeConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/ellipsoid_treeConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ellipsoid_tree)
    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/ellipsoid_treeConfigVersion.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
        ARCH_INDEPENDENT)
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/ellipsoid_treeConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/ellipsoid_treeConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ellipsoid_tree)
    install(FILES LICENSE DESTINATION ${CMAKE_INSTALL_DOCDIR})
endif()
