#pragma once
// SPDX-License-Identifier: MIT

/// @file
/// @brief Umbrella header — includes the whole ellipsoid_tree public API.
///
/// ellipsoid_tree: ellipsoid intersection tests and spatial trees. Header-only C++17;
/// depends only on Eigen. https://github.com/NickAlger/ellipsoid_tree

// Single source of truth for the version. CMakeLists.txt parses these macros to
// set the project version, and a CI check keeps pyproject.toml / CITATION.cff in
// sync; ELLIPSOID_TREE_VERSION is the composed "MAJOR.MINOR.PATCH" string.
#define ELLIPSOID_TREE_VERSION_MAJOR 0
#define ELLIPSOID_TREE_VERSION_MINOR 2
#define ELLIPSOID_TREE_VERSION_PATCH 0
#define ELLIPSOID_TREE_STRINGIZE_IMPL(x) #x
#define ELLIPSOID_TREE_STRINGIZE(x)      ELLIPSOID_TREE_STRINGIZE_IMPL(x)
#define ELLIPSOID_TREE_VERSION                        \
    ELLIPSOID_TREE_STRINGIZE(ELLIPSOID_TREE_VERSION_MAJOR) "." \
    ELLIPSOID_TREE_STRINGIZE(ELLIPSOID_TREE_VERSION_MINOR) "." \
    ELLIPSOID_TREE_STRINGIZE(ELLIPSOID_TREE_VERSION_PATCH)

// Umbrella header. Public headers are included here as they land:
#include "ellipsoid_tree/geometry.hpp"       // Box, Ball, Ellipsoid, Simplex (+ Segment, Halfspace)
#include "ellipsoid_tree/intersections.hpp"  // pairwise intersects(A, B [, tau]) overloads
#include "ellipsoid_tree/aabb_tree.hpp"      // box-tree core: visit() and visit_pairs()
#include "ellipsoid_tree/object_tree.hpp"    // BoxTree, BallTree, EllipsoidTree, SimplexTree
#include "ellipsoid_tree/geometric_sort.hpp" // axis-alternating geometric ordering
#include "ellipsoid_tree/kd_tree.hpp"        // k-nearest-neighbor queries
#include "ellipsoid_tree/simplex_mesh.hpp"   // SimplexMesh: point location, closest point, CG1 eval
#include "ellipsoid_tree/batch_picker.hpp"   // greedy non-overlapping ellipsoid batches
