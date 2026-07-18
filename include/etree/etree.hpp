#pragma once
// SPDX-License-Identifier: MIT
// etree — ellipsoid intersection tests and spatial trees.
// Header-only C++17; depends only on Eigen. https://github.com/NickAlger/ellipsoid_tree

#define ETREE_VERSION_MAJOR 0
#define ETREE_VERSION_MINOR 1
#define ETREE_VERSION_PATCH 0

// Umbrella header. Public headers are included here as they land:
#include "etree/geometry.hpp"       // Box, Ball, Ellipsoid, Simplex (+ Segment, Halfspace)
#include "etree/intersections.hpp"  // pairwise intersects(A, B [, tau]) overloads
#include "etree/aabb_tree.hpp"      // box-tree core: visit() and visit_pairs()
#include "etree/object_tree.hpp"    // BoxTree, BallTree, EllipsoidTree, SimplexTree
#include "etree/geometric_sort.hpp" // axis-alternating geometric ordering
#include "etree/kd_tree.hpp"        // k-nearest-neighbor queries
#include "etree/simplex_mesh.hpp"   // SimplexMesh: point location, closest point, CG1 eval
#include "etree/batch_picker.hpp"   // greedy non-overlapping ellipsoid batches
