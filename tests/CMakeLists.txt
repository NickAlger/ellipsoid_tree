# SPDX-License-Identifier: MIT
add_executable(etree_tests
    test_main.cpp
    test_parallel_for.cpp
    test_minimize_scalar.cpp
    test_geometry.cpp
    test_intersections_closed_form.cpp
    test_intersections_ellipsoid.cpp
    test_intersections_simplex.cpp
    test_aabb_tree.cpp
    test_object_trees.cpp
    test_kd_tree.cpp
    test_batch_picker.cpp
    test_simplex_mesh.cpp
    test_lp_cells.cpp)
target_link_libraries(etree_tests PRIVATE etree::etree)
target_include_directories(etree_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty)
target_compile_options(etree_tests PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra>)
add_test(NAME etree_tests COMMAND etree_tests)
