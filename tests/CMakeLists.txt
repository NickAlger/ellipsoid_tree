# SPDX-License-Identifier: MIT
add_executable(ellipsoid_tree_tests
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
    test_lp_cells.cpp
    test_plot2d.cpp
    test_plot2d_png.cpp)
target_link_libraries(ellipsoid_tree_tests PRIVATE ellipsoid_tree::ellipsoid_tree)
target_include_directories(ellipsoid_tree_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty)
target_compile_options(ellipsoid_tree_tests PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra>)
add_test(NAME ellipsoid_tree_tests COMMAND ellipsoid_tree_tests)
