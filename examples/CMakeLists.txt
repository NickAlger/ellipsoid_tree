# SPDX-License-Identifier: MIT
set(ETREE_EXAMPLE_NAMES
    intersections_gallery
    point_cloud
    ellipsoid_tree_queries
    batch_picking
    mesh_queries
    mesh_mesh
    kdtree_knn
    cg1_field)

foreach(example_name ${ETREE_EXAMPLE_NAMES})
    add_executable(example_${example_name} ${example_name}.cpp)
    target_link_libraries(example_${example_name} PRIVATE etree::etree)
    set_target_properties(example_${example_name} PROPERTIES OUTPUT_NAME ${example_name})
    target_compile_options(example_${example_name} PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra>)
endforeach()
