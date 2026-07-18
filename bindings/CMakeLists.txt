# SPDX-License-Identifier: MIT
set(PYBIND11_FINDPYTHON ON)
include(FetchContent)
FetchContent_Declare(pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.13.6
    GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(etree_python etree_bindings.cpp)
target_link_libraries(etree_python PRIVATE etree::etree)
set_target_properties(etree_python PROPERTIES OUTPUT_NAME etree)
