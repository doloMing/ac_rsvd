add_executable(ac_rsvd_dense_example dense_example.cpp)
target_include_directories(ac_rsvd_dense_example PRIVATE
    "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(ac_rsvd_dense_example PRIVATE ac_rsvd_core)
