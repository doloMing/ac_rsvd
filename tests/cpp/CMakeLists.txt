function(add_ac_rsvd_test name)
    add_executable(${name} ${name}.cpp)
    target_include_directories(${name} PRIVATE
        "${PROJECT_SOURCE_DIR}/src")
    target_link_libraries(${name} PRIVATE ac_rsvd_core)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

add_ac_rsvd_test(test_linalg)
add_ac_rsvd_test(test_random_certificate)
add_ac_rsvd_test(test_baselines)
add_ac_rsvd_test(test_ac_rsvd)
