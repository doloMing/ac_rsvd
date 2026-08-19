add_test(
    NAME test_torch_binding
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/test_torch_binding.py"
            "$<TARGET_FILE:ac_rsvd_torch>")
