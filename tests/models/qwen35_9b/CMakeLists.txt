# Qwen3.5-9B model checks; generic Qwen operator controls stay in qwen/.
function(add_qwen35_9b_test target source labels)
  add_executable(${target} ${CMAKE_CURRENT_SOURCE_DIR}/${source})
  target_link_libraries(${target} PRIVATE gufo_core)
  add_test(NAME ${target} COMMAND ${target})
  set_tests_properties(${target} PROPERTIES LABELS "qwen35_9b;${labels}"
    SKIP_RETURN_CODE 77)
endfunction()

if (ENGINE_ENABLE_HIP)
  add_qwen35_9b_test(qwen35_9b_target_test target_test.cpp
    "gpu;hip;target;external-model")
endif()
