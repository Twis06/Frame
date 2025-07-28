set(TEST_SOURCES 
test_base
test_jacobian
)

foreach(test_src ${TEST_SOURCES})
  add_executable(${test_src} ${test_src}.cpp ${BACKWARD_ENABLE})
  target_link_libraries(${test_src} ${THIRD_PARTY_LIBS})
  add_backward(${test_src})
endforeach(test_src)
