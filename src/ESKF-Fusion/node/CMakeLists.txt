set(NODE_SOURCES 
run_eskf_gins
run_eskf_lio
run_ekf_fusion
run_ekf_fusion_node
)

foreach(node_src ${NODE_SOURCES})
  add_executable(${node_src} ${node_src}.cpp ${BACKWARD_ENABLE})
  target_link_libraries(${node_src} ${THIRD_PARTY_LIBS} ESKF_Fusion)
  add_backward(${node_src})
endforeach(node_src)
