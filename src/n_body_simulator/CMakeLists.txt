add_executable(n_body_simulator n_body_simulator.cpp)

target_link_libraries(n_body_simulator vector3d)
target_include_directories(n_body_simulator PUBLIC ${CMAKE_SOURCE_DIR}/src/vector3d)