include_directories(${AlgorithmInc})
add_executable(Sample1 main.cpp)
set_target_properties(${OpenCV_LIBS} PROPERTIES MAP_IMPORTED_CONFIG_RELWITHDEBINFO RELEASE)
target_link_libraries(Sample1 ${OpenCV_LIBS} AlgorithmLib)