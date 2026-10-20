# SimpleCNN 은 헤더만 있는 라이브러리다. 이 파일은 예제/테스트 빌드용 (Linux, macOS, Windows 공통)
#   cmake -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build
#   ./build/MnistTest cuda
cmake_minimum_required(VERSION 3.16)
project(SimpleCNN CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	set(CMAKE_BUILD_TYPE Release)
endif()

add_library(SimpleCNN INTERFACE)
target_include_directories(SimpleCNN INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(SimpleCNN INTERFACE ${CMAKE_DL_LIBS})
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
	target_link_libraries(SimpleCNN INTERFACE OpenMP::OpenMP_CXX)
endif()

add_executable(MnistTest examples/MnistTest.cpp)
target_link_libraries(MnistTest PRIVATE SimpleCNN)

add_executable(UnitTest tests/UnitTest.cpp)
target_link_libraries(UnitTest PRIVATE SimpleCNN)

enable_testing()
add_test(NAME UnitTest COMMAND UnitTest)
