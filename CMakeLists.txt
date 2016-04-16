cmake_minimum_required(VERSION 3.3)
project(face)
#//查找库
find_package(OpenCV)

#//包含头文件
include_directories(${OpenCV_INCLUDE_DIRS})


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp)
add_executable(face ${SOURCE_FILES})

#//包含链接文件
target_link_libraries(face ${OpenCV_LIBS})