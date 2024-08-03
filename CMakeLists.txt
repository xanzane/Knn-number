cmake_minimum_required(VERSION 3.27)
project(Knn-number)

set(CMAKE_CXX_STANDARD 17)



set(OpenCV_DIR "D:\\Develop\\environment\\opencv\\mingw-build")


find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


add_executable(Knn-number main.cpp)
target_link_libraries(Knn-number ${OpenCV_LIBS})