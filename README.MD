## 基于knn算法识别手写数字

- 图片说明：32*32px @300ppi

- 使用说明

  本程序需要用到opencv库，所以在本地需要opencv库，并配置

- CMakeLists.txt配置

  ```cmake
  cmake_minimum_required(VERSION 3.27)
  project(Knn-number)
  
  set(CMAKE_CXX_STANDARD 17)
  
  
  # 路径替换到本地opencv库
  set(OpenCV_DIR "D:\\Develop\\environment\\opencv\\mingw-build")
  
  
  find_package(OpenCV REQUIRED)
  include_directories(${OpenCV_INCLUDE_DIRS})
  
  
  add_executable(Knn-number main.cpp)
  target_link_libraries(Knn-number ${OpenCV_LIBS})
  ```
  
  
  
  

