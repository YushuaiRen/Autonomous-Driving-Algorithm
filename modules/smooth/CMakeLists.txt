CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12) 
#设置项目名称
PROJECT(smooth)

# Find OSQP library and headers
find_package(osqp REQUIRED)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/modules)

FILE(GLOB_RECURSE PROJECT_HEADERS  "include/*.h")
FILE(GLOB_RECURSE PROJECT_SOURCES  "src/*.cpp")

#生成可执行文件
ADD_EXECUTABLE(${PROJECT_NAME} main.cpp ${PROJECT_SOURCES}  ${PROJECT_HEADERS})

# Link the OSQP shared library
target_link_libraries(${PROJECT_NAME} PRIVATE osqp::osqp)
