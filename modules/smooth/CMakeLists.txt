CMAKE_MINIMUM_REQUIRED(VERSION 3.5) 
#设置项目名称
PROJECT(smooth)

set( CMAKE_BUILD_TYPE "Release" )

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OSQP library and headers
# Find OSQP library and headers
FIND_PACKAGE(osqp REQUIRED)
FIND_PACKAGE(Eigen3)
FIND_PACKAGE(PythonLibs 2.7)
find_package(OpenCV REQUIRED)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include)
INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIR})

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/modules)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/lib)

FILE(GLOB_RECURSE PROJECT_HEADERS  "include/*.h")
FILE(GLOB_RECURSE PROJECT_SOURCES  "src/*.cpp")

#生成可执行文件
ADD_EXECUTABLE(${PROJECT_NAME} main.cpp ${PROJECT_SOURCES}  ${PROJECT_HEADERS})

target_include_directories(${PROJECT_NAME} PRIVATE ${PYTHON_INCLUDE_DIRS})
# target_link_libraries(${PROJECT_NAME} ${PYTHON_LIBRARIES})

# Link the OSQP shared library
TARGET_LINK_LIBRARIES(${PROJECT_NAME} PRIVATE osqp::osqp ${PYTHON_LIBRARIES} ${OpenCV_LIBS})
