CMAKE_MINIMUM_REQUIRED(VERSION 3.5)

PROJECT(rys-auto-pilot-algorithm)
SET(CMAKE_CXX_STANDARD 14)

INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR})

SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

ADD_SUBDIRECTORY(modules)
# ADD_SUBDIRECTORY(thirdparty)
