# Copyright (c) 2018 Intel Corporation

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#      http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
cmake_minimum_required(VERSION 3.3)

# Common to all tutorial sections
include(CMakeLists_common.cmake)

set(TARGET_NAME "driver_behavior")

option(SIMULATOR  "Use ets/ats for truck data simulation" OFF)

add_definitions(-DOPENVINO_VER=$ENV{OPENVINO_VER})
message(STATUS "OpenVINO_Version=$ENV{OPENVINO_VER}")

find_package(OpenCV)
if(OpenCV_FOUND)
    message(STATUS "OPENCV is enabled")
    message(STATUS "OpenCV_INCLUDE_DIRS=${OpenCV_INCLUDE_DIRS}")
    message(STATUS "OpenCV_LIBS=${OpenCV_LIBS}")
else()
    message(STATUS "OPENCV is disabled or not found, " ${TARGET_NAME} " skiped")
    return()
endif()

if ( SIMULATOR )
find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rcutils)
find_package(rosidl_typesupport_cpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(ets_msgs REQUIRED)
find_package(Boost COMPONENTS log system filesystem REQUIRED)

ament_export_dependencies(rosidl_default_runtime)
endif()

if( BUILD_SAMPLE_NAME AND NOT ${BUILD_SAMPLE_NAME} STREQUAL ${TARGET_NAME} )
    message(STATUS "SAMPLE ${TARGET_NAME} SKIPPED")
    return()
endif()

file (GLOB MAIN_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c
        )

file (GLOB MAIN_HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp 
	${CMAKE_CURRENT_SOURCE_DIR}/src/*.h
        )

link_directories(${CMAKE_SOURCE_DIR}/lib)

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj
source_group("src" FILES ${MAIN_SRC})
source_group("include" FILES ${MAIN_HEADERS})

include_directories (${OpenCV_INCLUDE_DIRS})
include_directories (${Boost_INCLUDE_DIRS})


# Create library file from sources.
add_executable(${TARGET_NAME} ${MAIN_SRC} ${MAIN_HEADERS})

add_dependencies(${TARGET_NAME} gflags)
add_definitions(-DBOOST_LOG_DYN_LINK)
if( SIMULATOR )
  add_definitions(-DSIMULATOR)
endif()

set_target_properties(${TARGET_NAME} PROPERTIES "CMAKE_CXX_FLAGS" "${CMAKE_CXX_FLAGS} -fPIE" COMPILE_PDB_NAME ${TARGET_NAME})

if ("$ENV{OPENVINO_VER}" STREQUAL "2019")
    target_link_libraries(${TARGET_NAME} format_reader IE::ie_cpu_extension ${IE_LIBRARIES} ${InferenceEngine_LIBRARIES} gflags)
else ()
    target_link_libraries(${TARGET_NAME} format_reader ${IE_LIBRARIES} ${InferenceEngine_LIBRARIES} gflags)
endif()

target_include_directories(${PROJECT_NAME} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

if(UNIX)
	target_link_libraries( ${TARGET_NAME} ${LIB_DL} pthread ${OpenCV_LIBRARIES} ${Boost_LIBRARIES} dlib::dlib ao sndfile gomp)
endif()

add_custom_command(TARGET driver_behavior POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy
                   ${CMAKE_CURRENT_SOURCE_DIR}/data/shape_predictor_68_face_landmarks.dat $<TARGET_FILE_DIR:driver_behavior>)

add_custom_command(TARGET driver_behavior POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy
                   ${CMAKE_CURRENT_SOURCE_DIR}/data/beep.ogg $<TARGET_FILE_DIR:driver_behavior>)

add_custom_command(TARGET driver_behavior POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy
                   ${CMAKE_CURRENT_SOURCE_DIR}/scripts/faces_gallery.json $<TARGET_FILE_DIR:driver_behavior>)

if ( SIMULATOR )
install(TARGETS driver_behavior
    DESTINATION bin)
install(FILES data/beep.ogg data/shape_predictor_68_face_landmarks.dat scripts/faces_gallery.json DESTINATION bin)

ament_target_dependencies(driver_behavior
    "rclcpp"
    "rcutils"
    "rosidl_typesupport_cpp"
    "std_msgs"
    "ets_msgs"
    "OpenCV")

ament_package()
endif()

