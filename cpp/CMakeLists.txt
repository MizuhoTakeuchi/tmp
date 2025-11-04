cmake_minimum_required(VERSION 3.12)
project(map_composer_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PCL 1.8 REQUIRED COMPONENTS io common registration)

add_library(map_composer
    src/map_composer.cpp
)
target_include_directories(map_composer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${PCL_INCLUDE_DIRS}
)
target_link_libraries(map_composer PUBLIC ${PCL_LIBRARIES})
add_definitions(${PCL_DEFINITIONS})

add_executable(map_composer_cli src/main.cpp)
target_link_libraries(map_composer_cli PRIVATE map_composer ${PCL_LIBRARIES})
target_include_directories(map_composer_cli PRIVATE ${PCL_INCLUDE_DIRS})

