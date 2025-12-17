cmake_minimum_required(VERSION 3.10)
project(evk VERSION 1.0.0 DESCRIPTION "Easy Vulkan")

set(EVK_DEBUG ON CACHE BOOL "Enables evk asserts for debugging")

if(EVK_DEBUG)
  add_compile_definitions(EVK_DEBUG)
endif()

# Option to control whether the example application is built
option(EVK_BUILD_EXAMPLE "Build the evk example application" OFF)

# evk
add_library(evk "${CMAKE_CURRENT_SOURCE_DIR}/evk_vulkan.cpp")
target_include_directories(evk PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(evk PRIVATE cxx_std_20)
# vulkan
find_package(Vulkan REQUIRED)
target_link_libraries(evk PUBLIC Vulkan::Vulkan)
target_compile_definitions(evk PUBLIC "EVK_VULKAN=1")

if(EVK_BUILD_EXAMPLE)
  add_executable(evk_example 
      "${CMAKE_CURRENT_SOURCE_DIR}/example/main.cpp"
      "${CMAKE_CURRENT_SOURCE_DIR}/example/main_llm.cpp"
      "${CMAKE_CURRENT_SOURCE_DIR}/evk_ai.cpp"
  )
  target_link_libraries(evk_example PRIVATE evk)
  set_target_properties(evk_example PROPERTIES CXX_STANDARD 20)
else()
  message(STATUS "EVK_BUILD_EXAMPLE is OFF - example will not be built")
endif()