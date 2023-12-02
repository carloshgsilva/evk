cmake_minimum_required(VERSION 3.8)
project(evk VERSION 1.0.0 DESCRIPTION "Easy Vulkan")

set(EVK_DEBUG ON CACHE BOOL "Enables evk asserts for debugging")

if(EVK_DEBUG)
  add_compile_definitions(EVK_DEBUG)
endif()

# evk
add_library(evk "${CMAKE_CURRENT_SOURCE_DIR}/evk_vulkan.cpp")
target_include_directories(evk PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(evk PRIVATE cxx_std_20)

# vulkan
find_package(Vulkan REQUIRED)
target_link_libraries(evk PUBLIC Vulkan::Vulkan)
target_compile_definitions(evk PUBLIC "EVK_VULKAN=1")
