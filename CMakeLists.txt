cmake_minimum_required(VERSION 3.21)
project(evk VERSION 1.0.0 DESCRIPTION "Easy Vulkan")

set(EVK_DEBUG ON CACHE BOOL "Enables evk asserts for debugging")

if(EVK_DEBUG)
  add_compile_definitions(EVK_DEBUG)
endif()

# Option to control whether the example application is built
option(EVK_BUILD_EXAMPLE "Build the evk example application" OFF)

set(EVK_BACKEND "Vulkan" CACHE STRING "Graphics backend: Vulkan or Metal")
set_property(CACHE EVK_BACKEND PROPERTY STRINGS Vulkan Metal)
if(EVK_BACKEND STREQUAL "Metal")
  if(NOT APPLE)
    message(FATAL_ERROR "The native Metal backend requires macOS")
  endif()
  enable_language(OBJCXX)
  add_library(evk evk_metal.mm)
  set_target_properties(evk PROPERTIES OBJCXX_STANDARD 20 OBJCXX_STANDARD_REQUIRED YES)
  target_compile_options(evk PRIVATE -fobjc-arc)
  target_compile_definitions(evk PUBLIC EVK_METAL=1)
  target_link_libraries(evk PUBLIC "-framework Metal" "-framework Foundation" "-framework QuartzCore" "-framework AppKit")
  find_program(Vulkan_GLSLC_EXECUTABLE glslc HINTS "$ENV{VULKAN_SDK}/bin" REQUIRED)
elseif(EVK_BACKEND STREQUAL "Vulkan")
  find_package(Vulkan REQUIRED)
  add_library(evk evk_vulkan.cpp)
  target_link_libraries(evk PUBLIC Vulkan::Vulkan)
  target_compile_definitions(evk PUBLIC EVK_VULKAN=1)
else()
  message(FATAL_ERROR "Unknown EVK_BACKEND: ${EVK_BACKEND}")
endif()
target_include_directories(evk PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(evk PRIVATE cxx_std_20)
add_library(evk::evk ALIAS evk)

option(EVK_BUILD_METAL_SHADER_TOOL "Build the offline SPIR-V to Metal shader translator" OFF)
if(EVK_BUILD_METAL_SHADER_TOOL)
  find_path(EVK_SPIRV_CROSS_INCLUDE spirv_msl.hpp
      HINTS "$ENV{VULKAN_SDK}/include/spirv_cross" REQUIRED)
  foreach(component msl glsl core)
    find_library(EVK_SPIRV_CROSS_${component} NAMES spirv-cross-${component}
        HINTS "$ENV{VULKAN_SDK}/lib" REQUIRED)
  endforeach()
  add_executable(evk_metal_shader tools/metal_shader.cpp)
  target_compile_features(evk_metal_shader PRIVATE cxx_std_20)
  target_include_directories(evk_metal_shader PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${EVK_SPIRV_CROSS_INCLUDE})
  target_link_libraries(evk_metal_shader PRIVATE
      ${EVK_SPIRV_CROSS_msl} ${EVK_SPIRV_CROSS_glsl} ${EVK_SPIRV_CROSS_core})
endif()

# evk::ai
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/evk_embed_shaders.cmake")

file(GLOB EVK_AI_SHADERS CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/ai/*.comp"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/oidn/*.comp")

add_library(evk_ai EXCLUDE_FROM_ALL
    "${CMAKE_CURRENT_SOURCE_DIR}/evk_ai.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/evk_ai_ir.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/evk_ai_image_compiler.cpp")
target_include_directories(evk_ai PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(evk_ai PUBLIC cxx_std_20)
target_link_libraries(evk_ai PUBLIC evk)
evk_embed_shaders(evk_ai "evk::ai::detail" ${EVK_AI_SHADERS})
add_library(evk::ai ALIAS evk_ai)

add_library(evk_oidn EXCLUDE_FROM_ALL "${CMAKE_CURRENT_SOURCE_DIR}/evk_ai_oidn.cpp")
target_include_directories(evk_oidn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(evk_oidn PUBLIC cxx_std_20)
target_link_libraries(evk_oidn PUBLIC evk_ai)
add_library(evk::oidn ALIAS evk_oidn)

if(EVK_BUILD_EXAMPLE)
  if(EVK_BACKEND STREQUAL "Metal")
    message(FATAL_ERROR "The Vulkan-specific evk example has not been ported to Metal")
  endif()
  add_executable(evk_example
      "${CMAKE_CURRENT_SOURCE_DIR}/example/main.cpp"
      "${CMAKE_CURRENT_SOURCE_DIR}/example/main_llm.cpp"
      "${CMAKE_CURRENT_SOURCE_DIR}/example/oidn_demo.cpp"
  )
  target_link_libraries(evk_example PRIVATE evk::oidn)
  set_target_properties(evk_example PROPERTIES CXX_STANDARD 20)
else()
  message(STATUS "EVK_BUILD_EXAMPLE is OFF - example will not be built")
endif()
