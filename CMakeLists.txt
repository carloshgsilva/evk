cmake_minimum_required(VERSION 3.8)
project(evk VERSION 1.0.0 DESCRIPTION "Easy Vulkan")

set(EVK_DEBUG ON CACHE BOOL "Enables evk asserts for debugging")

if(EVK_DEBUG)
  add_compile_definitions(EVK_DEBUG)
endif()

add_library(evk $<IF:$<PLATFORM_ID:Darwin>,
  "${CMAKE_CURRENT_SOURCE_DIR}/evk_metal.cpp",
  "${CMAKE_CURRENT_SOURCE_DIR}/evk_vulkan.cpp"
>)
target_include_directories(evk PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(evk PRIVATE cxx_std_20)

if(APPLE)
  # metal
  target_link_libraries(evk "-framework AppKit" "-framework QuartzCore" "-framework Metal")
  target_compile_definitions(evk PUBLIC "EVK_METAL=1")
  target_include_directories(evk PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/metal-cpp)
  # brew install shaderc spirv-cross
elseif(WIN32 OR UNIX)
  # vulkan
  find_package(Vulkan REQUIRED)
  target_link_libraries(evk PUBLIC Vulkan::Vulkan)
  target_compile_definitions(evk PUBLIC "EVK_VULKAN=1")
endif()