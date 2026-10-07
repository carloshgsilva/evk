function(evk_embed_shaders target shader_namespace)
  if(NOT Vulkan_GLSLC_EXECUTABLE)
    message(FATAL_ERROR
        "${target} requires glslc. Install the Vulkan SDK or set Vulkan_GLSLC_EXECUTABLE.")
  endif()

  set(generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated/${target}")
  set(registry_cpp "${generated_dir}/${target}_shaders.cpp")
  file(MAKE_DIRECTORY "${generated_dir}")

  set(registry_content
      "#include <cstddef>\n#include <cstdint>\n#include <stdexcept>\n#include <string>\n#include <string_view>\n#include <vector>\n\nnamespace ${shader_namespace} {\nnamespace {\n\ntemplate <std::size_t N>\nstd::vector<uint8_t> shader_bytes(const uint32_t (&words)[N]) {\n    const auto* first = reinterpret_cast<const uint8_t*>(words);\n    return {first, first + sizeof(words)};\n}\n\n")
  set(shader_outputs)
  set(shader_lookup)
  set(shader_definitions)
  if(EVK_BACKEND STREQUAL "Vulkan")
    list(APPEND shader_definitions -DEVK_VULKAN=1)
  endif()

  foreach(shader IN LISTS ARGN)
    get_filename_component(shader_name "${shader}" NAME_WE)
    string(MAKE_C_IDENTIFIER "${shader_name}" shader_symbol)
    set(shader_output "${generated_dir}/${shader_name}.inc")
    set(shader_depfile "${shader_output}.d")

    add_custom_command(
      OUTPUT "${shader_output}"
      COMMAND "${Vulkan_GLSLC_EXECUTABLE}"
          "${shader}"
          -std=460
          --target-env=vulkan1.3
          ${shader_definitions}
          -O
          -mfmt=c
          -MD
          -MF "${shader_depfile}"
          -MT "${shader_output}"
          -o "${shader_output}"
      DEPENDS "${shader}"
      DEPFILE "${shader_depfile}"
      VERBATIM
    )

    list(APPEND shader_outputs "${shader_output}")
    string(APPEND registry_content
        "alignas(uint32_t) const uint32_t shader_${shader_symbol}[] =\n#include \"${shader_output}\"\n;\n\n")
    string(APPEND shader_lookup
        "    if (name == \"${shader_name}\") return shader_bytes(shader_${shader_symbol});\n")
  endforeach()

  string(APPEND registry_content
      "} // namespace\n\nstd::vector<uint8_t> load_embedded_shader(std::string_view name) {\n${shader_lookup}    throw std::runtime_error(\"unknown embedded EVK shader: \" + std::string(name));\n}\n\n} // namespace ${shader_namespace}\n")

  file(GENERATE OUTPUT "${registry_cpp}" CONTENT "${registry_content}")
  set_source_files_properties(${shader_outputs}
      PROPERTIES GENERATED TRUE HEADER_FILE_ONLY TRUE)
  target_sources(${target} PRIVATE "${registry_cpp}" ${shader_outputs})
endfunction()
