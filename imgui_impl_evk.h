#pragma once
#include "imgui.h"      // IMGUI_IMPL_API
#include "evk.h"

IMGUI_IMPL_API bool     ImGui_ImplEvk_Init(evk::Cmd& cmd);
IMGUI_IMPL_API void     ImGui_ImplEvk_Shutdown();
IMGUI_IMPL_API void     ImGui_ImplEvk_PrepareRender(ImDrawData* draw_data, evk::Cmd& cmd);
IMGUI_IMPL_API void     ImGui_ImplEvk_RenderDrawData(ImDrawData* draw_data, evk::Cmd& cmd);
// The ImTextureID that draws `image` (a sampled image), for ImGui::Image and ImDrawList::AddImage: its resource id,
// which is never 0 for an image. An empty image gives 0, which ImGui treats as no texture.
IMGUI_IMPL_API ImTextureID ImGui_ImplEvk_Texture(const evk::Image& image);