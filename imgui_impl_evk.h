#pragma once
#include "imgui.h"      // IMGUI_IMPL_API
#include "evk.h"

IMGUI_IMPL_API bool     ImGui_ImplEvk_Init(evk::Cmd& cmd);
IMGUI_IMPL_API void     ImGui_ImplEvk_Shutdown();
IMGUI_IMPL_API void     ImGui_ImplEvk_PrepareRender(ImDrawData* draw_data, evk::Cmd& cmd);
IMGUI_IMPL_API void     ImGui_ImplEvk_RenderDrawData(ImDrawData* draw_data, evk::Cmd& cmd);