"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >NUL 2>&1 &&^
cmake --build build -- --quiet &&^
glslc shaders/matmul_coop.comp -std=460 --target-env=vulkan1.3 &&^
build\evk_example.exe