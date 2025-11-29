"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >NUL 2>&1 &&^
cmake --build build -- --quiet &&^
glslc shaders/matmul.comp -std=460 --target-env=vulkan1.3 -o shaders/bin/matmul.comp.spv &&^
glslc shaders/mse_loss.comp -std=460 --target-env=vulkan1.3 -o shaders/bin/mse_loss.comp.spv &&^
glslc shaders/sgd.comp -std=460 --target-env=vulkan1.3 -o shaders/bin/sgd.comp.spv &&^
glslc shaders/adam.comp -std=460 --target-env=vulkan1.3 -o shaders/bin/adam.comp.spv &&^
glslc shaders/add.comp -std=460 --target-env=vulkan1.3 -o shaders/bin/add.comp.spv &&^
glslc shaders/softmax.comp -std=460 --target-env=vulkan1.3 -o shaders/bin/softmax.comp.spv &&^
glslc shaders/cross_entropy.comp -std=460 --target-env=vulkan1.3 -o shaders/bin/cross_entropy.comp.spv &&^
glslc shaders/transpose.comp -std=460 --target-env=vulkan1.3 -o shaders/bin/transpose.comp.spv &&^
glslc shaders/flash_attention.comp -std=460 --target-env=vulkan1.3 -o shaders/bin/flash_attention.comp.spv &&^
glslc shaders/flash_attention_bwd.comp -std=460 --target-env=vulkan1.3 -o shaders/bin/flash_attention_bwd.comp.spv &&^
build\evk_example.exe