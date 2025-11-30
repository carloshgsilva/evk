
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >NUL 2>&1
cmake --build build -- --quiet

REM Detect if cmake build failed
if errorlevel 1 (
    echo Build failed!
    exit /b 1
)

REM Compile each shader
for %%s in (matmul mse_loss sgd adam add softmax cross_entropy transpose flash_attention flash_attention_bwd) do (
    glslc shaders/%%s.comp -std=460 --target-env=vulkan1.3 -o shaders/bin/%%s.comp.spv
    if errorlevel 1 (
        exit /b 1
    )
)

build\evk_example.exe