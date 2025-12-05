
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >NUL 2>&1
cmake --build build -- --quiet

REM Detect if cmake build failed
if errorlevel 1 (
    echo Build failed!
    exit /b 1
)

REM Compile each shader
for %%f in (shaders\*.comp) do (
    glslc "%%f" -std=460 --target-env=vulkan1.3 -o "shaders/bin/%%~nf.comp.spv"
    if errorlevel 1 (
        exit /b 1
    )
)

build\evk_example.exe