
if not defined INCLUDE (
    call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >NUL 2>&1
)

REM Configure the build and enable the example target
call cmake -S . -B build -D EVK_BUILD_EXAMPLE=ON -D CMAKE_BUILD_TYPE=Release
if errorlevel 1 (
    echo CMake configuration failed!
    exit /b 1
)

call cmake --build build --config Release -- --quiet

REM Detect if cmake build failed
if errorlevel 1 (
    echo Build failed!
    exit /b 1
)

REM Compile each shader
for %%f in (shaders\*.comp) do (
    glslc "%%f" -std=460 --target-env=vulkan1.3 -O -o "shaders/bin/%%~nf.comp.spv"
    if errorlevel 1 (
        exit /b 1
    )
)

call build\evk_example.exe