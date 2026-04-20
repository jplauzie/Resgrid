@echo off

call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

if errorlevel 1 (
    echo Failed to initialize MSVC environment
    exit /b 1
)

set PATH=C:\Program Files (x86)\SuiteSparse\bin;C:\Program Files (x86)\Intel\oneAPI\mkl\2025.0\redist\intel64;%PATH%

cl resgrid.cpp /std:c++17 /O2 /openmp /EHsc ^
/I"C:\Program Files (x86)\SuiteSparse\include\suitesparse" ^
/I"C:\Program Files (x86)\Intel\oneAPI\mkl\2025.0\include" ^
/link ^
/LIBPATH:"C:\Program Files (x86)\SuiteSparse\lib" ^
/LIBPATH:"C:\Program Files (x86)\Intel\oneAPI\mkl\2025.0\lib" ^
cholmod.lib amd.lib colamd.lib camd.lib ccolamd.lib suitesparseconfig.lib ^
mkl_intel_lp64.lib mkl_core.lib mkl_sequential.lib

echo Build complete