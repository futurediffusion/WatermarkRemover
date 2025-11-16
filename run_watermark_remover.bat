@echo off
setlocal ENABLEDELAYEDEXPANSION

REM Directorio base del proyecto
title WatermarkRemover
set "PROJECT_DIR=%~dp0"
set "VENV_DIR=%PROJECT_DIR%venv"
set "PYTHON=python"
set "EXIT_CODE=0"
set "TORCH_CUDA_INDEX_URL=https://download.pytorch.org/whl/cu126"
set "TORCH_CUDA_COMMAND=%PYTHON% -m pip install torch==2.6.0+cu126 torchvision==0.21.0 torchaudio==2.6.0+cu126 --index-url %TORCH_CUDA_INDEX_URL%"

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo [INFO] No se encontro entorno virtual. Creando uno nuevo...
    %PYTHON% -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [ERROR] No se pudo crear el entorno virtual.
        set "EXIT_CODE=1"
        goto :pause_and_exit
    )
    call "%VENV_DIR%\Scripts\activate"
    echo [INFO] Instalando dependencias...
    pip install --upgrade pip
    pip install -r "%PROJECT_DIR%requirements.txt"
    call :install_torch_if_needed
) else (
    echo [INFO] Usando el entorno virtual existente.
    call "%VENV_DIR%\Scripts\activate"
    pip show torch >nul 2>&1
    if errorlevel 1 (
        call :install_torch_if_needed
    )
)

echo [INFO] Iniciando WatermarkRemover con los parametros: %*
python "%PROJECT_DIR%watermark_remover.py" %*
set "EXIT_CODE=%ERRORLEVEL%"
if "%EXIT_CODE%"=="0" (
    echo [INFO] WatermarkRemover finalizo correctamente.
) else (
    echo [ERROR] WatermarkRemover finalizo con codigo de salida %EXIT_CODE%.
)

echo.
echo Presiona cualquier tecla para cerrar esta ventana...
:pause_and_exit
pause > nul
endlocal & exit /b %EXIT_CODE%

:install_torch_if_needed
echo [INFO] Instalando PyTorch recomendado para CUDA (si esta disponible)...
%TORCH_CUDA_COMMAND%
if errorlevel 1 (
    echo [ADVERTENCIA] No se pudo instalar la version CUDA recomendada. Instalando PyTorch para CPU...
    %PYTHON% -m pip install torch torchvision torchaudio
)
exit /b 0
