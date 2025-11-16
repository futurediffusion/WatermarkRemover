@echo off
setlocal ENABLEDELAYEDEXPANSION

REM Directorio base del proyecto
title WatermarkRemover
set "PROJECT_DIR=%~dp0"
set "VENV_DIR=%PROJECT_DIR%venv"
set "PYTHON=python"
set "EXIT_CODE=0"

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
) else (
    echo [INFO] Usando el entorno virtual existente.
    call "%VENV_DIR%\Scripts\activate"
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
