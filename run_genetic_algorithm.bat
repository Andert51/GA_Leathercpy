@echo off
REM Script para ejecutar el Algoritmo Genetico de Cortes Circulares
REM ================================================================

echo.
echo ===================================================================
echo ALGORITMO GENETICO PARA OPTIMIZACION DE CORTES CIRCULARES
echo ===================================================================
echo.

REM Verificar si Python esta instalado
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python no esta instalado o no esta en el PATH
    echo Por favor instale Python 3.8 o superior
    pause
    exit /b 1
)

REM Cambiar al directorio del script
cd /d "%~dp0"

REM Verificar si existe requirements.txt
if not exist "requirements.txt" (
    echo ERROR: No se encuentra el archivo requirements.txt
    echo Verifique que este en el directorio correcto
    pause
    exit /b 1
)

REM Preguntar si instalar dependencias
echo ¿Desea instalar/actualizar las dependencias de Python? (s/N)
set /p install_deps="Respuesta: "
if /i "%install_deps%"=="s" (
    echo.
    echo Instalando dependencias...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Fallo la instalacion de dependencias
        pause
        exit /b 1
    )
    echo Dependencias instaladas correctamente.
    echo.
)

REM Verificar si existe la imagen de entrada
if not exist "resources\images\piel_carnero.png" (
    echo ERROR: No se encuentra la imagen de entrada
    echo Verifique que resources\images\piel_carnero.png exista
    pause
    exit /b 1
)

REM Crear directorio de resultados si no existe
if not exist "results" mkdir results

REM Ejecutar el programa
echo Ejecutando el algoritmo genetico...
echo.
cd src
python run_ga.py

REM Verificar si la ejecucion fue exitosa
if errorlevel 1 (
    echo.
    echo ERROR: El programa termino con errores
    pause
    exit /b 1
)

echo.
echo ===================================================================
echo EJECUCION COMPLETADA EXITOSAMENTE
echo ===================================================================
echo.
echo Los resultados se encuentran en la carpeta 'results'
echo.

REM Preguntar si abrir la carpeta de resultados
set /p open_results="¿Desea abrir la carpeta de resultados? (s/N): "
if /i "%open_results%"=="s" (
    cd ..
    start explorer results
)

pause