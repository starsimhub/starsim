@echo off
REM Starsim installer for Windows.
REM Installs Miniforge (if conda isn't found) and pip-installs Starsim and Jupyter.
REM
REM Usage: download https://docs.starsim.org/install.bat, then double-click it, or run from Command Prompt:
REM    install.bat

setlocal enabledelayedexpansion

echo === Starsim installer ===

set "MINIFORGE_DIR=%USERPROFILE%\miniforge3"
set "CONDA_EXE="

REM Prefer conda already on PATH, otherwise look in the default Miniforge location.
where conda >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    for /f "delims=" %%i in ('where conda') do (
        if not defined CONDA_EXE set "CONDA_EXE=%%i"
    )
    echo ^>^>^> Found existing conda at: !CONDA_EXE!
) else if exist "%MINIFORGE_DIR%\Scripts\conda.exe" (
    set "CONDA_EXE=%MINIFORGE_DIR%\Scripts\conda.exe"
    echo ^>^>^> Found existing Miniforge at: %MINIFORGE_DIR%
) else (
    call :install_miniforge
    if errorlevel 1 exit /b 1
    set "CONDA_EXE=%MINIFORGE_DIR%\Scripts\conda.exe"
)

REM Starsim needs Python 3.12 or later; an older base env would otherwise get an old Starsim
"!CONDA_EXE!" run -n base python -c "import sys; sys.exit(sys.version_info < (3, 12))"
if errorlevel 1 (
    echo Starsim needs Python 3.12 or later, but the base conda env is older.
    echo Update it with "conda install -n base python=3.13", or install Starsim into a new env.
    pause
    exit /b 1
)

echo ^>^>^> Installing Starsim and Jupyter into the base conda env
"!CONDA_EXE!" run --no-capture-output -n base python -m pip install --upgrade pip
if errorlevel 1 exit /b 1
"!CONDA_EXE!" run --no-capture-output -n base pip install starsim jupyter
if errorlevel 1 exit /b 1

echo.
echo === Done ===
echo Open a new Command Prompt and check the install with:
echo     python -c "import starsim as ss; print(ss.__version__)"
echo.
echo Launch Jupyter with:
echo     jupyter notebook
echo.
pause
exit /b 0

:install_miniforge
set "INSTALLER=%TEMP%\Miniforge3-Windows-x86_64.exe"
set "URL=https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe"
echo ^>^>^> Downloading %URL%
powershell -NoProfile -Command "Invoke-WebRequest -Uri '%URL%' -OutFile '%INSTALLER%'"
if errorlevel 1 exit /b 1
echo ^>^>^> Installing Miniforge to %MINIFORGE_DIR%
start /wait "" "%INSTALLER%" /InstallationType=JustMe /RegisterPython=1 /AddToPath=1 /S /D=%MINIFORGE_DIR%
if errorlevel 1 exit /b 1
del "%INSTALLER%" >nul 2>nul
exit /b 0
