@echo off
cd /d "%~dp0"
if defined SUMO_HOME if exist "%SUMO_HOME%\bin\sumo-gui.exe" (
  start "" "%SUMO_HOME%\bin\sumo-gui.exe" -c "%~dp0toegyewon.sumocfg"
  exit /b
)
if exist "C:\Program Files (x86)\Eclipse\Sumo\bin\sumo-gui.exe" (
  start "" "C:\Program Files (x86)\Eclipse\Sumo\bin\sumo-gui.exe" -c "%~dp0toegyewon.sumocfg"
  exit /b
)
start "" sumo-gui -c "%~dp0toegyewon.sumocfg"
