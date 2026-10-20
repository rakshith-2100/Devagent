@echo off
rem Run the devagent CLI image against the current directory.
rem Build once:  docker build -t devagent .      (from the devagent source folder)
rem Settings come from %USERPROFILE%\.devagent\.env (create it with: devagent init)
docker run --rm -it ^
  -v "%cd%:/work" ^
  --env-file "%USERPROFILE%\.devagent\.env" ^
  devagent %*
