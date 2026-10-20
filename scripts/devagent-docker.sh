#!/usr/bin/env sh
# Run the devagent CLI image against the current directory.
# Build once:  docker build -t devagent .      (from the devagent source folder)
# Settings come from ~/.devagent/.env (create it with: devagent init)
exec docker run --rm -it \
  -v "$(pwd):/work" \
  --env-file "$HOME/.devagent/.env" \
  devagent "$@"
