# Packaging image: runs the whole devagent CLI inside a container.
#
#   docker build -t devagent .
#   docker run --rm -it -v "%cd%:/work" --env-file "%USERPROFILE%\.devagent\.env" devagent --goal "fix the failing test"
#
# Inside this image the agent uses the local sandbox (DEVAGENT_SANDBOX=local): the
# container itself is the isolation boundary, and it needs network for the model API.
# For per-command isolation with no network, use `devagent build-sandbox` on the host instead.
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 PYTHONUTF8=1
RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates curl \
 && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
 && apt-get install -y --no-install-recommends nodejs \
 && apt-get purge -y curl && apt-get autoremove -y && rm -rf /var/lib/apt/lists/* \
 && git config --system --add safe.directory '*'

WORKDIR /app
COPY pyproject.toml README.md ./
COPY devagent ./devagent
RUN pip install --no-cache-dir . pytest

ENV DEVAGENT_SANDBOX=local
WORKDIR /work
ENTRYPOINT ["devagent"]
CMD ["--help"]
