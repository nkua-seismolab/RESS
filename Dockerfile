# RESS runtime image: SeisComP 7.x Python API + scientific stack.
#
# Base image note: the SeisComP tarballs ship Python bindings compiled against
# the distro's system Python. ubuntu24.04 builds target Python 3.12, matching
# our requires-python; python:3.12-slim (Debian-based) would NOT be ABI
# compatible with any of the Debian tarballs (py3.11/py3.13).
#
# The SeisComP tarball URL/version can be overridden at build time:
#   docker build --build-arg SEISCOMP_TARBALL_URL=... .
FROM ubuntu:24.04

ARG SEISCOMP_TARBALL_URL=https://www.seiscomp.de/downloader/seiscomp-7.3.1-ubuntu24.04-x86_64.tar.gz

ENV DEBIAN_FRONTEND=noninteractive

# bootstrap only; SeisComP runtime libs are installed by its own deps script below
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    ca-certificates \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# SeisComP (only the Python API and client libraries are used; no daemon runs
# here). Runtime dependencies come from the tarball's own install-base.sh so
# the package list stays correct across SeisComP versions; the script uses
# interactive "apt install", hence the sed to a non-interactive form.
# Afterwards, strip everything not needed to import seiscomp.client/datamodel.
RUN wget -q -O /tmp/seiscomp.tar.gz "${SEISCOMP_TARBALL_URL}" \
    && mkdir -p /opt \
    && tar -xzf /tmp/seiscomp.tar.gz -C /opt \
    && rm /tmp/seiscomp.tar.gz \
    && apt-get update \
    && sed 's/^apt install/apt-get install -y --no-install-recommends/' \
        /opt/seiscomp/share/deps/ubuntu/24.04/install-base.sh | sh \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /opt/seiscomp/share/maps /opt/seiscomp/share/doc /opt/seiscomp/share/deps \
        /opt/seiscomp/include /opt/seiscomp/man /opt/seiscomp/etc/descriptions

ENV SEISCOMP_ROOT=/opt/seiscomp \
    VIRTUAL_ENV=/opt/RESS \
    PATH=/opt/RESS/bin:/opt/seiscomp/bin:$PATH \
    LD_LIBRARY_PATH=/opt/seiscomp/lib \
    PYTHONPATH=/opt/seiscomp/lib/python

# uv venv "RESS", pinned to the system Python 3.12: the SeisComP bindings are
# ABI-tied to it; a uv-managed standalone interpreter would break that.
RUN uv venv /opt/RESS --python /usr/bin/python3.12
COPY ress/requirements.txt /tmp/requirements.txt
RUN uv pip install --no-cache -r /tmp/requirements.txt

WORKDIR /app
COPY pyproject.toml ./
COPY ress/ ./ress/
COPY velocity_models/ ./velocity_models/
RUN uv pip install --no-cache --no-deps .

# config.yaml and the SDS archive are provided as volumes (see docker-compose.yml)
CMD ["python", "-m", "ress", "--config", "/app/config.yaml"]
