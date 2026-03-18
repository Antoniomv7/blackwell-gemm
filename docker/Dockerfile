# Pinned CUDA base image (do NOT use 'latest' for reproducibility)
FROM nvidia/cuda:13.0.0-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive

# System deps (minimal but useful for research/profiling/dev)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs ca-certificates curl wget \
    build-essential cmake ninja-build pkg-config \
    python3 python3-pip python3-venv \
    vim nano tmux htop \
    pciutils usbutils \
    && rm -rf /var/lib/apt/lists/*

# ---- Micromamba (fast conda) pinned version ----
ARG MAMBA_VERSION=1.5.8
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/${MAMBA_VERSION} \
  | tar -xvj -C /usr/local/bin/ --strip-components=1 bin/micromamba

# Create a non-root user matching common devcontainer conventions
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000

RUN if ! getent group ${USER_GID} >/dev/null 2>&1; then \
        groupadd --gid ${USER_GID} ${USERNAME}; \
    else \
        echo "Using existing group with GID ${USER_GID}"; \
    fi && \
    if id -u ${USERNAME} >/dev/null 2>&1; then \
        echo "User ${USERNAME} already exists"; \
    elif getent passwd ${USER_UID} >/dev/null 2>&1; then \
        FALLBACK_UID="$(awk -F: 'BEGIN{u=2000} {used[$3]=1} END{while(used[u])u++; print u}' /etc/passwd)"; \
        echo "UID ${USER_UID} already exists, creating ${USERNAME} with fallback UID ${FALLBACK_UID}"; \
        useradd --uid "${FALLBACK_UID}" --gid ${USER_GID} -m ${USERNAME} -s /bin/bash; \
    else \
        useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} -s /bin/bash; \
    fi

# Shell init for micromamba for the vscode user
USER ${USERNAME}
ENV MAMBA_ROOT_PREFIX=/home/${USERNAME}/micromamba
RUN micromamba shell init -s bash -p ${MAMBA_ROOT_PREFIX}

# Default workdir
WORKDIR /workspaces/blackwell-gemm

# Copy env spec and create env (this is the "environment inside docker")
# We copy only environment.yml first to maximize layer caching.
COPY --chown=${USERNAME}:${USERNAME} env/environment.yml /tmp/environment.yml
RUN micromamba create -y -n bw -f /tmp/environment.yml && \
    micromamba clean -a -y && \
    chmod o+rx /home/${USERNAME} && \
    chmod -R a+rX /home/${USERNAME}/micromamba

# Make the env the default
ENV CONDA_DEFAULT_ENV=bw
ENV PATH=/home/${USERNAME}/micromamba/envs/bw/bin:$PATH

# Optional: basic sanity tools
RUN python -V && pip -V && nvcc --version

CMD [ "bash" ]