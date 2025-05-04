FROM python:3.11-slim

# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

# Configure apt and install packages
RUN apt-get update \
    && apt-get -y install --no-install-recommends apt-utils dialog 2>&1 \
    && apt-get -y install git iproute2 procps lsb-release wget curl build-essential \
    && apt-get -y install gcc g++ make cmake \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# Switch back to dialog for any ad-hoc use of apt-get
ENV DEBIAN_FRONTEND=dialog

# Set up user
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Set up Python environment
RUN pip install --upgrade pip \
    && pip install setuptools wheel twine pytest pytest-cov black flake8 mypy pylint \
    && pip install ipython jupyter notebook

# Install BentoML and its dependencies
RUN pip install bentoml \
    && pip install numpy pandas scikit-learn torch torchvision \
    && pip install fastapi uvicorn pydantic \
    && pip install click cloudpickle pyyaml \
    && pip install prometheus-client opentelemetry-api opentelemetry-sdk

# Create a directory for BentoML
RUN mkdir -p /home/$USERNAME/.bentoml \
    && chown -R $USERNAME:$USERNAME /home/$USERNAME/.bentoml

# Set the default user
USER $USERNAME

# Set up working directory
WORKDIR /workspace

# Set up environment variables
ENV PATH="/home/$USERNAME/.local/bin:${PATH}"
