FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Europe/Paris

# Remove any third-party apt sources to avoid issues with expiring keys.
# Install some basic utilities
RUN rm -f /etc/apt/sources.list.d/*.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    sudo \
    git \
    wget \
    procps \
    git-lfs \
    zip \
    unzip \
    htop \
    vim \
    nano \
    bzip2 \
    libx11-6 \
    build-essential \
    libsndfile-dev \
    software-properties-common \
    tmux \
    rsync \
 && rm -rf /var/lib/apt/lists/*

RUN curl -sL https://deb.nodesource.com/setup_14.x  | bash - && \
    apt-get install -y nodejs && \
    npm install -g configurable-http-proxy

# Create a working directory
WORKDIR /chemCPA

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /chemCPA
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user

# Add user to sudo group
RUN usermod -aG sudo user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN mkdir $HOME/.cache $HOME/.config \
 && chmod -R 777 $HOME

# Create .local/bin directory and add it to PATH
RUN mkdir -p $HOME/.local/bin \
 && chmod -R 777 $HOME/.local
ENV PATH="$HOME/.local/bin:$PATH"

# Set up the Mamba environment as the non-root user
USER user

ENV MAMBA_ROOT_PREFIX=/home/user/micromamba
RUN wget -O Miniforge3.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" && \
    bash Miniforge3.sh -b -p "${HOME}/conda" && \
    rm Miniforge3.sh

ENV PATH="${HOME}/conda/bin:${PATH}"
SHELL ["/bin/bash", "-c"]
RUN conda init bash && \
    . ${HOME}/conda/etc/profile.d/conda.sh && \
    conda activate base && \
    echo "conda activate chemCPA" >> ~/.bashrc

WORKDIR $HOME/chemCPA

#######################################
# Start root user section
#######################################

USER root

# User Debian packages
## Security warning : Potential user code executed as root (build time)
RUN --mount=target=/root/packages.txt,source=packages.txt \
    apt-get update && \
    xargs -r -a /root/packages.txt apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=target=/root/on_startup.sh,source=on_startup.sh,readwrite \
    bash /root/on_startup.sh

RUN mkdir /data && chown user:user /data

#######################################
# End root user section
#######################################
USER user

# Copy all files from the current directory into the container
COPY --chown=user . $HOME/chemCPA

# Make scripts executable
RUN chmod +x $HOME/chemCPA/preprocessing/convert_notebooks.sh
RUN chmod +x $HOME/chemCPA/chemCPA/train_hydra_tmux.sh

# Create the Conda environment
RUN --mount=target=environment.yml,source=environment.yml \
    conda env create -f environment.yml && \
    conda clean -afy && \
    chown -R user:user /home/user/conda/envs/chemCPA && \
    chown -R user:user /home/user/.conda

# Activate the Conda environment and run additional pip commands
RUN . /home/user/conda/etc/profile.d/conda.sh && \
    conda activate chemCPA && \
    pip install -e . && \
    pip install sfaira && \
    pip install descriptastorus

RUN . /home/user/conda/etc/profile.d/conda.sh && \
    conda activate chemCPA && \
    pip install gdown jupytext

# Set LD_LIBRARY_PATH in conda activate.d
RUN . /home/user/conda/etc/profile.d/conda.sh && \
    conda activate chemCPA && \
    mkdir -p $CONDA_PREFIX/etc/conda/activate.d && \
    echo "export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH" > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

# Delete project_folder if it exists
RUN rm -rf $HOME/chemCPA/project_folder

# Ensure the docker_entrypoint.sh script is executable
RUN chmod +x docker_entrypoint.sh

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    GRADIO_ALLOW_FLAGGING=never \
    GRADIO_NUM_PORTS=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_THEME=huggingface \
    SYSTEM=spaces \
    SHELL=/bin/bash

# Ensure all files and directories are owned by the 'user'
USER root
RUN chown -R user:user $HOME
USER user

# Ensure proper permissions for /tmp and apt directories
USER root
RUN chmod 1777 /tmp && \
    mkdir -p /var/lib/apt/lists/partial && \
    chmod 755 /var/lib/apt/lists/partial && \
    chown -R user:user /var/lib/apt/lists/partial

# Create required apt directories with proper permissions
RUN mkdir -p /var/cache/apt/archives/partial && \
    chmod 755 /var/cache/apt/archives/partial && \
    chown -R user:user /var/cache/apt/archives/partial

# Final user switch
USER user

# At the end of the Dockerfile, add:
USER root
RUN chmod 1777 /tmp
RUN chmod 755 /home/user
USER user

# Create a jupyter_notebook_config.py file with CORS settings
RUN mkdir -p /home/user/.jupyter && \
    echo "c.NotebookApp.allow_origin = '*'" > /home/user/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.allow_credentials = True" >> /home/user/.jupyter/jupyter_notebook_config.py

# Ensure the config file has the correct permissions
RUN chown user:user /home/user/.jupyter/jupyter_notebook_config.py && \
    chmod 644 /home/user/.jupyter/jupyter_notebook_config.py

# Modify the CMD to use the custom config
CMD ["./docker_entrypoint.sh"]

