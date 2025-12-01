FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git wget bzip2 ca-certificates curl && \
    apt-get clean

# Install Miniconda
ARG CONDA_SH=Miniconda3-latest-Linux-x86_64.sh
RUN wget -q https://repo.anaconda.com/miniconda/${CONDA_SH} -O /tmp/${CONDA_SH} || \
    (curl -fsSL https://repo.anaconda.com/miniconda/${CONDA_SH} -o /tmp/${CONDA_SH}) && \
    bash /tmp/${CONDA_SH} -b -p /opt/conda && \
    rm /tmp/${CONDA_SH} && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    apt-get clean

WORKDIR /app

COPY requirements.txt .
ENV PATH=/opt/conda/bin:$PATH

# Create conda env to satisfy Python & GPU package requirements
RUN conda create -y -n IMAGDressing python=3.10 && \
    conda run -n IMAGDressing python -m pip install --upgrade pip

# Install PyTorch + CUDA using conda to ensure GPU compatibility
RUN conda install -n IMAGDressing -y -c pytorch -c nvidia pytorch=2.1 torchvision=0.16 pytorch-cuda=11.8 && \
    # Install other pip requirements (exclude torch and torchvision to avoid conflicts)
    grep -vE "^(torch|torchvision)==" requirements.txt > /tmp/reqs_pip.txt && \
    conda run -n IMAGDressing pip install -r /tmp/reqs_pip.txt && rm /tmp/reqs_pip.txt

# Install IMAGDressing from upstream repo and make it importable as 'imagdressing'
RUN set -eux; \
    git clone https://github.com/muzishen/IMAGDressing.git /tmp/IMAGDressing; \
    # Move dressing_sd to package name 'imagdressing' so our imports match
    if [ -d /tmp/IMAGDressing/dressing_sd ]; then \
        mv /tmp/IMAGDressing/dressing_sd /tmp/IMAGDressing/imagdressing; \
        if [ ! -f /tmp/IMAGDressing/imagdressing/__init__.py ]; then touch /tmp/IMAGDressing/imagdressing/__init__.py; fi; \
        if [ -d /tmp/IMAGDressing/imagdressing/pipelines ] && [ ! -f /tmp/IMAGDressing/imagdressing/pipelines/__init__.py ]; then touch /tmp/IMAGDressing/imagdressing/pipelines/__init__.py; fi; \
    fi; \
    # Create minimal packaging to allow pip installation
    printf "[build-system]\nrequires = [\"setuptools\", \"wheel\"]\nbuild-backend = \"setuptools.build_meta\"\n" > /tmp/IMAGDressing/pyproject.toml; \
    printf "from setuptools import setup, find_packages\nsetup(name=\"imagdressing\", packages=find_packages())\n" > /tmp/IMAGDressing/setup.py; \
    # Use conda-run to install upstream requirements and package into our env
    if [ -f /tmp/IMAGDressing/requirements.txt ]; then conda run -n IMAGDressing pip install -r /tmp/IMAGDressing/requirements.txt || true; fi; \
    conda run -n IMAGDressing pip install /tmp/IMAGDressing; \
    rm -rf /tmp/IMAGDressing

COPY . .

# Expose FastAPI port
EXPOSE 8000

CMD ["/bin/bash", "start.sh"]
