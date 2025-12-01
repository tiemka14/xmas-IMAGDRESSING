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
    /opt/conda/bin/conda clean -a -y && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    apt-get clean

WORKDIR /app

COPY requirements.txt .
ENV PATH=/opt/conda/bin:$PATH
ENV PYTHONPATH=/opt/VITON-HD

# Create conda env to satisfy Python & GPU package requirements
RUN conda create -y -n VITON_HD python=3.10 && \
    conda run -n VITON_HD python -m pip install --upgrade pip

# Install PyTorch + CUDA using conda to ensure GPU compatibility
RUN conda install -n VITON_HD -y -c pytorch -c nvidia pytorch=2.1 torchvision=0.16 pytorch-cuda=11.8 && \
    # Install other pip requirements (exclude torch and torchvision to avoid conflicts)
    grep -vE "^(torch|torchvision)==" requirements.txt > /tmp/reqs_pip.txt && \
    conda run -n VITON_HD pip install -r /tmp/reqs_pip.txt && rm /tmp/reqs_pip.txt

# Install VITON-HD from upstream repo into /opt/VITON-HD
RUN set -eux; \
    git clone https://github.com/shadow2496/VITON-HD.git /opt/VITON-HD; \
    # Keep repo at /opt/VITON-HD; add __init__ markers if necessary
    if [ -d /opt/VITON-HD ]; then \
        if [ ! -f /opt/VITON-HD/__init__.py ]; then touch /opt/VITON-HD/__init__.py; fi; \
    fi; \
    # Create minimal packaging to allow pip installation
    # (no longer using /tmp/IMAGDressing)
    printf "[build-system]\nrequires = [\"setuptools\", \"wheel\"]\nbuild-backend = \"setuptools.build_meta\"\n" > /opt/VITON-HD/pyproject.toml; \
    printf "from setuptools import setup, find_packages\nsetup(name=\"viton_hd\", packages=find_packages())\n" > /opt/VITON-HD/setup.py; \
    # Install runtime dependencies for VITON-HD into the conda env
    conda run -n VITON_HD pip install -U opencv-python torchgeometry || true; \
    conda run -n VITON_HD pip install /opt/VITON-HD || true;

COPY . .

# Expose FastAPI port
EXPOSE 8000

CMD ["/bin/bash", "start.sh"]
