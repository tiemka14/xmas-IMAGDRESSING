FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 python3-pip git && \
    apt-get clean

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

# Install IMAGDressing from upstream repo and make it importable as 'imagdressing'
RUN set -eux; \
    git clone https://github.com/muzishen/IMAGDressing.git /tmp/IMAGDressing; \
    # Move dressing_sd to package name 'imagdressing' so our imports match
    if [ -d /tmp/IMAGDressing/dressing_sd ]; then \
        mv /tmp/IMAGDressing/dressing_sd /tmp/IMAGDressing/imagdressing; \
        # Add package markers so it can be installed
        if [ ! -f /tmp/IMAGDressing/imagdressing/__init__.py ]; then touch /tmp/IMAGDressing/imagdressing/__init__.py; fi; \
        if [ -d /tmp/IMAGDressing/imagdressing/pipelines ] && [ ! -f /tmp/IMAGDressing/imagdressing/pipelines/__init__.py ]; then touch /tmp/IMAGDressing/imagdressing/pipelines/__init__.py; fi; \
    fi; \
    # Create minimal packaging to allow pip installation
    printf "[build-system]\nrequires = [\"setuptools\", \"wheel\"]\nbuild-backend = \"setuptools.build_meta\"\n" > /tmp/IMAGDressing/pyproject.toml; \
    printf "from setuptools import setup, find_packages\nsetup(name=\"imagdressing\", packages=find_packages())\n" > /tmp/IMAGDressing/setup.py; \
    # Install upstream requirements first so we have all dependencies
    if [ -f /tmp/IMAGDressing/requirements.txt ]; then pip3 install -r /tmp/IMAGDressing/requirements.txt; fi; \
    pip3 install /tmp/IMAGDressing; \
    rm -rf /tmp/IMAGDressing

COPY . .

# Expose FastAPI port
EXPOSE 8000

CMD ["/bin/bash", "start.sh"]
