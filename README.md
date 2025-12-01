# xmas-IMAGDRESSING

CI / Docker image build and push
---------------------------------

This repository contains a GitHub Actions workflow (`.github/workflows/docker-image.yml`) that builds and pushes a Docker image when you push to `main` or when a tag like `vX.Y.Z` is pushed.

Supported registries:
- GitHub Container Registry (GHCR): the workflow will use `ghcr.io/${{ github.repository_owner }}/xmas-imagdressing` and push when running on GitHub Actions. No additional secrets are required (uses `GITHUB_TOKEN` with `packages: write` permission).
- Docker Hub: optionally push to Docker Hub if you set `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` repository secrets.

Note: This project uses VITON-HD (https://github.com/shadow2496/VITON-HD) as the backend virtual try-on implementation. We clone the VITON-HD repo and install necessary dependencies during the Docker build — see `Dockerfile` for details.

Required GitHub repo permissions to push to GHCR:
- In your workflow file we set `permissions: packages: write` and `contents: read`. That allows the `GITHUB_TOKEN` to push packages to GHCR.

To enable pushing to Docker Hub (optional):
1. Create a Docker Hub access token (not your password). Create it at https://hub.docker.com/settings/security.
2. Add the following repository secrets in GitHub repo settings:
	- `DOCKERHUB_USERNAME`: your Docker Hub username
	- `DOCKERHUB_TOKEN`: the token value created in step 1

When both secrets are provided the workflow logs into Docker Hub and pushes the same image tags as are pushed to GHCR.

Image tags published by the workflow:
- `latest` (mutated on push to main)
- commit SHA as a tag (a unique tag for traceability)

Example of image names pushed by the workflow:
- GHCR: `ghcr.io/<OWNER>/xmas-imagdressing:latest` and `ghcr.io/<OWNER>/xmas-imagdressing:<COMMIT_SHA>`
- Docker Hub: `docker.io/<DOCKERHUB_USERNAME>/xmas-imagdressing:latest` and `docker.io/<DOCKERHUB_USERNAME>/xmas-imagdressing:<COMMIT_SHA>`

If you want to customize the image name or tags, edit `.github/workflows/docker-image.yml`.

Installation (Conda)
---------------------

These instructions follow the upstream VITON-HD README and will create a Conda env and install dependencies.

1) Create the Conda environment and activate it (we recommend `VITON_HD` name):

```bash
conda create --name VITON_HD python=3.10 -y
conda activate VITON_HD
```

2) Upgrade pip and install requirements:

```bash
pip install -U pip
pip install -r requirements.txt
```

3) Download the model from Hugging Face:

The code (`app/model.py`) now uses VITON-HD. To run with the VITON-HD pre-trained checkpoints, download them from the VITON-HD Google Drive and place them in `/opt/VITON-HD/checkpoints/` (or set environment variables as appropriate). To run VITON-HD tests directly use their `test.py` script in the `VITON-HD` repo.

- Use the HuggingFace Hub `from_pretrained` API in code (requires `huggingface_hub` and to accept terms on the model page):
	```python
	from diffusers import DiffusionPipeline
	# We now use VITON-HD; if you want to use a different pipeline replace the adapter with your desired model
	```
- Or download the repo/weights to a local folder (optional), then point `from_pretrained` at that path.

Tip: You can use `scripts/setup_conda_env.sh` to automate the environment creation and requirements installation for local development.

Docker & Python notes
---------------------
The Docker image now uses an NVIDIA CUDA 11.8 base and installs Miniconda. A dedicated Conda environment (`IMAGDressing`) with Python 3.10 will be created during image build and is used for all runtime steps. This helps satisfy the IMAGDressing upstream requirement for CUDA 11.8 and ensures PyTorch is installed with the matching CUDA toolkit.

Build & Run the Docker image (example):
```bash
# Build the image (multi-arch may be slower)
docker build -t xmas-imagdressing:latest .

# Run the container (requires NVIDIA Container Toolkit for GPU access):
docker run --gpus all -p 8000:8000 xmas-imagdressing:latest
```

Inside the container, the app runs with the Conda environment active by default (see `start.sh`).
