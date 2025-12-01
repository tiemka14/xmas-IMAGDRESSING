# xmas-IMAGDRESSING

CI / Docker image build and push
---------------------------------

This repository contains a GitHub Actions workflow (`.github/workflows/docker-image.yml`) that builds and pushes a Docker image when you push to `main` or when a tag like `vX.Y.Z` is pushed.

Supported registries:
- GitHub Container Registry (GHCR): the workflow will use `ghcr.io/${{ github.repository_owner }}/xmas-imagdressing` and push when running on GitHub Actions. No additional secrets are required (uses `GITHUB_TOKEN` with `packages: write` permission).
- Docker Hub: optionally push to Docker Hub if you set `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` repository secrets.

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
