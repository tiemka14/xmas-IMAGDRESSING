#!/bin/bash
# Activate conda environment created in Docker image and start the server
if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
	. /opt/conda/etc/profile.d/conda.sh
	conda activate IMAGDressing || true
fi

exec uvicorn app.main:app --host 0.0.0.0 --port 8000
