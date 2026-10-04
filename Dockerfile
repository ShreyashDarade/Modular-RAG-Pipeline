# syntax=docker/dockerfile:1.7
#
# One Dockerfile, three images - choose with --build-arg EXTRAS:
#
#   API replica (small, stateless; no OCR / torch):
#     docker build -t rag-api    --build-arg EXTRAS=api,mcp .
#   Ingestion worker (OCR, PDF, DOCX, XLSX; CPU torch by default):
#     docker build -t rag-worker --build-arg EXTRAS=worker,docx,xlsx .
#   GPU worker: add  --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu128
#   Everything in one container (development / small deployments):
#     docker build -t rag-all    --build-arg EXTRAS=all .
#
# The command is chosen at run time: `rag-api` (default), `rag-worker`, `rag-mcp`, `rag ...`.

ARG PYTHON_VERSION=3.13

FROM python:${PYTHON_VERSION}-slim AS builder
ARG EXTRAS=api,mcp
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1
RUN pip install --no-cache-dir uv
WORKDIR /build
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN uv venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH
# torch first, from the chosen index, so the worker never pulls the multi-GB CUDA wheels by accident
RUN case ",${EXTRAS}," in \
      *,worker,*|*,local,*|*,all,*) uv pip install torch torchvision --index-url "${TORCH_INDEX}" ;; \
    esac
RUN uv pip install ".[${EXTRAS}]"

FROM python:${PYTHON_VERSION}-slim AS runtime
ARG EXTRAS=api,mcp
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/opt/venv/bin:$PATH \
    DATA_DIR=/data \
    OCR_MODEL_DIR=/models \
    LOG_FORMAT=json
# libGL/glib: runtime needs of OpenCV (headless build) in the worker image
RUN case ",${EXTRAS}," in \
      *,worker,*|*,all,*) apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/* ;; \
    esac
RUN useradd --system --create-home --uid 10001 rag && mkdir -p /data /models && chown rag /data /models
COPY --from=builder /opt/venv /opt/venv
COPY --chown=rag config /app/config
USER rag
WORKDIR /app
VOLUME ["/data", "/models"]
EXPOSE 8000 9100
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"
CMD ["rag-api"]
