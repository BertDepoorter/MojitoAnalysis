# FROM gitlab.esa.int:4567/lisa-sgs/sandbox/lisa-cde:main
FROM gitlab.esa.int:4567/lisa-sgs/sim/lisasim-bench:feature-init-docker-env-gpu-arm64

WORKDIR /app

COPY . /app

# Base image configured non root user, but we need root to install missing packages.
USER root

ENV DEBIAN_FRONTEND=noninteractive

ARG CI_PROJECT_DIR=${CI_PROJECT_DIR}
ARG LISASIM_IN2P3_USER=${LISASIM_IN2P3_USER}
ARG LISASIM_READ_USER=${LISASIM_READ_USER}

WORKDIR ${CI_PROJECT_DIR}
COPY src/lisasim/EMRI/emri_env.yaml .
RUN conda update conda \
    && conda env update -n base --file emri_env.yaml
# RUN conda update conda \
#     && mamba update -n base --file emri_env.yaml

# # Config poetry so that it works with conda
RUN pipx ensurepath
RUN poetry config virtualenvs.path $CONDA_ENV_PATH \
    && poetry config virtualenvs.create false

RUN python -c 'import few; \
               print(f" - Backend cuda11x: {"available" if few.has_backend("cuda11x") else "unavailable"}"); \
               print(f" - Backend cuda12x: {"available" if few.has_backend("cuda12x") else "unavailable"}"); \
               print(f" - Backend cuda: {"available" if few.has_backend("cuda") else "unavailable"}");'

# Reverting to non root user as a best-practice.
USER lisauser

# ENTRYPOINT ["python"]
ENTRYPOINT jupyter-lab --allow-root --ip=0.0.0.0
EXPOSE 8888
