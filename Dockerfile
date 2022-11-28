FROM tensorflow/tensorflow:2.10.1-gpu

ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m pip install --upgrade pip
RUN pip3 install poetry

WORKDIR /app
COPY pyproject.toml poetry.lock README.md .
COPY dflow_samples dflow_samples

ENV PYTHONPATH=${PYTHONPATH}:/usr/lib/python3.8/site-packages:${PWD}

RUN poetry config virtualenvs.create false
RUN poetry config installer.max-workers 4

RUN poetry install --no-dev
