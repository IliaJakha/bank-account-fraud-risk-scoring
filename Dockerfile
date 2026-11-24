FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/mpl \
    PYTHONPATH=/app/src

WORKDIR /app

COPY pyproject.toml LICENSE README.md Makefile ./
COPY src ./src
COPY configs ./configs
COPY tests ./tests

RUN pip install --no-cache-dir -e ".[dev]"

ENTRYPOINT ["python", "-m", "aml_graph"]
CMD ["--help"]
