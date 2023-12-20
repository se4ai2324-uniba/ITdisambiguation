FROM python:3.8-slim-buster

# Installa git se necessario e pulisci la cache di apt in un unico passaggio
RUN apt-get update \
    && apt-get install -y git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Ottimizza l'uso della cache di Docker copiando prima i file delle dipendenze
COPY requirements_docker.txt /workdir/requirements.txt
WORKDIR /workdir
RUN pip install --no-cache-dir -r requirements.txt

# Copia il resto dei file necessari
COPY setup.py ./setup.py
COPY src ./src
COPY models ./models
COPY metrics ./metrics

EXPOSE 8000
CMD ["uvicorn", "src.api.api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
