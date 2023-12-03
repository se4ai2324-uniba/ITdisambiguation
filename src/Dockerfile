
FROM python:3.8-slim-buster

RUN apt-get update && apt-get install -y git
COPY requirements_docker.txt requirements.txt
RUN pip install -r requirements.txt
WORKDIR /workdir
COPY setup.py /workdir/setup.py
COPY src /workdir/src
COPY models /workdir/models
COPY metrics /workdir/metrics
EXPOSE 80
CMD ["uvicorn", "src.api.api:app", "--host", "0.0.0.0", "--port", "8000"]