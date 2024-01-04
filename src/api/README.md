APIs
================
This folder contains the source code for the APIs of our model.
APIs are hosted using uvicorn and the server can be launched simply by executing the following command while in the project's root folder
```
uvicorn src.api.api:app
```

Otherwise, if you want to use the APIs hosted by our server, you can connect to the following URL
```
https://itdisambiguation.azurewebsites.net/
```

Endpoints
----------------
The following endpoints are exposed by the server:
- **/models**: Gets a list of all the available models
- **/models/{model_name}**: Gets more informations about the specified model
- **/models/{model_name}/predict_context**: Predict the most relevant context given a list of contexts, an image and a target word
- **/models/{model_name}/predict_images**: Predict the most relevant image given a list of images, a context and a target word

API Documentation
The documentation is available at these links:

- ReDoc : https://itdisambiguation.azurewebsites.net/redoc
- Swagger : https://itdisambiguation.azurewebsites.net/docs

# Prometheus Monitoring for FastAPI
In addition to the API, we also implemented Prometheus for resource and performance monitoring. 
The monitoring is facilitated by the `prometheus_fastapi_instrumentator` package, which provides a convenient way to instrument a FastAPI application and collect various metrics.

## Instrumentator Configuration

The `Instrumentator` object is configured to collect metrics on request size, response size, request latency, and the total number of requests. It is set up to group status codes, ignore endpoints that do not have templated routes, and track in-progress requests. Additionally, the `/metrics` endpoint is excluded from instrumentation to prevent monitoring of the metrics path itself.

Here's a brief overview of each metric being collected:

### Request Size (`request_size`)

Tracks the size of incoming requests to the FastAPI application, providing insights into the incoming data load the server is handling. Given the nature of our project this metric is very important

### Response Size (`response_size`)

Measures the size of responses sent from the FastAPI application, which is useful for understanding the amount of data being served to clients.

### Latency (`latency`)

Records the latency of requests, offering a direct measure of the response times that clients are experiencing when interacting with the application.

### Total Number of Requests (`no_requests`)

Counts the total number of requests received by the FastAPI application, segmented by handler, method, and status. This metric is crucial for observing the traffic patterns and the load on the application.

## Environment Variables

- `METRICS_NAMESPACE`: Configurable namespace for the metrics, defaulting to "fastapi".
- `METRICS_SUBSYSTEM`: Configurable subsystem for the metrics, defaulting to "app".

These can be set to organize and distinguish metrics in environments where multiple applications or instances are being monitored.

[Prometheus Local Config](../../prometheus.yml)

[Prometheus Deploy Config](../../prometheus-deploy.yml)

[Instrumentator](./prometheus/instrumentator.py)

## Usage

Once configured, Prometheus scrape the `/metrics` endpoint of our FastApi application to collect the defined metrics.
Metric scraping of the backend deployed is available at this link : [Metrics](https://itdisambiguation.azurewebsites.net/metrics)

# Grafana
We used the Grafana tool to be able to graphically display some values ​​obtained from Prometheus metrics by performing queries.
The dashboard and metrics are displayed in the following readme:
[Main Readme](../../README.md)
