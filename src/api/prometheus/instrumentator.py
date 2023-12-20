import os
from prometheus_fastapi_instrumentator import Instrumentator, metrics


NAMESPACE = os.environ.get("METRICS_NAMESPACE", "fastapi")
SUBSYSTEM = os.environ.get("METRICS_SUBSYSTEM", "app")

# Crea un oggetto Instrumentator
instrumentator = Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
    should_instrument_requests_inprogress=True,
    excluded_handlers=["/metrics"],  # Escludi l'endpoint /metrics dall'istrumentazione
    inprogress_name="inprogress",
    inprogress_labels=True,
)

# Aggiungi metriche personalizzate o standard al tuo instrumentator

# Misura la dimensione delle richieste
instrumentator.add(
    metrics.request_size(
        should_include_handler=True,
        should_include_method=True,
        should_include_status=True,
        metric_name="request_size",
        metric_namespace=NAMESPACE,
        metric_subsystem=SUBSYSTEM,
    )
)

# Misura la dimensione delle risposte
instrumentator.add(
    metrics.response_size(
        should_include_handler=True,
        should_include_method=True,
        should_include_status=True,
        metric_name="response_size",
        metric_namespace=NAMESPACE,
        metric_subsystem=SUBSYSTEM,
    )
)

# Misura la latenza delle richieste
instrumentator.add(
    metrics.latency(
        should_include_handler=True,
        should_include_method=True,
        should_include_status=True,
        metric_name="latency",
        metric_namespace=NAMESPACE,
        metric_subsystem=SUBSYSTEM,
    )
)

# Conta il numero totale di richieste
instrumentator.add(
    metrics.requests(
        should_include_handler=True,
        should_include_method=True,
        should_include_status=True,
        metric_name="no_requests",
        metric_namespace=NAMESPACE,
        metric_subsystem=SUBSYSTEM,
    )
)
