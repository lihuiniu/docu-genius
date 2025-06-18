from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

def init_tracing(app):
    FastAPIInstrumentor.instrument_app(app)
