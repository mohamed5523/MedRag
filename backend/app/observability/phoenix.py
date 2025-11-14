import logging
import os

from openinference.instrumentation.langchain import LangChainInstrumentor
from openinference.instrumentation.openai import OpenAIInstrumentor
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter as OTLPHTTPSpanExporter,
)

try:
    # Optional gRPC exporter (fallback)
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter as OTLPGRPCSpanExporter,  # type: ignore
    )
except Exception:  # pragma: no cover - grpc exporter may not be installed
    OTLPGRPCSpanExporter = None  # type: ignore
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.trace import Span

logger = logging.getLogger(__name__)

# Suppress noisy OpenTelemetry export errors
logging.getLogger("opentelemetry.sdk._shared_internal").setLevel(logging.ERROR)
logging.getLogger("opentelemetry.exporter.otlp.proto.http.trace_exporter").setLevel(logging.ERROR)


class SilentErrorSpanExporter:
    """Wrapper around OTLPSpanExporter that suppresses connection errors but still exports."""
    
    def __init__(self, exporter: SpanExporter):
        self._exporter = exporter
        self._connection_warned = False
        self._success_count = 0
        self._failure_count = 0
    
    def export(self, spans: list[Span]) -> SpanExportResult:
        try:
            result = self._exporter.export(spans)
            # If we get a successful export, log success (first time and periodically)
            if result == SpanExportResult.SUCCESS:
                self._success_count += 1
                if self._success_count == 1 or self._success_count % 10 == 0:
                    logger.info(f"✅ Successfully exported {len(spans)} spans to Phoenix (batch #{self._success_count})")
            else:
                logger.debug(f"Export returned: {result}")
            return result
        except Exception as e:
            self._failure_count += 1
            # Only warn once about connection issues to reduce log noise
            if not self._connection_warned:
                logger.warning(
                    f"⚠️ Phoenix connection issue: {type(e).__name__}: {str(e)[:100]}. "
                    f"Spans are buffered and will retry automatically. "
                    f"Ensure Phoenix is running at the OTLP endpoint."
                )
                self._connection_warned = True
            elif self._failure_count % 50 == 0:
                logger.warning(f"⚠️ Phoenix export failures: {self._failure_count} (still retrying)")
            # Return FAILURE so BatchSpanProcessor knows to retry
            return SpanExportResult.FAILURE
    
    def shutdown(self) -> None:
        if hasattr(self._exporter, 'shutdown'):
            self._exporter.shutdown()
    
    def force_flush(self, timeout_millis: int = 30000) -> bool:
        if hasattr(self._exporter, 'force_flush'):
            return self._exporter.force_flush(timeout_millis)
        return True


def init_observability(app):
    if os.getenv("ENABLE_OBSERVABILITY", "true").lower() != "true":
        print("⚠️ Observability disabled by environment variable.")
        return
    
    try:
        service_name = os.getenv("SERVICE_NAME", "medrag-backend")
        resource = Resource.create({"service.name": service_name})

        provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(provider)

        # Allow switching protocol via env:
        #  - OTLP_PROTOCOL=http|grpc (default http)
        #  - OTLP_HTTP_ENDPOINT (default http://localhost:4318/v1/traces)
        #  - OTLP_GRPC_ENDPOINT (default localhost:4317)
        protocol = os.getenv("OTLP_PROTOCOL", "http").lower()
        http_endpoint = os.getenv("OTLP_HTTP_ENDPOINT", "http://localhost:4318/v1/traces")
        grpc_endpoint = os.getenv("OTLP_GRPC_ENDPOINT", "localhost:4317")
        
        # Create exporter with error handling
        try:
            if protocol == "grpc":
                if OTLPGRPCSpanExporter is None:
                    raise RuntimeError("gRPC exporter not available. Install opentelemetry-exporter-otlp-proto-grpc.")
                logger.info(f"🔗 Using OTLP gRPC exporter → {grpc_endpoint}")
                base_exporter = OTLPGRPCSpanExporter(endpoint=grpc_endpoint)
            else:
                logger.info(f"🔗 Using OTLP HTTP exporter → {http_endpoint}")
                base_exporter = OTLPHTTPSpanExporter(endpoint=http_endpoint)
            # Wrap exporter to suppress connection error noise
            exporter = SilentErrorSpanExporter(base_exporter)
            # BatchSpanProcessor will buffer spans and retry, making it resilient to connection issues
            # Try to use optimized parameters, but fall back to defaults if not supported
            try:
                # Try with custom parameters for faster export
                span_processor = BatchSpanProcessor(
                    exporter,
                    max_queue_size=512,  # Default is 2048
                    export_timeout_millis=30000,  # Default is 30000 (30s)
                    schedule_delay_millis=5000,  # Export every 5 seconds (default is 5000)
                )
            except TypeError:
                # Fall back to defaults if custom parameters aren't supported
                logger.debug("Using default BatchSpanProcessor parameters")
                span_processor = BatchSpanProcessor(exporter)
            provider.add_span_processor(span_processor)
            logger.info("✅ OpenTelemetry exporter configured")
            logger.info("ℹ️  Spans will be batched and exported every 5 seconds (or when batch fills)")
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize OTLP exporter: {e}. Spans will be dropped.")
            # Continue without exporter - app will still work, just no tracing
        
        # Auto-instrument frameworks and SDKs
        FastAPIInstrumentor.instrument_app(app, tracer_provider=provider)
        OpenAIInstrumentor().instrument()
        LangChainInstrumentor().instrument()
        logger.info("✅ Observability instrumentation initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize observability: {e}. Continuing without tracing.")


