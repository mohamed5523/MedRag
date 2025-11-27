import sys
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class _Span:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def set_attribute(self, *args, **kwargs):
        return None

    def add_event(self, *args, **kwargs):
        return None

    def record_exception(self, *args, **kwargs):
        return None


class _Tracer:
    def start_as_current_span(self, *args, **kwargs):
        return _Span()


class _TracerProvider:
    def force_flush(self, *args, **kwargs):
        return True


if "opentelemetry" not in sys.modules:
    otel_module = types.ModuleType("opentelemetry")
    trace_module = types.ModuleType("trace")

    def _get_tracer(*args, **kwargs):
        return _Tracer()

    def _get_current_span():
        return _Span()

    def _get_tracer_provider():
        return _TracerProvider()

    trace_module.get_tracer = _get_tracer
    trace_module.get_current_span = _get_current_span
    trace_module.get_tracer_provider = _get_tracer_provider

    otel_module.trace = trace_module

    sys.modules["opentelemetry"] = otel_module
    sys.modules["opentelemetry.trace"] = trace_module

