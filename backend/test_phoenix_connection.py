#!/usr/bin/env python3
"""
Test script to verify Phoenix OTLP connection and trace export.
Run this to diagnose why traces aren't appearing in Phoenix.
"""

import logging
import os
import time

import pytest
from opentelemetry import trace

# This file is a diagnostics script, but pytest collects it as a unit test.
# Make it optional so local/unit test runs don't fail if OTLP exporter deps
# aren't installed (or Phoenix isn't running).
try:  # pragma: no cover
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
except Exception as exc:  # pragma: no cover
    OTLPSpanExporter = None  # type: ignore[assignment]
    _otlp_import_error = exc
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExportResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_phoenix_connection():
    """Test connecting to Phoenix and exporting a test trace."""

    if os.getenv("RUN_PHOENIX_TESTS", "").strip() != "1":
        pytest.skip("Phoenix/OTLP connectivity test is disabled (set RUN_PHOENIX_TESTS=1 to enable).")

    if OTLPSpanExporter is None:  # pragma: no cover
        pytest.skip(f"OTLP exporter not available: {_otlp_import_error}")
    
    endpoint = os.getenv("OTLP_HTTP_ENDPOINT", "http://localhost:6006/v1/traces")
    service_name = os.getenv("SERVICE_NAME", "medrag-backend")
    
    logger.info(f"🔍 Testing Phoenix connection...")
    logger.info(f"   Endpoint: {endpoint}")
    logger.info(f"   Service Name: {service_name}")
    
    # Create resource
    resource = Resource.create({"service.name": service_name})
    
    # Create tracer provider
    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)
    
    # Create exporter
    try:
        exporter = OTLPSpanExporter(endpoint=endpoint)
        logger.info(f"✅ OTLP exporter created successfully")
    except Exception as e:
        logger.error(f"❌ Failed to create exporter: {e}")
        return False
    
    # Add batch processor
    processor = BatchSpanProcessor(exporter, schedule_delay_millis=1000)  # 1 second delay for testing - Note: 'schedule' not 'scheduled'
    provider.add_span_processor(processor)
    
    # Create a test trace
    tracer = trace.get_tracer("test.tracer")
    logger.info("📊 Creating test span...")
    
    with tracer.start_as_current_span("test.operation") as span:
        span.set_attribute("test.attribute", "test_value")
        span.set_attribute("test.number", 42)
        span.add_event("test.event", {"message": "This is a test event"})
        logger.info("✅ Test span created")
    
    # Force flush
    logger.info("🔄 Flushing spans...")
    try:
        provider.force_flush(timeout_millis=5000)
        logger.info("✅ Force flush completed")
    except Exception as e:
        logger.warning(f"⚠️ Force flush had issues: {e}")
    
    # Wait a bit for export
    logger.info("⏳ Waiting 2 seconds for export to complete...")
    time.sleep(2)
    
    logger.info("✨ Test complete!")
    logger.info("📋 Check Phoenix UI at http://localhost:6006")
    logger.info("   Look for a trace from service: 'medrag-backend'")
    logger.info("   Operation name: 'test.operation'")
    
    return True

if __name__ == "__main__":
    test_phoenix_connection()

