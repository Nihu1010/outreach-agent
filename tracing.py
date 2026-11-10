"""Phoenix tracing. Optional: the app runs without it.

Turn it on by setting PHOENIX_COLLECTOR_ENDPOINT (and PHOENIX_API_KEY for Phoenix Cloud).
"""
import os
from contextlib import contextmanager

from opentelemetry import trace

_tracer = trace.get_tracer("outreach-agent")  # no-op until a provider is registered
_enabled = False


def init_tracing() -> bool:
    global _enabled, _tracer
    if _enabled:
        return True
    if not os.getenv("PHOENIX_COLLECTOR_ENDPOINT"):
        return False
    from phoenix.otel import register

    provider = register(project_name=os.getenv("PHOENIX_PROJECT", "outreach-agent"))
    try:
        from openinference.instrumentation.anthropic import AnthropicInstrumentor
        AnthropicInstrumentor().instrument(tracer_provider=provider)
    except ImportError:
        pass
    try:
        from openinference.instrumentation.openai import OpenAIInstrumentor
        OpenAIInstrumentor().instrument(tracer_provider=provider)
    except ImportError:
        pass
    _tracer = trace.get_tracer("outreach-agent")
    _enabled = True
    return True


@contextmanager
def span(name: str, kind: str = "CHAIN", **attrs):
    """One step of the agent. kind = CHAIN (a whole run) or TOOL (one tool call)."""
    with _tracer.start_as_current_span(name) as s:
        s.set_attribute("openinference.span.kind", kind)
        for k, v in attrs.items():
            s.set_attribute(k, str(v))
        yield s
