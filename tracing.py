"""Phoenix tracing. Optional: the app runs without it.

Turn it on by setting PHOENIX_COLLECTOR_ENDPOINT (and PHOENIX_API_KEY for Phoenix Cloud).
Demo traffic goes to PHOENIX_PROJECT (default "outreach-agent").
Eval runs go to PHOENIX_EVAL_PROJECT (default "outreach-agent-evals") via use_project().
"""
import os
from contextlib import contextmanager

from opentelemetry import trace

_tracer = trace.get_tracer("outreach-agent")  # no-op until a provider is registered
_enabled = False
_providers = {}      # project name -> tracer provider
_current = None      # project name currently receiving traces


def _instrumentors():
    out = []
    try:
        from openinference.instrumentation.anthropic import AnthropicInstrumentor
        out.append(AnthropicInstrumentor())
    except ImportError:
        pass
    try:
        from openinference.instrumentation.openai import OpenAIInstrumentor
        out.append(OpenAIInstrumentor())
    except ImportError:
        pass
    return out


def _provider(project: str):
    if project not in _providers:
        from phoenix.otel import register
        _providers[project] = register(project_name=project, set_global_tracer_provider=False, verbose=False)
    return _providers[project]


def _switch(project: str):
    """Send agent spans and auto-traced LLM calls to this project."""
    global _tracer, _current
    provider = _provider(project)
    for inst in _instrumentors():
        if inst.is_instrumented_by_opentelemetry:
            inst.uninstrument()
        inst.instrument(tracer_provider=provider)
    _tracer = provider.get_tracer("outreach-agent")
    _current = project


def init_tracing() -> bool:
    global _enabled
    if _enabled:
        return True
    if not os.getenv("PHOENIX_COLLECTOR_ENDPOINT"):
        return False
    _switch(os.getenv("PHOENIX_PROJECT", "outreach-agent"))
    _enabled = True
    return True


@contextmanager
def use_project(project: str = ""):
    """Temporarily trace into another project (used by eval runs), then switch back."""
    if not _enabled:
        yield
        return
    previous = _current
    _switch(project or os.getenv("PHOENIX_EVAL_PROJECT", "outreach-agent-evals"))
    try:
        yield
    finally:
        for p in _providers.values():
            p.force_flush()
        _switch(previous)


@contextmanager
def span(name: str, kind: str = "CHAIN", **attrs):
    """One step of the agent. kind = CHAIN (a whole run), TOOL (one step) or RETRIEVER (search)."""
    with _tracer.start_as_current_span(name) as s:
        s.set_attribute("openinference.span.kind", kind)
        for k, v in attrs.items():
            s.set_attribute(k, str(v))
        yield s
