"""Pre-register all submodules in sys.modules to work around a Python 3.14
import-machinery KeyError that occurs on Streamlit Community Cloud.

By loading each submodule via ``importlib.util.spec_from_file_location`` we
bypass the broken ``_find_and_load`` path and place the module directly in
``sys.modules``.  Subsequent ``from modules.X import Y`` statements then hit
the cache and never trigger the faulty finder.

IMPORTANT: the list below MUST be topologically sorted — every dependency has
to be pre-registered *before* the modules that import it. Otherwise a
pre-load of a dependent triggers a normal ``from modules.X import Y`` mid-flight,
which on Python 3.14 re-enters the broken finder. Its exception is swallowed by
the ``except`` below, leaving a *partially-initialized* module cached in
``sys.modules`` — e.g. ``modules.agent`` without ``agent_answer`` — which then
surfaces as ``ImportError: cannot import name 'agent_answer'``. Loading
``agent`` (and every other dependant) last guarantees all of its imports are
already cached here.
"""

import importlib
import importlib.util
import os
import sys

_pkg_dir = os.path.dirname(os.path.abspath(__file__))
_pkg_name = __name__  # "modules"

# Topologically sorted: leaf modules (no internal deps) first, dependants last.
# `agent` is intentionally last — it imports alerts, anomaly_detector,
# data_sources, knowledge_base, llm_client and network_monitor. `ai_ops` and
# `reports` are dependants of the data/monitoring layer too.
_SUBMODULES = [
    # Leaves — only external dependencies (pandas/numpy/sklearn/etc.)
    "settings",
    "ui",
    "storage",
    "network_monitor",
    "knowledge_base",
    "anomaly_detector",
    "alerts",
    "forecasting",
    "ai_ops",
    # Depend on the leaves above
    "llm_client",          # -> settings
    "data_sources",        # -> network_monitor
    "chatbot",             # -> knowledge_base, llm_client, storage
    "diagnostics_bridge",  # -> knowledge_base, llm_client, storage
    "remediation",         # -> storage
    "reports",             # -> data_sources, alerts, llm_client, remediation
    # Must be LAST: imports several of the modules above
    "agent",               # -> alerts, anomaly_detector, data_sources,
                           #    knowledge_base, llm_client, network_monitor
]

for _name in _SUBMODULES:
    _fqn = f"{_pkg_name}.{_name}"
    if _fqn in sys.modules:
        continue
    _path = os.path.join(_pkg_dir, f"{_name}.py")
    if not os.path.isfile(_path):
        continue
    try:
        _spec = importlib.util.spec_from_file_location(_fqn, _path)
        if _spec is None or _spec.loader is None:
            continue
        _mod = importlib.util.module_from_spec(_spec)
        sys.modules[_fqn] = _mod
        _spec.loader.exec_module(_mod)
        # Mirror normal package import behaviour: expose each submodule as an
        # attribute of the package (``modules.diagnostics_bridge`` etc.) so
        # attribute access and mock.patch/monkeypatch.setattr("modules.X...")
        # work exactly as with a regular package.
        setattr(sys.modules[_pkg_name], _name, _mod)
    except Exception:
        # Never leave a partially-initialized module cached — remove it so the
        # normal import path can retry (or surface a real error) instead of a
        # confusing "cannot import name '<symbol>'" later.
        sys.modules.pop(_fqn, None)
