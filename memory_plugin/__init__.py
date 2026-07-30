"""
Shared Memory Plugin for CI-Bench
=================================

This is the COMPLETE memory plugin system moved from mini-swe-agent
to the ROOT directory so BOTH agents can use it.

Components:
- memory_plugin.py: Core MemoryPlugin class with L1/L2/L3 retrieval
- ci_memory_system.py: Integration system for CI context
- ci_memory_llm_analysis.py: LLM-based analysis for L1
- ci_memory_l2_analysis.py: L2 consecutive analysis
- ci_memory_staged_analysis.py: Staged L3 analysis
- ci_memory_llm_analysis_multistage.py: Multi-stage LLM analysis

Usage for mini-swe-agent:
    from memory_plugin import MemoryPlugin

Usage for OpenHands:
    from memory_plugin import MemoryPlugin

Both agents use the SAME plugin, SAME logic, SAME results!
"""

__all__ = ["MemoryPlugin"]
__version__ = "1.0.0"


def __getattr__(name):
    if name == "MemoryPlugin":
        from .memory_plugin import MemoryPlugin

        return MemoryPlugin
    raise AttributeError(name)
