"""
Multi-Agent System Package
"""

__version__ = "1.0.0"
__author__ = "Multi-Agent System"
__description__ = "Minimal multi-agent system with coordinator and specialized agents"

from coordinator import Coordinator
from agents import ResearchAgent, AnalysisAgent, BaseAgent
from memory_system import MemoryAgent, SimpleVectorStore

__all__ = [
    "Coordinator",
    "ResearchAgent",
    "AnalysisAgent",
    "BaseAgent",
    "MemoryAgent",
    "SimpleVectorStore"
]
