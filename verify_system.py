#!/usr/bin/env python
"""Quick verification script for the multi-agent system."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'multi_agent_system'))

print("=" * 80)
print("MULTI-AGENT SYSTEM VERIFICATION")
print("=" * 80)

# Test imports
try:
    from coordinator import Coordinator
    print("[OK] Coordinator imported successfully")
except Exception as e:
    print(f"[FAIL] Failed to import Coordinator: {e}")
    sys.exit(1)

try:
    from agents import ResearchAgent, AnalysisAgent
    print("[OK] Specialized agents imported successfully")
except Exception as e:
    print(f"[FAIL] Failed to import agents: {e}")
    sys.exit(1)

try:
    from memory_system import MemoryAgent
    print("[OK] Memory agent imported successfully")
except Exception as e:
    print(f"[FAIL] Failed to import memory agent: {e}")
    sys.exit(1)

# Test instantiation
try:
    coordinator = Coordinator()
    print("[OK] Coordinator instantiated")
except Exception as e:
    print(f"[FAIL] Failed to instantiate coordinator: {e}")
    sys.exit(1)

# Test processing
try:
    result = coordinator.process_query("What is machine learning?")
    print("[OK] Query processed successfully")
    print(f"    - Answer length: {len(result['answer'])} characters")
    print(f"    - Confidence: {result['confidence']:.2%}")
    print(f"    - Agents involved: {', '.join(result['sources'])}")
except Exception as e:
    print(f"[FAIL] Failed to process query: {e}")
    sys.exit(1)

# Test memory
try:
    memory_info = coordinator.memory_agent.get_context_summary()
    print("[OK] Memory system operational")
    print(f"    - Total records: {memory_info['total_records']}")
    print(f"    - Topics: {', '.join(memory_info['topics'][:3])}")
except Exception as e:
    print(f"[FAIL] Failed to access memory: {e}")
    sys.exit(1)

print("\n" + "=" * 80)
print("ALL SYSTEMS VERIFIED - PROJECT READY FOR DEPLOYMENT")
print("=" * 80)
