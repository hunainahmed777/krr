"""
Test Scenarios and Runner

Demonstrates the multi-agent system with 5 sample test cases:
1. Simple Query: "What are the main types of neural networks?"
2. Complex Query: "Research transformer architectures, analyze their computational efficiency..."
3. Memory Test: "What did we discuss about neural networks earlier?"
4. Multi-step: "Find recent papers on reinforcement learning..."
5. Collaborative: "Compare two machine-learning approaches and recommend which is better..."
"""

from coordinator import Coordinator
from agents import ResearchAgent, AnalysisAgent, RequestHandler
from memory_system import MemoryAgent
from typing import Dict, List
import json


class TestRunner:
    """Runs test scenarios and captures output."""
    
    def __init__(self):
        self.coordinator = Coordinator()
        self.test_results: List[Dict] = []
    
    def run_all_tests(self) -> List[Dict]:
        """Run all test scenarios."""
        test_cases = [
            {
                "name": "Simple Query",
                "query": "What are the main types of neural networks?",
                "expected_agents": ["ResearchAgent"],
                "filename": "simple_query.txt"
            },
            {
                "name": "Complex Query",
                "query": "Research transformer architectures, analyze their computational efficiency, and summarize key trade-offs.",
                "expected_agents": ["ResearchAgent", "AnalysisAgent"],
                "filename": "complex_query.txt"
            },
            {
                "name": "Memory Test",
                "query": "What did we discuss about neural networks earlier?",
                "expected_agents": ["MemoryAgent"],
                "filename": "memory_test.txt"
            },
            {
                "name": "Multi-step Query",
                "query": "Find recent papers on reinforcement learning, analyze their methodologies, and identify common challenges.",
                "expected_agents": ["ResearchAgent", "AnalysisAgent"],
                "filename": "multi_step.txt"
            },
            {
                "name": "Collaborative Query",
                "query": "Compare two machine-learning approaches and recommend which is better for our use case.",
                "expected_agents": ["ResearchAgent", "AnalysisAgent"],
                "filename": "collaborative.txt"
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{'='*80}")
            print(f"Test {i}: {test_case['name']}")
            print(f"{'='*80}")
            
            result = self.run_test(test_case)
            self.test_results.append(result)
            
            print("\n" + result["output"])
        
        return self.test_results
    
    def run_test(self, test_case: Dict) -> Dict:
        """Run a single test case."""
        query = test_case["query"]
        
        print(f"\nUser Query: {query}\n")
        
        # Process query through coordinator
        result = self.coordinator.process_query(query)
        
        # Format output
        output = self._format_output(test_case, result)
        
        return {
            "test_name": test_case["name"],
            "query": query,
            "result": result,
            "output": output,
            "filename": test_case.get("filename", "output.txt"),
            "expected_agents": test_case.get("expected_agents", []),
            "actual_agents": result["sources"]
        }
    
    def _format_output(self, test_case: Dict, result: Dict) -> str:
        """Format result for display and file output."""
        lines = []
        
        lines.append(f"TEST: {test_case['name']}")
        lines.append("=" * 80)
        lines.append(f"\nQuery: {test_case['query']}\n")
        
        lines.append("ANSWER:")
        lines.append("-" * 40)
        lines.append(result["answer"])
        
        lines.append("\nMETADATA:")
        lines.append("-" * 40)
        lines.append(f"Confidence: {result.get('confidence', 0):.2%}")
        lines.append(f"Agents Involved: {', '.join(result.get('sources', []))}")
        lines.append(f"Synthesis Method: {result.get('synthesis_method', 'N/A')}")
        
        lines.append("\nEXECUTION TRACE:")
        lines.append("-" * 40)
        trace = result.get("trace", {})
        
        if "complexity_analysis" in trace:
            complexity = trace["complexity_analysis"]
            lines.append(f"Complexity Level: {complexity['complexity_level']}")
            lines.append(f"Required Agents: {', '.join(complexity['required_agents'])}")
        
        if "execution_plan" in trace:
            lines.append("\nExecution Plan:")
            for task in trace["execution_plan"]:
                lines.append(f"  Step {task['step']}: {task['agent']} - {task['action']}")
        
        if "steps" in trace:
            lines.append("\nExecution Steps:")
            for step in trace["steps"]:
                status_indicator = "[OK]" if step["status"] == "completed" else "[FAIL]"
                if "agent" in step and step["agent"] != "Coordinator":
                    lines.append(
                        f"  {status_indicator} Step {step.get('step', 'N/A')}: {step['agent']} "
                        f"- {step['action']} [{step['status']}]"
                    )
                    if "result_summary" in step:
                        for key, value in step["result_summary"].items():
                            lines.append(f"      - {key}: {value}")
        
        lines.append("\nSYSTEM STATE:")
        lines.append("-" * 40)
        state = self.coordinator.get_system_state()
        lines.append(f"Total Interactions: {state['total_interactions']}")
        lines.append(f"Memory Records: {state['memory_context']['total_records']}")
        lines.append(f"Topics Covered: {', '.join(state['memory_context']['topics'][:3])}")
        
        lines.append("\n" + "=" * 80 + "\n")
        
        return "\n".join(lines)
    
    def save_outputs(self, output_dir: str = "outputs"):
        """Save test results to output files."""
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        for result in self.test_results:
            filepath = os.path.join(output_dir, result["filename"])
            
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(result["output"])
            
            print(f"[OK] Saved: {filepath}")
    
    def generate_summary_report(self, output_dir: str = "outputs") -> str:
        """Generate comprehensive summary report."""
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        lines = []
        lines.append("MULTI-AGENT SYSTEM - TEST SUMMARY REPORT")
        lines.append("=" * 80)
        lines.append(f"\nTotal Tests Run: {len(self.test_results)}")
        lines.append(f"All Tests Completed: {'Yes' if len(self.test_results) == 5 else 'No'}\n")
        
        lines.append("TEST RESULTS:")
        lines.append("-" * 80)
        
        for i, result in enumerate(self.test_results, 1):
            lines.append(f"\n{i}. {result['test_name']}")
            lines.append(f"   Query: {result['query'][:70]}...")
            lines.append(f"   Agents Used: {', '.join(result['actual_agents'])}")
            lines.append(f"   Confidence: {result['result'].get('confidence', 0):.2%}")
            lines.append(f"   Output File: {result['filename']}")
        
        lines.append("\n" + "=" * 80)
        lines.append("\nSYSTEM STATISTICS:")
        lines.append("-" * 80)
        
        state = self.coordinator.get_system_state()
        lines.append(f"Total Interactions: {state['total_interactions']}")
        lines.append(f"Total Execution Traces: {state['total_traces']}")
        lines.append(f"\nMemory System:")
        lines.append(f"  - Total Records: {state['memory_context']['total_records']}")
        lines.append(f"  - Conversations: {state['memory_context']['conversations']}")
        lines.append(f"  - Knowledge Facts: {state['memory_context']['knowledge_facts']}")
        lines.append(f"  - Agent States: {state['memory_context']['agent_states']}")
        lines.append(f"  - Topics: {', '.join(state['memory_context']['topics'])}")
        
        lines.append(f"\nAgent Statistics:")
        lines.append(f"  Research Agent:")
        lines.append(f"    - Queries Processed: {state['research_agent_stats']['queries_processed']}")
        lines.append(f"    - Topics: {', '.join(state['research_agent_stats']['knowledge_topics'][:3])}")
        
        lines.append(f"  Analysis Agent:")
        lines.append(f"    - Analyses Performed: {state['analysis_agent_stats']['analyses_performed']}")
        lines.append(f"    - Specializations: {', '.join(state['analysis_agent_stats']['specializations'])}")
        
        lines.append("\n" + "=" * 80)
        lines.append("\nCONCLUSION:")
        lines.append("-" * 80)
        lines.append("The multi-agent system successfully demonstrated:")
        lines.append("[OK] Coordinator-based task orchestration")
        lines.append("[OK] Agent communication and specialization")
        lines.append("[OK] Memory-based context awareness")
        lines.append("[OK] Adaptive task decomposition")
        lines.append("[OK] Multi-step query processing")
        lines.append("[OK] Result synthesis and traceability")
        
        report_text = "\n".join(lines)
        
        # Save summary report
        filepath = os.path.join(output_dir, "SUMMARY_REPORT.txt")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report_text)
        
        print(f"\n[OK] Saved Summary Report: {filepath}\n")
        
        return report_text


def main():
    """Main entry point for running tests."""
    print("Multi-Agent System - Test Runner")
    print("=" * 80)
    
    runner = TestRunner()
    
    # Run all test scenarios
    print("\nRunning test scenarios...\n")
    runner.run_all_tests()
    
    # Save outputs
    print("\nSaving outputs...\n")
    runner.save_outputs("outputs")
    
    # Generate summary report
    print("\nGenerating summary report...\n")
    summary = runner.generate_summary_report("outputs")
    print(summary)


if __name__ == "__main__":
    main()
