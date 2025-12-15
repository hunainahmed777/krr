"""
Interactive CLI for Multi-Agent System

Provides a user-friendly interface to interact with the system,
query agents, and inspect memory and execution traces.
"""

from coordinator import Coordinator
from agents import ResearchAgent, AnalysisAgent
from memory_system import MemoryAgent
import json
from typing import Optional


class MultiAgentCLI:
    """Interactive command-line interface for the multi-agent system."""
    
    def __init__(self):
        self.coordinator = Coordinator()
        self.running = True
    
    def start(self):
        """Start the interactive CLI."""
        self.print_banner()
        self.print_help()
        
        while self.running:
            try:
                user_input = input("\n> ").strip()
                
                if not user_input:
                    continue
                
                self.process_command(user_input)
            
            except KeyboardInterrupt:
                print("\n\nShutting down...")
                self.running = False
            except Exception as e:
                print(f"Error: {e}")
    
    def process_command(self, user_input: str):
        """Process user command."""
        # Check for special commands
        if user_input.lower() == "help":
            self.print_help()
        elif user_input.lower() == "exit" or user_input.lower() == "quit":
            self.running = False
            print("Goodbye!")
        elif user_input.lower() == "status":
            self.show_system_status()
        elif user_input.lower() == "memory":
            self.show_memory_stats()
        elif user_input.lower() == "history":
            self.show_interaction_history()
        elif user_input.lower() == "traces":
            self.show_execution_traces()
        elif user_input.lower().startswith("search "):
            query = user_input[7:]
            self.search_memory(query)
        elif user_input.lower() == "clear":
            import os
            os.system("cls" if os.name == "nt" else "clear")
        else:
            # Treat as user query
            self.process_query(user_input)
    
    def process_query(self, query: str):
        """Process a user query through the coordinator."""
        print("\n" + "=" * 80)
        print("Processing query...\n")
        
        result = self.coordinator.process_query(query)
        
        print("RESPONSE:")
        print("-" * 80)
        print(result["answer"])
        
        print("\nMETADATA:")
        print("-" * 40)
        print(f"Confidence: {result.get('confidence', 0):.2%}")
        print(f"Sources: {', '.join(result.get('sources', []))}")
        
        # Show trace summary
        trace = result.get("trace", {})
        if "complexity_analysis" in trace:
            print(f"Complexity: {trace['complexity_analysis']['complexity_level']}")
    
    def search_memory(self, query: str):
        """Search the memory system."""
        print("\nSearching memory for similar content...\n")
        
        results = self.coordinator.memory_agent.search_by_similarity(query, top_k=3)
        
        if results:
            for i, result in enumerate(results, 1):
                record = result["record"]
                similarity = result["similarity"]
                print(f"{i}. [{record['record_type']}] (Similarity: {similarity:.2%})")
                print(f"   Content: {record['content'][:100]}...")
                print(f"   Topic: {record['topic']}")
                print()
        else:
            print("No similar memories found.")
    
    def show_system_status(self):
        """Display system status."""
        state = self.coordinator.get_system_state()
        
        print("\n" + "=" * 80)
        print("SYSTEM STATUS")
        print("=" * 80)
        print(f"Coordinator: {state['coordinator']}")
        print(f"Total Interactions: {state['total_interactions']}")
        print(f"Total Execution Traces: {state['total_traces']}")
        print(f"\nRecent Queries:")
        for i, query in enumerate(state['recent_queries'], 1):
            print(f"  {i}. {query[:60]}...")
    
    def show_memory_stats(self):
        """Display memory statistics."""
        memory = self.coordinator.memory_agent
        context = memory.get_context_summary()
        
        print("\n" + "=" * 80)
        print("MEMORY STATISTICS")
        print("=" * 80)
        print(f"Total Records: {context['total_records']}")
        print(f"Conversations: {context['conversations']}")
        print(f"Knowledge Facts: {context['knowledge_facts']}")
        print(f"Agent States: {context['agent_states']}")
        print(f"\nTopics Covered:")
        for topic in context['topics']:
            print(f"  - {topic}")
    
    def show_interaction_history(self):
        """Show interaction history."""
        history = self.coordinator.interaction_history
        
        print("\n" + "=" * 80)
        print("INTERACTION HISTORY")
        print("=" * 80)
        
        if not history:
            print("No interactions yet.")
            return
        
        for i, interaction in enumerate(history, 1):
            print(f"{i}. {interaction['query'][:60]}...")
            print(f"   Timestamp: {interaction['timestamp']}")
            print(f"   Agents: {', '.join(interaction['agents_involved'])}")
            print(f"   Confidence: {interaction['confidence']:.2%}\n")
    
    def show_execution_traces(self):
        """Show execution traces of recent queries."""
        traces = self.coordinator.execution_traces
        
        print("\n" + "=" * 80)
        print("EXECUTION TRACES")
        print("=" * 80)
        
        if not traces:
            print("No execution traces yet.")
            return
        
        # Show last 2 traces
        for trace in traces[-2:]:
            print(f"\nQuery: {trace['query']}")
            print(f"Timestamp: {trace['timestamp']}")
            print(f"Complexity Level: {trace['complexity_analysis']['complexity_level']}")
            print(f"Steps Executed:")
            
            for step in trace['steps']:
                if 'agent' in step and step.get('agent') != 'Coordinator':
                    print(f"  - {step['agent']}: {step['action']} [{step['status']}]")
    
    @staticmethod
    def print_banner():
        """Print welcome banner."""
        print("\n" + "=" * 80)
        print("  MULTI-AGENT SYSTEM - INTERACTIVE CLI")
        print("  Coordinator-based architecture with specialized agents")
        print("=" * 80 + "\n")
    
    @staticmethod
    def print_help():
        """Print help information."""
        help_text = """
COMMANDS:
  help          - Show this help message
  exit/quit     - Exit the system
  status        - Show system status and recent queries
  memory        - Show memory statistics
  history       - Show interaction history
  traces        - Show execution traces
  search <term> - Search memory for similar content
  clear         - Clear screen
  
QUERY:
  Type any natural language question to query the system.
  Examples:
    - "What are neural networks?"
    - "Compare optimization techniques"
    - "What did we discuss earlier about transformers?"

The system will automatically route your query to appropriate agents,
maintain conversation context, and provide traceable reasoning.
        """
        print(help_text)


def main():
    """Main entry point."""
    cli = MultiAgentCLI()
    cli.start()


if __name__ == "__main__":
    main()
