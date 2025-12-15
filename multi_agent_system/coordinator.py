"""
Coordinator Agent (Manager)

Orchestrates the multi-agent system:
- Receives user queries
- Analyzes complexity
- Routes tasks to specialized agents
- Coordinates dependencies
- Synthesizes results
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from agents import ResearchAgent, AnalysisAgent, RequestHandler, Message
from memory_system import MemoryAgent
import json


class TaskDecomposer:
    """Decomposes complex queries into subtasks."""
    
    @staticmethod
    def analyze_complexity(query: str) -> Dict[str, Any]:
        """
        Analyze query complexity and determine agent requirements.
        
        Returns:
            Dictionary with complexity level, required agents, and task plan
        """
        query_lower = query.lower()
        
        # Keyword analysis
        keywords = {
            "research_keywords": ["find", "research", "what", "types of", "information about", "papers", "techniques"],
            "analysis_keywords": ["compare", "analyze", "efficiency", "effective", "trade-off", "recommend", "better", "which"],
            "memory_keywords": ["earlier", "discuss", "learn", "remember", "previous", "past"]
        }
        
        # Detect required agents
        required_agents = []
        complexity_score = 0
        
        if any(kw in query_lower for kw in keywords["research_keywords"]):
            required_agents.append("ResearchAgent")
            complexity_score += 1
        
        if any(kw in query_lower for kw in keywords["analysis_keywords"]):
            required_agents.append("AnalysisAgent")
            complexity_score += 2
        
        if any(kw in query_lower for kw in keywords["memory_keywords"]):
            required_agents.append("MemoryAgent")
            complexity_score += 1
        
        # If no agents detected, assume research
        if not required_agents:
            required_agents = ["ResearchAgent"]
            complexity_score = 1
        
        # Determine complexity level
        if complexity_score <= 1:
            complexity_level = "simple"
        elif complexity_score <= 2:
            complexity_level = "moderate"
        else:
            complexity_level = "complex"
        
        return {
            "complexity_level": complexity_level,
            "complexity_score": complexity_score,
            "required_agents": required_agents,
            "query": query
        }
    
    @staticmethod
    def create_execution_plan(complexity_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create execution plan with task sequencing.
        
        Returns:
            List of tasks with agent assignments and dependencies
        """
        required_agents = complexity_info["required_agents"]
        
        tasks = []
        
        # Step 1: Check memory for relevant past discussions
        if "MemoryAgent" in required_agents:
            tasks.append({
                "step": 1,
                "agent": "MemoryAgent",
                "action": "retrieve_context",
                "depends_on": [],
                "priority": 1
            })
        
        # Step 2: Conduct research
        if "ResearchAgent" in required_agents:
            depends_on = [1] if "MemoryAgent" in required_agents else []
            tasks.append({
                "step": 2,
                "agent": "ResearchAgent",
                "action": "search_knowledge",
                "depends_on": depends_on,
                "priority": 2
            })
        
        # Step 3: Perform analysis
        if "AnalysisAgent" in required_agents:
            depends_on = [2] if "ResearchAgent" in required_agents else []
            tasks.append({
                "step": 3,
                "agent": "AnalysisAgent",
                "action": "analyze",
                "depends_on": depends_on,
                "priority": 2
            })
        
        # Step 4: Store in memory
        if len(tasks) > 0:
            depends_on = [t["step"] for t in tasks]
            tasks.append({
                "step": len(tasks) + 1,
                "agent": "MemoryAgent",
                "action": "store_knowledge",
                "depends_on": depends_on,
                "priority": 3
            })
        
        return tasks


class Coordinator:
    """
    Manager agent that orchestrates the multi-agent system.
    
    Responsibilities:
    - Receive and understand user queries
    - Decompose tasks and create execution plans
    - Route tasks to specialized agents
    - Coordinate dependencies
    - Synthesize and present results
    - Maintain system state and context
    """
    
    def __init__(self, research_agent: ResearchAgent = None,
                 analysis_agent: AnalysisAgent = None,
                 memory_agent: MemoryAgent = None,
                 request_handler: RequestHandler = None):
        self.name = "Coordinator"
        self.research_agent = research_agent or ResearchAgent()
        self.analysis_agent = analysis_agent or AnalysisAgent()
        self.memory_agent = memory_agent or MemoryAgent()
        self.request_handler = request_handler or RequestHandler()
        
        self.task_decomposer = TaskDecomposer()
        self.interaction_history: List[Dict[str, Any]] = []
        self.execution_traces: List[Dict[str, Any]] = []
    
    def process_query(self, user_query: str) -> Dict[str, Any]:
        """
        Process a user query through the multi-agent system.
        
        Args:
            user_query: Natural language user question
        
        Returns:
            Dictionary with answer, reasoning, and trace information
        """
        trace = {
            "query": user_query,
            "timestamp": datetime.now().isoformat(),
            "steps": []
        }
        
        # Step 1: Store query in conversation memory
        self.memory_agent.store_conversation(
            content=user_query,
            topic=self._extract_topic(user_query),
            confidence=1.0
        )
        trace["steps"].append({
            "step": 0,
            "action": "store_conversation",
            "agent": "MemoryAgent",
            "status": "completed"
        })
        
        # Step 2: Analyze query complexity and create plan
        complexity_info = self.task_decomposer.analyze_complexity(user_query)
        execution_plan = self.task_decomposer.create_execution_plan(complexity_info)
        
        trace["complexity_analysis"] = complexity_info
        trace["execution_plan"] = execution_plan
        
        # Step 3: Execute plan
        results = self._execute_plan(user_query, execution_plan, trace)
        
        # Step 4: Synthesize final answer
        final_answer = self._synthesize_answer(user_query, results)
        
        trace["steps"].append({
            "step": "final",
            "action": "synthesize_answer",
            "agent": "Coordinator",
            "status": "completed",
            "result_summary": {
                "answer_length": len(final_answer.get("answer", "")),
                "agents_involved": len(results)
            }
        })
        
        # Store interaction history
        interaction_record = {
            "query": user_query,
            "answer": final_answer["answer"],
            "agents_involved": list(results.keys()),
            "timestamp": datetime.now().isoformat(),
            "confidence": final_answer.get("confidence", 0.8)
        }
        self.interaction_history.append(interaction_record)
        self.execution_traces.append(trace)
        
        # Add tracing information
        final_answer["trace"] = trace
        final_answer["interaction_count"] = len(self.interaction_history)
        
        return final_answer
    
    def _execute_plan(self, query: str, plan: List[Dict], trace: Dict) -> Dict[str, Any]:
        """Execute the task plan and collect results."""
        results = {}
        completed_steps = set()
        
        for task in sorted(plan, key=lambda x: x["step"]):
            step_num = task["step"]
            agent_name = task["agent"]
            action = task["action"]
            depends_on = task["depends_on"]
            
            # Check dependencies
            if depends_on and not all(d in completed_steps for d in depends_on):
                continue
            
            step_trace = {
                "step": step_num,
                "agent": agent_name,
                "action": action,
                "status": "running"
            }
            
            try:
                # Route to appropriate agent
                if agent_name == "MemoryAgent":
                    result = self._run_memory_agent(query, action, results)
                elif agent_name == "ResearchAgent":
                    result = self._run_research_agent(query, results)
                elif agent_name == "AnalysisAgent":
                    result = self._run_analysis_agent(query, results)
                else:
                    result = {"error": f"Unknown agent: {agent_name}"}
                
                results[agent_name] = result
                step_trace["status"] = "completed"
                step_trace["result_summary"] = self._summarize_result(result)
                completed_steps.add(step_num)
                
            except Exception as e:
                step_trace["status"] = "failed"
                step_trace["error"] = str(e)
                result = {"error": str(e)}
                results[agent_name] = result
            
            trace["steps"].append(step_trace)
        
        return results
    
    def _run_memory_agent(self, query: str, action: str, results: Dict) -> Dict[str, Any]:
        """Run memory agent operations."""
        if action == "retrieve_context":
            topic = self._extract_topic(query)
            context = self.memory_agent.search_by_similarity(query, top_k=3)
            past_discussion = self.memory_agent.retrieve_past_discussion(topic)
            
            return {
                "action": action,
                "similar_memories": context,
                "past_discussion": past_discussion,
                "memory_summary": self.memory_agent.get_context_summary()
            }
        
        elif action == "store_knowledge":
            # Extract knowledge from results
            knowledge_content = self._extract_knowledge(results)
            
            # Determine which agent produced the best result
            source_agent = "system"
            if "AnalysisAgent" in results:
                source_agent = "AnalysisAgent"
            elif "ResearchAgent" in results:
                source_agent = "ResearchAgent"
            
            # Store in memory
            record = self.memory_agent.store_knowledge(
                content=knowledge_content,
                topic=self._extract_topic(results.get("query", "")),
                source_agent=source_agent,
                confidence=0.85
            )
            
            return {
                "action": action,
                "stored_record": record.to_dict(),
                "message": "Knowledge stored successfully"
            }
        
        return {"action": action, "status": "completed"}
    
    def _run_research_agent(self, query: str, results: Dict) -> Dict[str, Any]:
        """Run research agent."""
        topic = self._extract_topic(query)
        research_result = self.research_agent.search_knowledge(query, topic)
        
        return research_result
    
    def _run_analysis_agent(self, query: str, results: Dict) -> Dict[str, Any]:
        """Run analysis agent with research results."""
        research_findings = results.get("ResearchAgent", {}).get("findings", [])
        
        if not research_findings:
            return {"error": "No research findings to analyze"}
        
        # Detect analysis type from query
        if "compare" in query.lower() or "vs" in query.lower():
            analysis = self.analysis_agent.compare_items(
                research_findings,
                criteria=self._extract_criteria(query)
            )
        elif "trade" in query.lower():
            # Create options dict from findings
            options = {f"Option {i}": [finding] for i, finding in enumerate(research_findings[:3])}
            analysis = self.analysis_agent.analyze_trade_offs(options)
        else:
            analysis = self.analysis_agent.identify_patterns(research_findings)
        
        return analysis
    
    def _synthesize_answer(self, query: str, results: Dict) -> Dict[str, Any]:
        """Synthesize final answer from agent results."""
        answer_parts = []
        confidence_scores = []
        
        # Add memory context if available
        if "MemoryAgent" in results and results["MemoryAgent"].get("past_discussion"):
            answer_parts.append(
                f"From past discussion: {results['MemoryAgent']['past_discussion'][:200]}...\n"
            )
            confidence_scores.append(0.9)
        
        # Add research findings
        if "ResearchAgent" in results:
            research = results["ResearchAgent"]
            if research.get("findings"):
                answer_parts.append("Research Findings:\n")
                for i, finding in enumerate(research["findings"][:3], 1):
                    answer_parts.append(f"{i}. {finding}\n")
            confidence_scores.append(research.get("confidence", 0.8))
        
        # Add analysis
        if "AnalysisAgent" in results:
            analysis = results["AnalysisAgent"]
            answer_parts.append("\nAnalysis:\n")
            if analysis.get("recommendation"):
                answer_parts.append(f"Recommendation: {analysis['recommendation']}\n")
            if analysis.get("summary"):
                answer_parts.append(f"Summary: {analysis['summary']}\n")
            if analysis.get("analysis"):
                answer_parts.append("Details:\n")
                for detail in analysis.get("analysis", [])[:2]:
                    answer_parts.append(f"- {detail}\n")
            confidence_scores.append(analysis.get("confidence", 0.85))
        
        # Calculate overall confidence
        overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5
        
        return {
            "answer": "".join(answer_parts) if answer_parts else "Unable to process query.",
            "confidence": overall_confidence,
            "sources": list(results.keys()),
            "synthesis_method": "multi-agent"
        }
    
    def _extract_topic(self, text: str) -> str:
        """Extract primary topic from text."""
        topics = {
            "neural_networks": ["neural", "network", "cnn", "rnn", "lstm"],
            "transformers": ["transformer", "bert", "gpt", "attention"],
            "optimization": ["optimiz", "gradient", "adam"],
            "reinforcement_learning": ["reinforcement", "rl", "q-learning"],
            "ml_approaches": ["supervised", "unsupervised", "transfer", "learning"]
        }
        
        text_lower = text.lower()
        for topic, keywords in topics.items():
            if any(kw in text_lower for kw in keywords):
                return topic
        
        return "general_ml"
    
    def _extract_criteria(self, query: str) -> str:
        """Extract comparison criteria from query."""
        criteria_map = {
            "efficiency": ["efficient", "fast", "speed"],
            "effectiveness": ["effective", "best", "better", "compare"],
            "accuracy": ["accurate", "accuracy"],
            "scalability": ["scale", "large"]
        }
        
        query_lower = query.lower()
        for criteria, keywords in criteria_map.items():
            if any(kw in query_lower for kw in keywords):
                return criteria
        
        return "general characteristics"
    
    def _summarize_result(self, result: Dict) -> Dict[str, Any]:
        """Create a summary of agent result."""
        summary = {}
        
        if "findings" in result:
            summary["findings_count"] = len(result["findings"])
        if "confidence" in result:
            summary["confidence"] = result["confidence"]
        if "error" in result:
            summary["has_error"] = True
        if "recommendation" in result:
            summary["has_recommendation"] = True
        
        return summary
    
    def _extract_knowledge(self, results: Dict) -> str:
        """Extract key knowledge from results."""
        parts = []
        
        if "ResearchAgent" in results and results["ResearchAgent"].get("findings"):
            parts.append("Key Findings: " + "; ".join(
                results["ResearchAgent"]["findings"][:2]
            ))
        
        if "AnalysisAgent" in results and results["AnalysisAgent"].get("recommendation"):
            parts.append("Key Recommendation: " + results["AnalysisAgent"]["recommendation"])
        
        return " ".join(parts) if parts else "Analysis completed"
    
    def get_system_state(self) -> Dict[str, Any]:
        """Get current system state and statistics."""
        return {
            "coordinator": self.name,
            "total_interactions": len(self.interaction_history),
            "total_traces": len(self.execution_traces),
            "memory_context": self.memory_agent.get_context_summary(),
            "research_agent_stats": self.research_agent.get_research_summary(),
            "analysis_agent_stats": self.analysis_agent.get_analysis_summary(),
            "recent_queries": [h["query"] for h in self.interaction_history[-3:]]
        }
