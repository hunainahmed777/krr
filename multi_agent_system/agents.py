"""
Specialized Agent Implementations

Includes:
- ResearchAgent: Information retrieval from knowledge base
- AnalysisAgent: Comparisons, reasoning, and calculations
- Base Agent: Common functionality
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json


@dataclass
class Message:
    """Inter-agent message."""
    sender: str
    receiver: str
    content: str
    message_type: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "content": self.content,
            "message_type": self.message_type,
            "metadata": self.metadata
        }


# Pre-loaded knowledge base for research agent
KNOWLEDGE_BASE = {
    "neural_networks": [
        "CNN (Convolutional Neural Networks): Specialized for image processing, excels at feature extraction through convolutional layers and pooling.",
        "RNN (Recurrent Neural Networks): Designed for sequential data, maintains memory through hidden states, good for time series and NLP.",
        "LSTM (Long Short-Term Memory): Advanced RNN variant that solves vanishing gradient problem, crucial for long-term dependencies.",
        "GRU (Gated Recurrent Unit): Simplified LSTM variant with fewer parameters, faster training but similar performance.",
        "Autoencoders: Unsupervised learning for dimensionality reduction and feature learning, symmetric architecture.",
        "Transformers: State-of-the-art for NLP, uses self-attention mechanism, parallelizable, excellent for large-scale language models.",
        "GANs (Generative Adversarial Networks): Generative model with adversarial training between generator and discriminator."
    ],
    "transformers": [
        "BERT (Bidirectional Encoder Representations from Transformers): Bidirectional pre-training, excellent for understanding context.",
        "GPT (Generative Pre-trained Transformers): Unidirectional decoder, excellent for text generation tasks.",
        "Vision Transformers (ViT): Apply transformer architecture to image classification by treating images as sequences of patches.",
        "T5 (Text-to-Text Transfer Transformer): Unified framework treating all NLP tasks as text-to-text generation.",
        "Transformer efficiency: Computational complexity O(n²) for sequence length n, requires optimization for long sequences.",
        "Attention mechanism: Transformer core, allows direct modeling of dependencies between distant elements.",
        "Trade-offs: High accuracy but computationally expensive, requires large amounts of training data, prone to overfitting on small datasets."
    ],
    "optimization_techniques": [
        "Gradient Descent: Foundational optimization, iteratively adjusts parameters in direction of steepest descent.",
        "Stochastic Gradient Descent (SGD): Uses mini-batches instead of full dataset, faster and often finds better minima.",
        "Adam Optimizer: Combines momentum with adaptive learning rates, currently most popular, often outperforms SGD.",
        "RMSprop: Adaptive learning rate method, works well for non-stationary problems.",
        "AdaGrad: Adapts learning rate based on historical gradients, good for sparse data.",
        "Momentum: Accelerates convergence by accumulating gradients, helps escape local minima.",
        "Learning rate scheduling: Gradually reducing learning rate improves final model quality."
    ],
    "reinforcement_learning": [
        "Q-Learning: Model-free RL, learns action-value function through temporal difference learning, fundamental for deep RL.",
        "Policy Gradient: Directly optimizes policy, includes REINFORCE and Actor-Critic methods.",
        "Deep Q-Networks (DQN): Combines Q-learning with deep learning, breakthrough in game playing AI.",
        "Policy Gradient Methods: Direct policy optimization, includes REINFORCE, A3C, PPO.",
        "Actor-Critic Methods: Combines policy gradient with value function, more stable training.",
        "PPO (Proximal Policy Optimization): State-of-the-art policy gradient, easier to tune than TRPO.",
        "Common challenges: Exploration-exploitation tradeoff, sample efficiency, credit assignment, non-stationarity from learning."
    ],
    "ml_approaches": [
        "Supervised Learning: Learns from labeled data, includes regression and classification tasks.",
        "Unsupervised Learning: Discovers patterns in unlabeled data, includes clustering and dimensionality reduction.",
        "Semi-supervised Learning: Combines small labeled dataset with large unlabeled dataset for improved performance.",
        "Transfer Learning: Leverages knowledge from one domain to improve performance in related domain.",
        "Meta-Learning: Learning to learn, enables quick adaptation to new tasks with few examples.",
        "Federated Learning: Decentralized training preserving privacy, increasingly important for distributed systems."
    ]
}


class BaseAgent:
    """Base class for all agents."""
    
    def __init__(self, name: str):
        self.name = name
        self.message_log: List[Message] = []
        self.confidence_scores: Dict[str, float] = {}
    
    def log_message(self, message: Message):
        """Log incoming message."""
        self.message_log.append(message)
    
    def create_message(self, receiver: str, content: str, 
                      message_type: str = "task",
                      metadata: Optional[Dict] = None) -> Message:
        """Create a message to another agent."""
        return Message(
            sender=self.name,
            receiver=receiver,
            content=content,
            message_type=message_type,
            metadata=metadata or {}
        )


class ResearchAgent(BaseAgent):
    """
    Simulates information retrieval from a pre-loaded knowledge base.
    Handles research queries and returns findings with confidence scores.
    """
    
    def __init__(self):
        super().__init__("ResearchAgent")
        self.knowledge_base = KNOWLEDGE_BASE
        self.queries_processed = 0
    
    def search_knowledge(self, query: str, topic: str = None) -> Dict[str, Any]:
        """
        Search the knowledge base for relevant information.
        
        Args:
            query: Natural language query
            topic: Optional topic to constrain search
        
        Returns:
            Dictionary with findings, confidence, and metadata
        """
        self.queries_processed += 1
        
        # Normalize query
        query_lower = query.lower()
        
        # Determine relevant topic if not provided
        if not topic:
            topic = self._infer_topic(query_lower)
        
        topic_lower = topic.lower()
        
        # Fetch relevant information
        results = []
        if topic_lower in self.knowledge_base:
            # Get all facts for this topic
            facts = self.knowledge_base[topic_lower]
            
            # Filter by relevance to query
            relevant_facts = []
            for fact in facts:
                if self._matches_query(query_lower, fact.lower()):
                    relevant_facts.append(fact)
            
            # If no exact match, return all for the topic
            if not relevant_facts:
                relevant_facts = facts
            
            results = relevant_facts
        else:
            # Cross-topic search
            for topic_key, facts in self.knowledge_base.items():
                for fact in facts:
                    if self._matches_query(query_lower, fact.lower()):
                        results.append(fact)
        
        # Calculate confidence based on match quality
        confidence = min(0.95, 0.7 + (len(results) / 10) * 0.25) if results else 0.3
        
        return {
            "query": query,
            "topic": topic,
            "findings": results[:5],  # Top 5 results
            "count": len(results),
            "confidence": confidence,
            "source": "knowledge_base",
            "processed": self.queries_processed
        }
    
    def _infer_topic(self, query: str) -> str:
        """Infer the topic from query keywords."""
        topic_keywords = {
            "neural_networks": ["neural", "network", "cnn", "rnn", "lstm", "gru", "autoencoder"],
            "transformers": ["transformer", "bert", "gpt", "attention", "vision"],
            "optimization_techniques": ["optimization", "optimizer", "gradient", "adam", "sgd"],
            "reinforcement_learning": ["reinforcement", "rl", "q-learning", "policy", "actor-critic"],
            "ml_approaches": ["supervised", "unsupervised", "transfer", "meta", "federated"]
        }
        
        for topic, keywords in topic_keywords.items():
            if any(kw in query for kw in keywords):
                return topic
        
        return "neural_networks"  # Default topic
    
    def _matches_query(self, query: str, text: str) -> bool:
        """Check if text matches query keywords."""
        keywords = query.split()
        return any(kw in text for kw in keywords if len(kw) > 2)
    
    def get_research_summary(self) -> Dict[str, Any]:
        """Get summary of research conducted."""
        return {
            "agent": self.name,
            "queries_processed": self.queries_processed,
            "total_knowledge_facts": sum(len(v) for v in self.knowledge_base.values()),
            "knowledge_topics": list(self.knowledge_base.keys())
        }


class AnalysisAgent(BaseAgent):
    """
    Performs comparisons, reasoning, and simple calculations on retrieved data.
    Synthesizes research findings into actionable insights.
    """
    
    def __init__(self):
        super().__init__("AnalysisAgent")
        self.analyses_performed = 0
    
    def compare_items(self, items: List[str], criteria: str = None) -> Dict[str, Any]:
        """
        Compare multiple items based on given criteria.
        
        Args:
            items: List of items to compare
            criteria: Comparison criteria (e.g., "effectiveness", "efficiency")
        
        Returns:
            Dictionary with comparison analysis and recommendations
        """
        self.analyses_performed += 1
        criteria = criteria or "general characteristics"
        
        comparison = {
            "items": items,
            "criteria": criteria,
            "analysis": self._perform_comparison(items, criteria),
            "recommendation": self._generate_recommendation(items, criteria),
            "confidence": 0.85,
            "performed": self.analyses_performed
        }
        
        return comparison
    
    def analyze_trade_offs(self, options: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Analyze trade-offs between different options.
        
        Args:
            options: Dictionary mapping option names to characteristics
        
        Returns:
            Dictionary with trade-off analysis
        """
        self.analyses_performed += 1
        
        analysis = {
            "options": options,
            "trade_offs": self._identify_tradeoffs(options),
            "summary": self._summarize_tradeoffs(options),
            "confidence": 0.8,
            "performed": self.analyses_performed
        }
        
        return analysis
    
    def identify_patterns(self, data: List[str]) -> Dict[str, Any]:
        """
        Identify patterns and themes in data.
        
        Args:
            data: List of data points to analyze
        
        Returns:
            Dictionary with identified patterns
        """
        self.analyses_performed += 1
        
        patterns = {
            "data_points": len(data),
            "patterns": self._extract_patterns(data),
            "themes": self._extract_themes(data),
            "confidence": 0.75,
            "performed": self.analyses_performed
        }
        
        return patterns
    
    def _perform_comparison(self, items: List[str], criteria: str) -> List[str]:
        """Generate comparison analysis."""
        comparisons = []
        for i, item in enumerate(items[:3]):  # Compare first 3
            if i == 0:
                comparisons.append(f"{item} serves as baseline for {criteria}.")
            else:
                comparisons.append(f"{item} offers distinct advantages/trade-offs compared to baseline in {criteria}.")
        return comparisons
    
    def _generate_recommendation(self, items: List[str], criteria: str) -> str:
        """Generate recommendation based on comparison."""
        if not items:
            return "Insufficient data for recommendation."
        
        best_item = items[0]
        reasoning = f"Based on {criteria}, {best_item} appears to be the most suitable choice."
        
        if len(items) > 1:
            reasoning += f" However, consider {items[1]} as an alternative depending on specific constraints."
        
        return reasoning
    
    def _identify_tradeoffs(self, options: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Identify trade-offs between options."""
        tradeoffs = {}
        for option, chars in options.items():
            tradeoffs[option] = {
                "strengths": chars[0] if chars else "Unknown",
                "weaknesses": chars[1] if len(chars) > 1 else "Unknown"
            }
        return tradeoffs
    
    def _summarize_tradeoffs(self, options: Dict[str, List[str]]) -> str:
        """Generate summary of trade-offs."""
        if len(options) < 2:
            return "Need at least 2 options to identify trade-offs."
        
        summary = f"Analysis of {len(options)} options reveals distinct trade-offs. "
        summary += "Each option excels in specific dimensions while having limitations in others. "
        summary += "Choose based on priority and constraints."
        
        return summary
    
    def _extract_patterns(self, data: List[str]) -> List[str]:
        """Extract patterns from data."""
        patterns = []
        if data:
            patterns.append(f"Data contains {len(data)} elements.")
            
            # Look for common keywords
            all_text = " ".join(data).lower()
            if "optimization" in all_text:
                patterns.append("Pattern: Performance optimization is a recurring theme.")
            if "learning" in all_text:
                patterns.append("Pattern: Learning mechanisms are emphasized across data.")
            if "efficient" in all_text or "efficiency" in all_text:
                patterns.append("Pattern: Efficiency considerations appear multiple times.")
        
        return patterns
    
    def _extract_themes(self, data: List[str]) -> List[str]:
        """Extract themes from data."""
        themes = []
        all_text = " ".join(data).lower()
        
        theme_keywords = {
            "Scalability": ["scalable", "scale", "large"],
            "Efficiency": ["efficient", "fast", "speed"],
            "Accuracy": ["accurate", "performance"],
            "Usability": ["simple", "easy", "practical"]
        }
        
        for theme, keywords in theme_keywords.items():
            if any(kw in all_text for kw in keywords):
                themes.append(theme)
        
        return themes or ["General technical domain"]
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of analyses performed."""
        return {
            "agent": self.name,
            "analyses_performed": self.analyses_performed,
            "specializations": ["comparisons", "trade-off analysis", "pattern identification"]
        }


class RequestHandler:
    """Handles message passing and routing between agents."""
    
    def __init__(self):
        self.message_queue: List[Message] = []
        self.processed_messages: List[Message] = []
    
    def send_message(self, message: Message) -> bool:
        """Queue a message for delivery."""
        self.message_queue.append(message)
        return True
    
    def get_messages_for(self, agent_name: str) -> List[Message]:
        """Get all queued messages for a specific agent."""
        messages = [m for m in self.message_queue if m.receiver == agent_name]
        self.message_queue = [m for m in self.message_queue if m.receiver != agent_name]
        return messages
    
    def process_message(self, message: Message):
        """Mark message as processed."""
        self.processed_messages.append(message)
