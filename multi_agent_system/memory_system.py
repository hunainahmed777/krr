"""
Enhanced Memory System with Vector Search

Implements:
- Conversation Memory: Full history with timestamps
- Knowledge Base: Persistent store of facts with provenance
- Agent State Memory: Tracks agent learning and accomplishments
- Vector Search: Semantic similarity for retrieval
"""

import json
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import math


@dataclass
class MemoryRecord:
    """Structured memory record with metadata."""
    id: str
    content: str
    timestamp: str
    source_agent: str
    topic: str
    confidence: float
    record_type: str  # 'conversation', 'knowledge', 'agent_state'
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        return asdict(self)


class SimpleVectorStore:
    """Lightweight in-memory vector store using TF-IDF-like similarity."""
    
    def __init__(self):
        self.vectors: Dict[str, List[float]] = {}
        self.vocabulary: Dict[str, int] = {}
        self.records: Dict[str, MemoryRecord] = {}
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        return text.lower().split()
    
    def _build_vocabulary(self, texts: List[str]):
        """Build vocabulary from texts."""
        vocab = set()
        for text in texts:
            vocab.update(self._tokenize(text))
        self.vocabulary = {word: idx for idx, word in enumerate(sorted(vocab))}
    
    def _text_to_vector(self, text: str) -> List[float]:
        """Convert text to simple TF vector."""
        if not self.vocabulary:
            return []
        
        vector = [0.0] * len(self.vocabulary)
        tokens = self._tokenize(text)
        for token in tokens:
            if token in self.vocabulary:
                vector[self.vocabulary[token]] += 1.0
        
        # Normalize
        magnitude = math.sqrt(sum(v ** 2 for v in vector)) or 1.0
        vector = [v / magnitude for v in vector]
        return vector
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if not vec1 or not vec2:
            return 0.0
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        return dot_product
    
    def add(self, record: MemoryRecord):
        """Add record to vector store."""
        # Rebuild vocabulary with all texts
        all_texts = [r.content for r in self.records.values()] + [record.content]
        self._build_vocabulary(all_texts)
        
        # Convert to vectors
        for rec_id, rec in self.records.items():
            self.vectors[rec_id] = self._text_to_vector(rec.content)
        
        self.records[record.id] = record
        self.vectors[record.id] = self._text_to_vector(record.content)
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[MemoryRecord, float]]:
        """Search for similar records using vector similarity."""
        if not self.records:
            return []
        
        query_vector = self._text_to_vector(query)
        similarities = []
        
        for rec_id, record in self.records.items():
            if rec_id in self.vectors:
                similarity = self._cosine_similarity(query_vector, self.vectors[rec_id])
                similarities.append((record, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def search_by_topic(self, topic: str, top_k: int = 5) -> List[MemoryRecord]:
        """Search records by topic keyword."""
        results = [rec for rec in self.records.values() 
                  if topic.lower() in rec.topic.lower()]
        return results[:top_k]


class MemoryAgent:
    """
    Manages long-term storage, retrieval, and context updates.
    Serves as the persistent knowledge layer for the system.
    """
    
    def __init__(self):
        self.vector_store = SimpleVectorStore()
        self.conversation_history: List[MemoryRecord] = []
        self.knowledge_base: List[MemoryRecord] = []
        self.agent_state_memory: List[MemoryRecord] = []
    
    def store_conversation(self, content: str, topic: str, 
                          confidence: float = 1.0) -> MemoryRecord:
        """Store a conversation turn."""
        record = MemoryRecord(
            id=self._generate_id(content),
            content=content,
            timestamp=datetime.now().isoformat(),
            source_agent="user",
            topic=topic,
            confidence=confidence,
            record_type="conversation",
            metadata={"turn": len(self.conversation_history) + 1}
        )
        self.conversation_history.append(record)
        self.vector_store.add(record)
        return record
    
    def store_knowledge(self, content: str, topic: str, 
                       source_agent: str = "system",
                       confidence: float = 0.8) -> MemoryRecord:
        """Store a learned fact or finding."""
        record = MemoryRecord(
            id=self._generate_id(content),
            content=content,
            timestamp=datetime.now().isoformat(),
            source_agent=source_agent,
            topic=topic,
            confidence=confidence,
            record_type="knowledge",
            metadata={"provenance": source_agent}
        )
        self.knowledge_base.append(record)
        self.vector_store.add(record)
        return record
    
    def store_agent_state(self, agent_name: str, task: str, 
                         result: str, confidence: float = 0.9) -> MemoryRecord:
        """Track what an agent learned/accomplished."""
        record = MemoryRecord(
            id=self._generate_id(f"{agent_name}:{task}"),
            content=result,
            timestamp=datetime.now().isoformat(),
            source_agent=agent_name,
            topic=task,
            confidence=confidence,
            record_type="agent_state",
            metadata={"agent": agent_name, "task": task}
        )
        self.agent_state_memory.append(record)
        self.vector_store.add(record)
        return record
    
    def search_by_similarity(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for similar memories using vector similarity."""
        results = self.vector_store.search(query, top_k=top_k)
        return [{"record": rec.to_dict(), "similarity": float(sim)} 
                for rec, sim in results]
    
    def search_by_topic(self, topic: str, record_type: Optional[str] = None) -> List[Dict]:
        """Search memories by topic and optionally by record type."""
        if record_type == "conversation":
            source = self.conversation_history
        elif record_type == "knowledge":
            source = self.knowledge_base
        elif record_type == "agent_state":
            source = self.agent_state_memory
        else:
            source = self.conversation_history + self.knowledge_base + self.agent_state_memory
        
        results = [rec for rec in source if topic.lower() in rec.topic.lower()]
        return [rec.to_dict() for rec in results]
    
    def get_context_summary(self) -> Dict:
        """Get summary of all memories."""
        return {
            "total_records": len(self.conversation_history) + len(self.knowledge_base) + len(self.agent_state_memory),
            "conversations": len(self.conversation_history),
            "knowledge_facts": len(self.knowledge_base),
            "agent_states": len(self.agent_state_memory),
            "topics": list(set([rec.topic for rec in 
                               self.conversation_history + self.knowledge_base + self.agent_state_memory]))
        }
    
    def retrieve_past_discussion(self, topic: str) -> Optional[str]:
        """Retrieve summary of past discussion on a topic."""
        results = self.search_by_topic(topic, record_type="conversation")
        if results:
            return "\n".join([r["content"] for r in results[:3]])
        return None
    
    @staticmethod
    def _generate_id(content: str) -> str:
        """Generate unique ID based on content hash."""
        return hashlib.md5(content.encode()).hexdigest()[:12]
