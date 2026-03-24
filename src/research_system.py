"""
Main Autonomous Research Agent System orchestrator.
"""
import asyncio
import logging
from typing import Dict, Any, List
from datetime import datetime

from .models.data_models import ResearchResults, TopicMap, PaperMetadata, Claim
from .agents.topic_expansion_agent import TopicExpansionAgent
from .agents.enhanced_paper_discovery_agent import EnhancedPaperDiscoveryAgent as PaperDiscoveryAgent
from .agents.claim_extraction_agent import ClaimExtractionAgent
from .agents.claim_normalization_agent import ClaimNormalizationAgent
from .agents.contradiction_detection_agent import ContradictionDetectionAgent
from .agents.research_gap_detection_agent import ResearchGapDetectionAgent
from .agents.citation_builder_agent import CitationBuilderAgent
from .memory.memory_store import MemoryStore


class AutonomousResearchSystem:
    """
    Main orchestrator for the autonomous research system.
    Coordinates all agents to perform end-to-end research.
    """
    
    def __init__(self, storage_path: str = "data/memory"):
        self.memory_store = MemoryStore(storage_path)
        self.logger = self._setup_logging()
        
        # Initialize agents
        self.topic_expansion_agent = TopicExpansionAgent(self.memory_store)
        self.paper_discovery_agent = PaperDiscoveryAgent(self.memory_store)
        self.claim_extraction_agent = ClaimExtractionAgent(self.memory_store)
        self.claim_normalization_agent = ClaimNormalizationAgent(self.memory_store)
        self.contradiction_detection_agent = ContradictionDetectionAgent(self.memory_store)
        self.research_gap_detection_agent = ResearchGapDetectionAgent(self.memory_store)
        self.citation_builder_agent = CitationBuilderAgent(self.memory_store)
        
        self.logger.info("Autonomous Research System initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('research_system.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger("AutonomousResearchSystem")
    
    async def research(self, topic: str) -> ResearchResults:
        """
        Perform autonomous research on a given topic or author name.
        
        Args:
            topic: Research topic, question, or author name
            
        Returns:
            ResearchResults: Comprehensive research findings
        """
        self.logger.info(f"Starting autonomous research on topic: {topic}")
        start_time = datetime.now()
        
        try:
            # Detect if query is an author name
            is_author = self._is_author_query(topic)
            
            # Stage 1: Topic Expansion
            self.logger.info("Stage 1: Topic Expansion")
            topic_map = await self.topic_expansion_agent.process(topic)
            
            # Tag topic_map with author flag so discovery agent knows
            if is_author:
                self.logger.info(f"Detected author query: {topic}")
                topic_map.main_topic = topic  # keep original name
                topic_map.keywords = [topic]  # use name as keyword
                topic_map.subtopics = [f"papers by {topic}"]
                # Store author flag in related_areas as a signal
                topic_map.related_areas = [f"__author__:{topic}"]
            
            # Stage 2: Paper Discovery
            self.logger.info("Stage 2: Paper Discovery")
            async with self.paper_discovery_agent:
                papers = await self.paper_discovery_agent.process(topic_map)
            
            # Stage 3: Claim Extraction
            self.logger.info("Stage 3: Claim Extraction")
            claims = await self.claim_extraction_agent.process(papers)
            
            # Stage 4: Claim Normalization & Verification
            self.logger.info("Stage 4: Claim Normalization & Verification")
            normalized_claims = await self.claim_normalization_agent.process(claims)
            
            # Stage 5: Contradiction Detection
            self.logger.info("Stage 5: Contradiction Detection")
            contradictions = await self.contradiction_detection_agent.process(normalized_claims)
            
            # Stage 6: Research Gap Detection
            self.logger.info("Stage 6: Research Gap Detection")
            research_gaps = await self.research_gap_detection_agent.process(topic_map, normalized_claims)
            
            # Stage 7: Citation Building
            self.logger.info("Stage 7: Citation Building")
            citations = await self.citation_builder_agent.process(papers)
            
            # Stage 8: Store in Long-term Memory
            self.logger.info("Stage 8: Storing in Long-term Memory")
            await self._store_in_memory(topic_map, papers, normalized_claims, contradictions, research_gaps)
            
            # Compile results
            results = ResearchResults(
                topic_map=topic_map,
                papers=papers,
                claims=normalized_claims,
                contradictions=contradictions,
                research_gaps=research_gaps,
                citations=citations,
                total_papers_analyzed=len(papers),
                total_claims_extracted=len(normalized_claims)
            )
            
            duration = datetime.now() - start_time
            self.logger.info(f"Research completed in {duration.total_seconds():.2f} seconds")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error during research: {e}")
            raise

    def _is_author_query(self, query: str) -> bool:
        """Detect if the query looks like an author name rather than a topic."""
        query = query.strip()
        parts = query.split()
        
        # Must be 2-4 words
        if not (2 <= len(parts) <= 4):
            return False
        
        # All parts should start with a capital letter
        if not all(p[0].isupper() for p in parts if p):
            return False
        
        # Should not contain common topic words
        topic_words = {
            'graph', 'network', 'learning', 'deep', 'neural', 'index',
            'theory', 'analysis', 'detection', 'classification', 'model',
            'algorithm', 'method', 'approach', 'system', 'based', 'using',
            'wiener', 'polarity', 'metric', 'dimension', 'topological',
            'machine', 'artificial', 'intelligence', 'data', 'science'
        }
        if any(p.lower() in topic_words for p in parts):
            return False
        
        # No digits
        if any(c.isdigit() for c in query):
            return False
        
        return True
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory store statistics."""
        return await self.memory_store.get_cache_stats()
    
    async def _store_in_memory(self, topic_map: TopicMap, papers: List[PaperMetadata], 
                             claims: List[Claim], contradictions: List[Any], 
                             research_gaps: List[Any]) -> None:
        """Store research results in long-term memory."""
        
        # Store topic map
        await self.memory_store.store("latest_topic_map", topic_map)
        
        # Store papers
        await self.memory_store.store("latest_papers", papers)
        
        # Store claims
        await self.memory_store.store("latest_claims", claims)
        
        # Store contradictions
        await self.memory_store.store("latest_contradictions", contradictions)
        
        # Store research gaps
        await self.memory_store.store("latest_research_gaps", research_gaps)
        
        self.logger.info("Research results stored in long-term memory")