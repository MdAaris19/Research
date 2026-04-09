"""
Enhanced Paper Discovery Agent - Searches multiple academic databases.
"""
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
import xml.etree.ElementTree as ET
from datetime import datetime
import re
import json
import difflib
import os
from dotenv import load_dotenv

load_dotenv()

from ..models.data_models import PaperMetadata, TopicMap
from .base_agent import BaseAgent


class EnhancedPaperDiscoveryAgent(BaseAgent):
    """Enhanced agent that searches multiple academic databases."""
    
    def __init__(self, memory_store=None, max_papers_per_source: int = 50):
        super().__init__("EnhancedPaperDiscoveryAgent", memory_store)
        self.max_papers_per_source = max_papers_per_source
        self.session = None
        self.primary_source_target = max(10, min(self.max_papers_per_source, 25))
        
        # API configurations
        self.apis = {
            "openalex": {
                "enabled": True,
                "requires_key": False,
                "base_url": "https://api.openalex.org/works",
                "author_url": "https://api.openalex.org/authors",
                "email": os.getenv("OPENALEX_EMAIL") or os.getenv("CROSSREF_EMAIL") or ""
            },
            "arxiv": {
                "enabled": True,
                "requires_key": False,
                "base_url": "http://export.arxiv.org/api/query"
            },
            "semantic_scholar": {
                "enabled": True,
                "requires_key": False,  # Free tier available
                "base_url": "https://api.semanticscholar.org/graph/v1/paper/search",
                "api_key": os.getenv("SEMANTIC_SCHOLAR_API_KEY")
            },
            "crossref": {
                "enabled": True,
                "requires_key": False,  # Free but rate limited
                "base_url": "https://api.crossref.org/works",
                "email": os.getenv("CROSSREF_EMAIL", "your-email@example.com")
            },
            "pubmed": {
                "enabled": True,
                "requires_key": False,  # Free
                "base_url": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                "api_key": os.getenv("PUBMED_API_KEY")
            },
            "springer": {
                "enabled": bool(os.getenv("SPRINGER_META_API_KEY") or os.getenv("SPRINGER_API_KEY")),
                "requires_key": True,
                "base_url": "https://api.springernature.com/meta/v2/json",
                "api_key": os.getenv("SPRINGER_META_API_KEY") or os.getenv("SPRINGER_API_KEY"),
                "openaccess_api_key": os.getenv("SPRINGER_OPENACCESS_API_KEY")
            },
            "elsevier": {
                "enabled": bool(os.getenv("ELSEVIER_API_KEY")),
                "requires_key": True,
                "base_url": "https://api.elsevier.com/content/search/sciencedirect",
                "api_key": os.getenv("ELSEVIER_API_KEY"),
                "inst_token": os.getenv("ELSEVIER_INST_TOKEN")
            },
            "wiley": {
                "enabled": bool(os.getenv("WILEY_API_KEY")),
                "requires_key": True,
                "base_url": "https://api.wiley.com/onlinelibrary/tdm/v1/articles",
                "api_key": os.getenv("WILEY_API_KEY")
            }
        }
    
    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=20)
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def process(self, topic_map: TopicMap) -> List[PaperMetadata]:
        """
        Discover papers from multiple sources based on the topic map.
        """
        self.log_operation("enhanced_paper_discovery_start", {
            "main_topic": topic_map.main_topic,
            "keywords_count": len(topic_map.keywords),
            "enabled_sources": [name for name, config in self.apis.items() if config["enabled"]]
        })
        
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=20)
            self.session = aiohttp.ClientSession(timeout=timeout)
        
        # Check if this is an author search
        author_name = None
        paper_title_query = None
        metric_query = None
        for area in topic_map.related_areas:
            if area.startswith("__author__:"):
                author_name = area.replace("__author__:", "").strip()
                break
            if area.startswith("__paper_title__:"):
                paper_title_query = area.replace("__paper_title__:", "").strip()
            if area.startswith("__metric__:"):
                metric_query = area.replace("__metric__:", "").strip()
        
        all_papers = []
        
        if author_name:
            # Author search — use dedicated author search methods
            self.logger.info(f"Running author search for: {author_name}")
            core_tasks = [
                ("openalex", self._search_openalex_by_author(author_name)),
                ("crossref", self._search_crossref_by_author(author_name)),
            ]
            supplemental_tasks = [
                ("semantic_scholar", self._search_semantic_scholar_by_author(author_name)),
                ("arxiv", self._search_arxiv_by_author(author_name)),
            ]
        elif paper_title_query:
            self.logger.info(f"Running paper title search for: {paper_title_query}")
            core_tasks = [
                ("openalex", self._search_openalex_by_title(paper_title_query)),
                ("crossref", self._search_crossref_by_title(paper_title_query)),
            ]
            supplemental_tasks = [
                ("semantic_scholar", self._search_semantic_scholar_by_title(paper_title_query)),
                ("pubmed", self._search_pubmed_by_title(paper_title_query)),
                ("arxiv", self._search_arxiv_by_title(paper_title_query)),
            ]
        else:
            core_tasks = []
            supplemental_tasks = []
            if self.apis["openalex"]["enabled"]:
                core_tasks.append(("openalex", self._search_openalex(topic_map)))
            if self.apis["crossref"]["enabled"]:
                core_tasks.append(("crossref", self._search_crossref(topic_map)))
            if self.apis["semantic_scholar"]["enabled"]:
                supplemental_tasks.append(("semantic_scholar", self._search_semantic_scholar(topic_map)))
            if self.apis["pubmed"]["enabled"]:
                supplemental_tasks.append(("pubmed", self._search_pubmed(topic_map)))
            if self.apis["springer"]["enabled"] and self.apis["springer"]["api_key"]:
                supplemental_tasks.append(("springer", self._search_springer(topic_map)))
            if self.apis["elsevier"]["enabled"] and self.apis["elsevier"]["api_key"]:
                supplemental_tasks.append(("elsevier", self._search_elsevier(topic_map)))
            if self.apis["wiley"]["enabled"] and self.apis["wiley"]["api_key"]:
                supplemental_tasks.append(("wiley", self._search_wiley(topic_map)))
            if self.apis["arxiv"]["enabled"]:
                supplemental_tasks.append(("arxiv", self._search_arxiv(topic_map)))

        source_counts = {}
        core_papers, core_counts = await self._run_source_tasks(core_tasks)
        all_papers.extend(core_papers)
        source_counts.update(core_counts)

        needs_supplemental = (
            not core_papers or
            len(core_papers) < (self.primary_source_target * 2) or
            paper_title_query is not None or
            author_name is not None
        )

        if needs_supplemental and supplemental_tasks:
            supplemental_papers, supplemental_counts = await self._run_source_tasks(supplemental_tasks)
            all_papers.extend(supplemental_papers)
            source_counts.update(supplemental_counts)
        
        # Remove duplicates and rank papers
        unique_papers = self._remove_duplicates(all_papers)
        ranked_papers = self._rank_papers(unique_papers, topic_map)
        if paper_title_query:
            ranked_papers = self._filter_exact_title_matches(ranked_papers, paper_title_query)
        elif metric_query:
            ranked_papers = self._prioritize_metric_matches(ranked_papers, metric_query)
        
        await self.store_result("enhanced_discovered_papers", ranked_papers)
        
        self.log_operation("enhanced_paper_discovery_complete", {
            "total_papers": len(ranked_papers),
            "source_counts": source_counts,
            "unique_papers": len(unique_papers)
        })
        
        return ranked_papers

    async def _run_source_tasks(self, named_tasks: List[Any]) -> tuple[List[PaperMetadata], Dict[str, int]]:
        """Execute a list of source tasks and collect results."""
        if not named_tasks:
            return [], {}

        results = await asyncio.gather(*(task for _, task in named_tasks), return_exceptions=True)
        papers: List[PaperMetadata] = []
        source_counts: Dict[str, int] = {}

        for (source_name, _), result in zip(named_tasks, results):
            if isinstance(result, list):
                papers.extend(result)
                source_counts[source_name] = len(result)
            elif isinstance(result, Exception):
                self.logger.warning(f"Error searching {source_name}: {result}")
                source_counts[source_name] = 0

        return papers, source_counts

    async def _search_openalex(self, topic_map: TopicMap) -> List[PaperMetadata]:
        """Search OpenAlex as the primary general source."""
        metric_query = self._extract_marker_value(topic_map, "__metric__:")
        query = metric_query if metric_query else topic_map.main_topic
        return await self._search_openalex_works(query)

    async def _search_openalex_by_title(self, title_query: str) -> List[PaperMetadata]:
        """Search OpenAlex by paper title."""
        return await self._search_openalex_works(title_query)

    async def _search_openalex_by_author(self, author_name: str) -> List[PaperMetadata]:
        """Search OpenAlex by author using the authors endpoint first."""
        papers = []
        headers = self._build_mailto_headers(self.apis["openalex"].get("email"))
        try:
            params = {"search": author_name, "per-page": 3}
            async with self.session.get(self.apis["openalex"]["author_url"], params=params, headers=headers) as response:
                if response.status != 200:
                    return papers
                data = await response.json()
                authors = data.get("results", [])
                if not authors:
                    return papers
                author_id = authors[0].get("id", "")
                if not author_id:
                    return papers

            params = {
                "filter": f"author.id:{author_id}",
                "per-page": self.max_papers_per_source,
                "sort": "relevance_score:desc"
            }
            async with self.session.get(self.apis["openalex"]["base_url"], params=params, headers=headers) as response:
                if response.status == 200:
                    papers = self._parse_openalex_response(await response.json())
        except Exception as e:
            self.logger.error(f"OpenAlex author search error: {e}")
        return papers

    async def _search_openalex_works(self, query: str) -> List[PaperMetadata]:
        """Search OpenAlex works endpoint."""
        papers = []
        params = {
            "search": query,
            "per-page": self.max_papers_per_source,
            "sort": "relevance_score:desc"
        }
        headers = self._build_mailto_headers(self.apis["openalex"].get("email"))
        try:
            async with self.session.get(self.apis["openalex"]["base_url"], params=params, headers=headers) as response:
                if response.status == 200:
                    papers = self._parse_openalex_response(await response.json())
        except Exception as e:
            self.logger.error(f"OpenAlex search error: {e}")
        return papers

    def _build_mailto_headers(self, email: str) -> Dict[str, str]:
        """Build polite-pool headers when contact email is available."""
        if not email or email == "your-email@example.com":
            return {}
        return {"User-Agent": f"AutonomousResearchSystem/1.0 ({email})"}

    async def _search_arxiv_by_title(self, title_query: str) -> List[PaperMetadata]:
        """Search ArXiv using an exact-title oriented query."""
        papers = []
        params = {
            "search_query": f'ti:"{title_query}" OR all:"{title_query}"',
            "start": 0,
            "max_results": self.max_papers_per_source,
            "sortBy": "relevance",
            "sortOrder": "descending"
        }
        try:
            async with self.session.get(self.apis["arxiv"]["base_url"], params=params) as response:
                if response.status == 200:
                    papers = self._parse_arxiv_response(await response.text())
        except Exception as e:
            self.logger.error(f"ArXiv title search error: {e}")
        return papers

    async def _search_semantic_scholar_by_title(self, title_query: str) -> List[PaperMetadata]:
        """Search Semantic Scholar using an exact-title oriented query."""
        papers = []
        params = {
            "query": f'"{title_query}"',
            "limit": min(self.max_papers_per_source, 100),
            "fields": "paperId,title,authors,year,venue,abstract,citationCount,url,externalIds"
        }
        headers = {}
        if self.apis["semantic_scholar"]["api_key"]:
            headers["x-api-key"] = self.apis["semantic_scholar"]["api_key"]
        try:
            async with self.session.get(self.apis["semantic_scholar"]["base_url"], params=params, headers=headers) as response:
                if response.status == 200:
                    papers = self._parse_semantic_scholar_response(await response.json())
        except Exception as e:
            self.logger.error(f"Semantic Scholar title search error: {e}")
        return papers

    async def _search_crossref_by_title(self, title_query: str) -> List[PaperMetadata]:
        """Search CrossRef using title-specific fields."""
        papers = []
        params = {
            "query.title": title_query,
            "rows": self.max_papers_per_source,
            "sort": "relevance",
            "mailto": self.apis["crossref"]["email"]
        }
        try:
            async with self.session.get(self.apis["crossref"]["base_url"], params=params) as response:
                if response.status == 200:
                    papers = self._parse_crossref_response(await response.json())
        except Exception as e:
            self.logger.error(f"CrossRef title search error: {e}")
        return papers

    async def _search_pubmed_by_title(self, title_query: str) -> List[PaperMetadata]:
        """Search PubMed by title phrase."""
        papers = []
        params = {
            "db": "pubmed",
            "term": f'"{title_query}"[Title]',
            "retmax": self.max_papers_per_source,
            "retmode": "json"
        }
        try:
            async with self.session.get(self.apis["pubmed"]["base_url"], params=params) as response:
                if response.status == 200:
                    papers = await self._parse_pubmed_response(await response.json())
        except Exception as e:
            self.logger.error(f"PubMed title search error: {e}")
        return papers
    
    async def _search_arxiv_by_author(self, author_name: str) -> List[PaperMetadata]:
        """Search ArXiv for papers by a specific author."""
        papers = []
        # ArXiv author search uses au: prefix
        query = f'au:"{author_name}"'
        url = self.apis["arxiv"]["base_url"]
        params = {
            "search_query": query,
            "start": 0,
            "max_results": self.max_papers_per_source,
            "sortBy": "submittedDate",
            "sortOrder": "descending"
        }
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    content = await response.text()
                    papers = self._parse_arxiv_response(content)
        except Exception as e:
            self.logger.error(f"ArXiv author search error: {e}")
        return papers

    async def _search_semantic_scholar_by_author(self, author_name: str) -> List[PaperMetadata]:
        """Search Semantic Scholar for papers by a specific author."""
        papers = []
        # First find the author ID
        try:
            author_url = "https://api.semanticscholar.org/graph/v1/author/search"
            params = {"query": author_name, "fields": "authorId,name,paperCount"}
            headers = {}
            if self.apis["semantic_scholar"]["api_key"]:
                headers["x-api-key"] = self.apis["semantic_scholar"]["api_key"]

            async with self.session.get(author_url, params=params, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    authors = data.get("data", [])
                    if not authors:
                        return papers
                    # Take the first (most relevant) author match
                    author_id = authors[0].get("authorId")
                    if not author_id:
                        return papers

                    # Now fetch their papers
                    papers_url = f"https://api.semanticscholar.org/graph/v1/author/{author_id}/papers"
                    paper_params = {
                        "fields": "paperId,title,authors,year,venue,abstract,citationCount,externalIds,url",
                        "limit": self.max_papers_per_source
                    }
                    async with self.session.get(papers_url, params=paper_params, headers=headers) as presp:
                        if presp.status == 200:
                            pdata = await presp.json()
                            papers = self._parse_semantic_scholar_response(pdata)
        except Exception as e:
            self.logger.error(f"Semantic Scholar author search error: {e}")
        return papers

    async def _search_crossref_by_author(self, author_name: str) -> List[PaperMetadata]:
        """Search CrossRef for papers by a specific author."""
        papers = []
        url = self.apis["crossref"]["base_url"]
        params = {
            "query.author": author_name,
            "rows": self.max_papers_per_source,
            "sort": "relevance",
            "mailto": self.apis["crossref"]["email"]
        }
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    papers = self._parse_crossref_response(data)
        except Exception as e:
            self.logger.error(f"CrossRef author search error: {e}")
        return papers

    async def _search_arxiv(self, topic_map: TopicMap) -> List[PaperMetadata]:
        """Search ArXiv (existing implementation)."""
        papers = []

        metric_query = self._extract_marker_value(topic_map, "__metric__:")
        if metric_query:
            query = f'"{metric_query}"'
        else:
            query_terms = [topic_map.main_topic] + topic_map.keywords[:5]
            query = " AND ".join([f'"{term}"' for term in query_terms])
        
        url = self.apis["arxiv"]["base_url"]
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": self.max_papers_per_source,
            "sortBy": "relevance",
            "sortOrder": "descending"
        }
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    content = await response.text()
                    papers = self._parse_arxiv_response(content)
        except Exception as e:
            self.logger.error(f"ArXiv search error: {e}")
        
        return papers
    
    async def _search_semantic_scholar(self, topic_map: TopicMap) -> List[PaperMetadata]:
        """Search Semantic Scholar API."""
        papers = []

        metric_query = self._extract_marker_value(topic_map, "__metric__:")
        query = f'"{metric_query}"' if metric_query else topic_map.main_topic
        url = self.apis["semantic_scholar"]["base_url"]
        
        params = {
            "query": query,
            "limit": min(self.max_papers_per_source, 100),  # API limit
            "fields": "paperId,title,authors,year,venue,abstract,citationCount,url,externalIds"
        }
        
        headers = {}
        if self.apis["semantic_scholar"]["api_key"]:
            headers["x-api-key"] = self.apis["semantic_scholar"]["api_key"]
        
        try:
            async with self.session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    papers = self._parse_semantic_scholar_response(data)
                else:
                    self.logger.warning(f"Semantic Scholar API returned {response.status}")
        except Exception as e:
            self.logger.error(f"Semantic Scholar search error: {e}")
        
        return papers
    
    async def _search_crossref(self, topic_map: TopicMap) -> List[PaperMetadata]:
        """Search CrossRef API (covers many publishers)."""
        papers = []

        metric_query = self._extract_marker_value(topic_map, "__metric__:")
        query = metric_query if metric_query else topic_map.main_topic
        url = self.apis["crossref"]["base_url"]
        
        params = {
            "query": query,
            "rows": self.max_papers_per_source,
            "sort": "relevance",
            "mailto": self.apis["crossref"]["email"]  # For polite pool
        }
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    papers = self._parse_crossref_response(data)
        except Exception as e:
            self.logger.error(f"CrossRef search error: {e}")
        
        return papers
    
    async def _search_pubmed(self, topic_map: TopicMap) -> List[PaperMetadata]:
        """Search PubMed API."""
        papers = []

        metric_query = self._extract_marker_value(topic_map, "__metric__:")
        query_text = f'"{metric_query}"' if metric_query else topic_map.main_topic
        query = query_text.replace(" ", "+")
        url = self.apis["pubmed"]["base_url"]
        
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": self.max_papers_per_source,
            "retmode": "json"
        }
        
        if self.apis["pubmed"]["api_key"]:
            params["api_key"] = self.apis["pubmed"]["api_key"]
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    # PubMed requires additional calls to get full details
                    papers = await self._parse_pubmed_response(data)
        except Exception as e:
            self.logger.error(f"PubMed search error: {e}")
        
        return papers
    
    async def _search_springer(self, topic_map: TopicMap) -> List[PaperMetadata]:
        """Search Springer Nature API."""
        papers = []
        
        if not self.apis["springer"]["api_key"]:
            return papers
        
        query = topic_map.main_topic
        url = self.apis["springer"]["base_url"]
        
        params = {
            "q": query,
            "p": self.max_papers_per_source,
            "api_key": self.apis["springer"]["api_key"]
        }
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    papers = self._parse_springer_response(data)
        except Exception as e:
            self.logger.error(f"Springer search error: {e}")
        
        return papers
    
    async def _search_elsevier(self, topic_map: TopicMap) -> List[PaperMetadata]:
        """Search Elsevier ScienceDirect API."""
        papers = []
        
        if not self.apis["elsevier"]["api_key"]:
            return papers
        
        query = topic_map.main_topic
        url = self.apis["elsevier"]["base_url"]
        
        headers = {
            "X-ELS-APIKey": self.apis["elsevier"]["api_key"],
            "Accept": "application/json"
        }
        
        if self.apis["elsevier"]["inst_token"]:
            headers["X-ELS-Insttoken"] = self.apis["elsevier"]["inst_token"]
        
        params = {
            "query": query,
            "count": self.max_papers_per_source
        }
        
        try:
            async with self.session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    papers = self._parse_elsevier_response(data)
        except Exception as e:
            self.logger.error(f"Elsevier search error: {e}")
        
        return papers
    
    async def _search_wiley(self, topic_map: TopicMap) -> List[PaperMetadata]:
        """Search Wiley API."""
        papers = []
        
        if not self.apis["wiley"]["api_key"]:
            return papers

        metric_query = self._extract_marker_value(topic_map, "__metric__:")
        query = metric_query if metric_query else topic_map.main_topic
        headers = {
            "Wiley-TDM-Client-Token": self.apis["wiley"]["api_key"],
            "Accept": "application/json"
        }
        params = {
            "q": query,
            "limit": self.max_papers_per_source
        }

        try:
            async with self.session.get(self.apis["wiley"]["base_url"], params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    papers = self._parse_wiley_response(data)
                else:
                    self.logger.warning(f"Wiley API returned {response.status}")
        except Exception as e:
            self.logger.error(f"Wiley search error: {e}")

        return papers
    
    def _parse_arxiv_response(self, xml_content: str) -> List[PaperMetadata]:
        """Parse ArXiv API XML response (existing implementation)."""
        papers = []
        
        try:
            root = ET.fromstring(xml_content)
            namespace = {"atom": "http://www.w3.org/2005/Atom"}
            
            for entry in root.findall("atom:entry", namespace):
                title_elem = entry.find("atom:title", namespace)
                summary_elem = entry.find("atom:summary", namespace)
                published_elem = entry.find("atom:published", namespace)
                id_elem = entry.find("atom:id", namespace)
                
                if title_elem is not None and summary_elem is not None:
                    title = title_elem.text.strip().replace('\n', ' ')
                    abstract = summary_elem.text.strip().replace('\n', ' ')
                    
                    authors = []
                    for author in entry.findall("atom:author", namespace):
                        name_elem = author.find("atom:name", namespace)
                        if name_elem is not None:
                            authors.append(name_elem.text.strip())
                    
                    year = 2023
                    if published_elem is not None:
                        try:
                            date_str = published_elem.text
                            year = int(date_str[:4])
                        except (ValueError, IndexError):
                            pass
                    
                    arxiv_id = None
                    if id_elem is not None:
                        arxiv_match = re.search(r'(\d{4}\.\d{4,5})', id_elem.text)
                        if arxiv_match:
                            arxiv_id = arxiv_match.group(1)
                    
                    paper = PaperMetadata(
                        title=title,
                        authors=authors,
                        year=year,
                        venue="arXiv",
                        arxiv_id=arxiv_id,
                        abstract=abstract,
                        url=id_elem.text if id_elem is not None else None
                    )
                    papers.append(paper)
        
        except ET.ParseError as e:
            self.logger.error(f"Failed to parse ArXiv XML: {e}")
        
        return papers
    
    def _parse_semantic_scholar_response(self, data: dict) -> List[PaperMetadata]:
        """Parse Semantic Scholar API response."""
        papers = []
        
        for item in data.get("data", []):
            try:
                authors = [author.get("name", "") for author in item.get("authors", [])]
                
                # Extract DOI if available
                doi = None
                external_ids = item.get("externalIds", {})
                if external_ids and "DOI" in external_ids:
                    doi = external_ids["DOI"]
                
                paper = PaperMetadata(
                    title=item.get("title", ""),
                    authors=authors,
                    year=item.get("year", 2023),
                    venue=item.get("venue", "Unknown"),
                    doi=doi,
                    abstract=item.get("abstract", "") or "",  # Handle None abstracts
                    url=item.get("url", ""),
                    impact_score=item.get("citationCount", 0)
                )
                papers.append(paper)
            except Exception as e:
                self.logger.warning(f"Error parsing Semantic Scholar paper: {e}")
                continue
        
        return papers

    def _parse_openalex_response(self, data: dict) -> List[PaperMetadata]:
        """Parse OpenAlex API response."""
        papers = []

        for item in data.get("results", []):
            try:
                authors = [
                    authorship.get("author", {}).get("display_name", "")
                    for authorship in item.get("authorships", [])
                    if authorship.get("author", {}).get("display_name")
                ]

                doi = item.get("doi")
                if doi:
                    doi = doi.replace("https://doi.org/", "")

                source = (item.get("primary_location") or {}).get("source") or {}
                venue = source.get("display_name") or "Unknown"

                abstract = ""
                abstract_index = item.get("abstract_inverted_index") or {}
                if abstract_index:
                    max_index = max((max(indexes) for indexes in abstract_index.values() if indexes), default=-1)
                    if max_index >= 0:
                        abstract_tokens = [""] * (max_index + 1)
                        for word, indexes in abstract_index.items():
                            for idx in indexes:
                                if 0 <= idx < len(abstract_tokens):
                                    abstract_tokens[idx] = word
                        abstract = " ".join(token for token in abstract_tokens if token)

                paper = PaperMetadata(
                    title=item.get("title", ""),
                    authors=authors,
                    year=item.get("publication_year") or 2023,
                    venue=venue,
                    doi=doi,
                    abstract=abstract,
                    url=((item.get("primary_location") or {}).get("landing_page_url") or item.get("id", "")),
                    impact_score=item.get("cited_by_count", 0)
                )
                papers.append(paper)
            except Exception as e:
                self.logger.warning(f"Error parsing OpenAlex paper: {e}")
                continue

        return papers
    
    def _parse_crossref_response(self, data: dict) -> List[PaperMetadata]:
        """Parse CrossRef API response."""
        papers = []
        
        for item in data.get("message", {}).get("items", []):
            try:
                # Extract authors
                authors = []
                for author in item.get("author", []):
                    given = author.get("given", "")
                    family = author.get("family", "")
                    if given and family:
                        authors.append(f"{given} {family}")
                    elif family:
                        authors.append(family)
                
                # Extract year
                year = 2023
                published = item.get("published-print") or item.get("published-online")
                if published and "date-parts" in published:
                    try:
                        year = published["date-parts"][0][0]
                    except (IndexError, TypeError):
                        pass
                
                # Extract venue
                venue = ""
                if "container-title" in item and item["container-title"]:
                    venue = item["container-title"][0]
                
                paper = PaperMetadata(
                    title=" ".join(item.get("title", [""])),
                    authors=authors,
                    year=year,
                    venue=venue,
                    doi=item.get("DOI", ""),
                    abstract=item.get("abstract", ""),
                    url=item.get("URL", "")
                )
                papers.append(paper)
            except Exception as e:
                self.logger.warning(f"Error parsing CrossRef paper: {e}")
                continue
        
        return papers
    
    async def _parse_pubmed_response(self, data: dict) -> List[PaperMetadata]:
        """Parse PubMed API response."""
        papers = []
        
        # PubMed search returns IDs, need additional calls for details
        # This is a simplified implementation
        id_list = data.get("esearchresult", {}).get("idlist", [])
        
        if not id_list:
            return papers
        
        # Fetch details for the IDs (simplified)
        # In a full implementation, you'd make additional API calls
        for pmid in id_list[:10]:  # Limit to avoid too many requests
            try:
                paper = PaperMetadata(
                    title=f"PubMed Paper {pmid}",  # Would fetch real title
                    authors=["Unknown"],  # Would fetch real authors
                    year=2023,  # Would fetch real year
                    venue="PubMed",
                    abstract="Abstract would be fetched from additional API call",
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                )
                papers.append(paper)
            except Exception as e:
                self.logger.warning(f"Error creating PubMed paper: {e}")
                continue
        
        return papers
    
    def _parse_springer_response(self, data: dict) -> List[PaperMetadata]:
        """Parse Springer Nature API response."""
        papers = []
        
        for item in data.get("records", []):
            try:
                authors = []
                for creator in item.get("creators", []):
                    authors.append(creator.get("creator", ""))
                
                paper = PaperMetadata(
                    title=item.get("title", ""),
                    authors=authors,
                    year=int(item.get("publicationDate", "2023")[:4]),
                    venue=item.get("publicationName", "Springer"),
                    doi=item.get("doi", ""),
                    abstract=item.get("abstract", ""),
                    url=item.get("url", "")
                )
                papers.append(paper)
            except Exception as e:
                self.logger.warning(f"Error parsing Springer paper: {e}")
                continue
        
        return papers
    
    def _parse_elsevier_response(self, data: dict) -> List[PaperMetadata]:
        """Parse Elsevier API response."""
        papers = []
        
        entries = data.get("search-results", {}).get("entry", [])
        
        for item in entries:
            try:
                authors = []
                if "authors" in item:
                    for author in item["authors"].get("author", []):
                        given_name = author.get("given-name", "")
                        surname = author.get("surname", "")
                        if given_name and surname:
                            authors.append(f"{given_name} {surname}")
                
                paper = PaperMetadata(
                    title=item.get("dc:title", ""),
                    authors=authors,
                    year=int(item.get("prism:coverDate", "2023")[:4]),
                    venue=item.get("prism:publicationName", "Elsevier"),
                    doi=item.get("prism:doi", ""),
                    abstract=item.get("dc:description", ""),
                    url=item.get("link", [{}])[0].get("@href", "")
                )
                papers.append(paper)
            except Exception as e:
                self.logger.warning(f"Error parsing Elsevier paper: {e}")
                continue
        
        return papers

    def _parse_wiley_response(self, data: dict) -> List[PaperMetadata]:
        """Parse Wiley API response."""
        papers = []

        items = data.get("articles") or data.get("results") or []
        for item in items:
            try:
                authors = item.get("authors") or []
                if authors and isinstance(authors[0], dict):
                    authors = [author.get("displayName", "") or author.get("name", "") for author in authors]

                publication_date = item.get("publicationDate", "2023")
                year_match = re.search(r"(19|20)\d{2}", publication_date)
                year = int(year_match.group(0)) if year_match else 2023

                paper = PaperMetadata(
                    title=item.get("title", ""),
                    authors=[author for author in authors if author],
                    year=year,
                    venue=item.get("publicationTitle", "Wiley"),
                    doi=item.get("doi", ""),
                    abstract=item.get("abstract", ""),
                    url=item.get("url") or item.get("link", "")
                )
                papers.append(paper)
            except Exception as e:
                self.logger.warning(f"Error parsing Wiley paper: {e}")
                continue

        return papers
    
    def _remove_duplicates(self, papers: List[PaperMetadata]) -> List[PaperMetadata]:
        """Remove duplicate papers based on title and DOI."""
        seen_titles = set()
        seen_dois = set()
        unique_papers = []
        
        for paper in papers:
            # Normalize title for comparison
            normalized_title = re.sub(r'[^\w\s]', '', paper.title.lower()).strip()
            
            # Check for duplicates
            is_duplicate = False
            
            if paper.doi and paper.doi in seen_dois:
                is_duplicate = True
            elif normalized_title in seen_titles:
                is_duplicate = True
            
            if not is_duplicate:
                if paper.doi:
                    seen_dois.add(paper.doi)
                seen_titles.add(normalized_title)
                unique_papers.append(paper)
        
        return unique_papers

    def _extract_marker_value(self, topic_map: TopicMap, prefix: str) -> Optional[str]:
        """Read a query marker from related areas metadata."""
        for area in topic_map.related_areas:
            if area.startswith(prefix):
                return area.replace(prefix, "", 1).strip()
        return None

    def _normalize_text(self, text: str) -> str:
        """Normalize text for approximate matching."""
        return re.sub(r'[^\w\s]', '', (text or '').lower()).strip()

    def _title_similarity(self, left: str, right: str) -> float:
        """Compute approximate title similarity."""
        return difflib.SequenceMatcher(None, self._normalize_text(left), self._normalize_text(right)).ratio()

    def _title_overlap(self, query: str, title: str) -> float:
        """Compute token overlap between a query and title."""
        query_tokens = set(self._normalize_text(query).split())
        title_tokens = set(self._normalize_text(title).split())
        if not query_tokens or not title_tokens:
            return 0.0
        return len(query_tokens & title_tokens) / max(len(query_tokens), 1)

    def _filter_exact_title_matches(self, papers: List[PaperMetadata], title_query: str) -> List[PaperMetadata]:
        """Keep only papers that closely match an exact title query."""
        scored = []
        for paper in papers:
            similarity = self._title_similarity(title_query, paper.title)
            overlap = self._title_overlap(title_query, paper.title)
            if similarity >= 0.72 or overlap >= 0.8:
                paper.relevance_score = max(paper.relevance_score, min(1.0, (similarity * 0.75) + (overlap * 0.25)))
                scored.append((similarity, overlap, paper))

        if not scored:
            return papers[:10]

        scored.sort(key=lambda item: (item[0], item[1], item[2].relevance_score), reverse=True)
        return [paper for _, _, paper in scored[:10]]

    def _prioritize_metric_matches(self, papers: List[PaperMetadata], metric_query: str) -> List[PaperMetadata]:
        """Boost papers that explicitly mention the requested metric or index."""
        boosted = []
        for paper in papers:
            overlap = self._title_overlap(metric_query, f"{paper.title} {paper.abstract}")
            phrase_hit = self._normalize_text(metric_query) in self._normalize_text(f"{paper.title} {paper.abstract}")
            bonus = 0.25 if phrase_hit else 0.0
            paper.relevance_score = min(1.0, max(paper.relevance_score, overlap) + bonus)
            boosted.append(paper)

        boosted.sort(key=lambda paper: paper.relevance_score, reverse=True)
        return boosted
    
    def _rank_papers(self, papers: List[PaperMetadata], topic_map: TopicMap) -> List[PaperMetadata]:
        """Rank papers by relevance, impact, and recency."""
        
        def calculate_relevance_score(paper: PaperMetadata) -> float:
            score = 0.0
            title_lower = paper.title.lower()
            abstract_lower = paper.abstract.lower()
            paper_title_query = self._extract_marker_value(topic_map, "__paper_title__:")
            metric_query = self._extract_marker_value(topic_map, "__metric__:")

            if paper_title_query:
                title_similarity = self._title_similarity(paper_title_query, paper.title)
                title_overlap = self._title_overlap(paper_title_query, paper.title)
                score += min((title_similarity * 0.7) + (title_overlap * 0.3), 0.9)
                return min(score, 1.0)

            if metric_query:
                metric_text = self._normalize_text(metric_query)
                searchable = self._normalize_text(f"{paper.title} {paper.abstract}")
                if metric_text and metric_text in searchable:
                    score += 0.45
            
            # Main topic match
            if topic_map.main_topic.lower() in title_lower:
                score += 0.3
            if topic_map.main_topic.lower() in abstract_lower:
                score += 0.2
            
            # Keyword matches
            for keyword in topic_map.keywords:
                if keyword.lower() in title_lower:
                    score += 0.1
                if keyword.lower() in abstract_lower:
                    score += 0.05
            
            # Recency bonus
            current_year = datetime.now().year
            if current_year - paper.year <= 5:
                score += 0.1 * (6 - (current_year - paper.year)) / 5
            
            # Impact score bonus
            if paper.impact_score > 0:
                score += min(paper.impact_score / 100, 0.2)
            
            # Venue impact (simplified)
            high_impact_venues = [
                "Nature", "Science", "Cell", "NEJM", "Lancet",
                "ICLR", "NeurIPS", "ICML", "AAAI", "IJCAI"
            ]
            if any(venue.lower() in paper.venue.lower() for venue in high_impact_venues):
                score += 0.2
            
            return min(score, 1.0)
        
        # Calculate relevance scores
        for paper in papers:
            if paper.relevance_score == 0.0:
                paper.relevance_score = calculate_relevance_score(paper)
        
        # Sort by relevance score (descending)
        ranked_papers = sorted(papers, key=lambda p: p.relevance_score, reverse=True)
        
        return ranked_papers
    
    def configure_api(self, api_name: str, api_key: str = None, **kwargs):
        """Configure API settings."""
        if api_name in self.apis:
            if api_key:
                self.apis[api_name]["api_key"] = api_key
                self.apis[api_name]["enabled"] = True
            
            for key, value in kwargs.items():
                if key in self.apis[api_name]:
                    self.apis[api_name][key] = value
            
            self.logger.info(f"Configured {api_name} API")
        else:
            self.logger.warning(f"Unknown API: {api_name}")
    
    def get_api_status(self) -> Dict[str, Any]:
        """Get status of all configured APIs."""
        status = {}
        for name, config in self.apis.items():
            status[name] = {
                "enabled": config["enabled"],
                "requires_key": config["requires_key"],
                "has_key": bool(config.get("api_key")),
                "ready": config["enabled"] and (not config["requires_key"] or bool(config.get("api_key")))
            }
        return status
