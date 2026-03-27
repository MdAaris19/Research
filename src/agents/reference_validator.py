"""
Reference Validator - Validates and corrects uploaded reference files.

This agent processes uploaded reference files to:
1. Check and correct format compliance
2. Fix capitalization and spelling mistakes
3. Detect and remove duplicate papers
4. Verify paper authenticity and data accuracy
"""

import re
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict
import difflib
from datetime import datetime

# Import requests with fallback
try:
    import requests
except ImportError:
    requests = None

# Try relative imports first, then absolute
try:
    from ..models.data_models import PaperMetadata
    from .base_agent import BaseAgent
except ImportError:
    # Fallback to absolute imports for testing
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from models.data_models import PaperMetadata
    
    # Create a simple base agent for testing
    class BaseAgent:
        def __init__(self, name, memory_store=None):
            self.name = name
            self.memory_store = memory_store
            self.logger = logger

logger = logging.getLogger(__name__)


class ReferenceValidationResult:
    """Result of reference validation process."""
    
    def __init__(self):
        self.original_count = 0
        self.corrected_references = []
        self.original_references = {}  # key -> original parsed ref for accurate before/after
        self.duplicates_removed = []
        self.format_corrections = []
        self.spelling_corrections = []
        self.capitalization_corrections = []
        self.verification_results = []
        self.invalid_papers = []
        self.final_count = 0
        self.processing_log = []

class ReferenceValidator(BaseAgent):
    """Agent for validating and correcting reference files."""
    
    def __init__(self, memory_store=None):
        super().__init__("ReferenceValidator", memory_store)
        self.common_words = self._load_common_words()
        self.journal_names = self._load_journal_names()
        self.author_patterns = self._load_author_patterns()
    
    async def process(self, input_data):
        """Required abstract method implementation."""
        # This method is required by BaseAgent but we use process_reference_file instead
        if isinstance(input_data, str):
            return await self.process_reference_file(input_data, 'bibitem')
        elif isinstance(input_data, dict):
            content = input_data.get('content', '')
            file_format = input_data.get('format', 'bibitem')
            return await self.process_reference_file(content, file_format)
        else:
            raise ValueError("Invalid input data for reference validation")
        
    def _load_common_words(self) -> Set[str]:
        """Load common words for spell checking."""
        return {
            # Common academic words
            'the', 'and', 'of', 'in', 'for', 'on', 'with', 'by', 'from', 'to', 'at',
            'analysis', 'approach', 'method', 'system', 'model', 'algorithm', 'framework',
            'learning', 'machine', 'deep', 'neural', 'network', 'networks', 'artificial',
            'intelligence', 'data', 'mining', 'processing', 'classification', 'prediction',
            'optimization', 'performance', 'evaluation', 'experimental', 'results',
            'application', 'applications', 'based', 'using', 'novel', 'improved',
            'efficient', 'effective', 'robust', 'scalable', 'automatic', 'automated',
            # Scientific terms
            'research', 'study', 'investigation', 'survey', 'review', 'comparison',
            'implementation', 'development', 'design', 'architecture', 'structure',
            'feature', 'features', 'extraction', 'selection', 'detection', 'recognition',
            'clustering', 'regression', 'supervised', 'unsupervised', 'reinforcement'
        }
    
    def _load_journal_names(self) -> Dict[str, str]:
        """Load common journal names and their correct formats."""
        return {
            # Nature journals
            'nature': 'Nature',
            'nature machine intelligence': 'Nature Machine Intelligence',
            'nature communications': 'Nature Communications',
            'nature methods': 'Nature Methods',
            
            # IEEE journals
            'ieee': 'IEEE',
            'ieee transactions on pattern analysis and machine intelligence': 'IEEE Transactions on Pattern Analysis and Machine Intelligence',
            'ieee transactions on neural networks and learning systems': 'IEEE Transactions on Neural Networks and Learning Systems',
            'ieee transactions on image processing': 'IEEE Transactions on Image Processing',
            'ieee transactions on knowledge and data engineering': 'IEEE Transactions on Knowledge and Data Engineering',
            
            # ACM journals
            'acm': 'ACM',
            'acm computing surveys': 'ACM Computing Surveys',
            'acm transactions on graphics': 'ACM Transactions on Graphics',
            
            # Other major journals
            'science': 'Science',
            'cell': 'Cell',
            'lancet': 'The Lancet',
            'pnas': 'Proceedings of the National Academy of Sciences',
            'journal of machine learning research': 'Journal of Machine Learning Research',
            'artificial intelligence': 'Artificial Intelligence',
            'machine learning': 'Machine Learning',
            
            # Conferences
            'neurips': 'Advances in Neural Information Processing Systems',
            'icml': 'International Conference on Machine Learning',
            'iclr': 'International Conference on Learning Representations',
            'aaai': 'AAAI Conference on Artificial Intelligence',
            'ijcai': 'International Joint Conference on Artificial Intelligence',
            'cvpr': 'IEEE Conference on Computer Vision and Pattern Recognition',
            'iccv': 'IEEE International Conference on Computer Vision',
            'eccv': 'European Conference on Computer Vision'
        }

    def _normalize_journal_name(self, journal: str) -> str:
        """Normalize a journal name for robust matching."""
        if not journal:
            return ''

        normalized = journal.lower().strip()
        normalized = re.sub(r'[^a-z0-9&+\-/\s]', ' ', normalized)
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized.strip()

    def _extract_embedded_journal_from_title(self, title: str) -> Tuple[str, str]:
        """Split a trailing journal name accidentally captured inside the title."""
        if not title:
            return title, ''

        stripped_title = title.strip().strip(' .,;')
        lower_title = stripped_title.lower()

        for known_journal, correct_name in sorted(self.journal_names.items(), key=lambda item: len(item[0]), reverse=True):
            match = re.search(rf'(?P<sep>[,.;:]\s+)(?P<journal>{re.escape(known_journal)})$', lower_title)
            if match:
                title_prefix = stripped_title[:match.start()].rstrip(' ,.;:')
                if title_prefix:
                    return title_prefix, correct_name

        generic_match = re.search(r'(?P<title>.+?),\s*(?P<journal>[^,]+)$', stripped_title)
        if generic_match:
            candidate_title = generic_match.group('title').strip()
            candidate_journal = generic_match.group('journal').strip()
            journal_words = candidate_journal.split()

            if (
                len(candidate_title.split()) >= 5
                and 2 <= len(journal_words) <= 8
                and all(
                    word.lower() in {'and', 'of', 'the', 'in', 'on', 'for', 'with', 'an', 'a'}
                    or word[:1].isupper()
                    for word in journal_words
                )
            ):
                return candidate_title, candidate_journal

        return stripped_title, ''

    def _normalize_text_for_match(self, text: str) -> str:
        """Normalize free text for similarity checks."""
        if not text:
            return ''

        normalized = text.lower()
        normalized = re.sub(r'[^a-z0-9\s]', ' ', normalized)
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized.strip()

    def _extract_author_surnames(self, authors: str) -> Set[str]:
        """Extract author surnames from a reference author string."""
        if not authors:
            return set()

        surnames = set()
        for author in authors.split(','):
            parts = [part for part in author.strip().split() if part]
            if parts:
                surnames.add(self._normalize_text_for_match(parts[-1]))
        return {name for name in surnames if name}

    def _extract_crossref_author_surnames(self, item: Dict[str, Any]) -> Set[str]:
        """Extract author surnames from a CrossRef item."""
        surnames = set()
        for author in item.get('author', []) or []:
            family = author.get('family', '')
            normalized = self._normalize_text_for_match(family)
            if normalized:
                surnames.add(normalized)
        return surnames

    def _extract_bibliographic_tail(self, clean_text: str, ref_data: Dict[str, Any]) -> str:
        """Extract volume/issue/year/pages or article number from the tail."""
        tail_patterns = [
            re.compile(
                r'^(?P<body>.+?)\s+\\textbf\{(?P<volume>[^}]+)\}(?:\((?P<issue>[^)]+)\))?\s*(?:\((?P<year>\d{4})\))?\s*(?P<pages>[A-Za-z]?\d+(?:\s*[-–—]{1,2}\s*\d+)?)?\.?$'
            ),
            re.compile(
                r'^(?P<body>.+?)\s+(?P<volume>\d+)(?:\((?P<issue>[^)]+)\))?\s*\((?P<year>\d{4})\)\s*(?P<pages>[A-Za-z]?\d+(?:\s*[-–—]{1,2}\s*\d+)?)?\.?$'
            ),
            re.compile(
                r'^(?P<body>.+?)\s+\\textbf\{(?P<volume>[^}]+)\}(?:\((?P<issue>[^)]+)\))?\s+(?P<pages>[A-Za-z]?\d+(?:\s*[-–—]{1,2}\s*\d+)?)\.?$'
            ),
        ]

        text = clean_text.strip().rstrip('.')
        for pattern in tail_patterns:
            match = pattern.match(text)
            if not match:
                continue

            groups = match.groupdict()
            if groups.get('volume'):
                ref_data['volume'] = groups['volume'].strip()
            if groups.get('issue'):
                ref_data['issue'] = groups['issue'].strip()
            if groups.get('year'):
                ref_data['year'] = int(groups['year'])
            if groups.get('pages'):
                ref_data['pages'] = self._normalize_pages(groups['pages'])

            return groups['body'].strip().rstrip(',. ;')

        return clean_text

    def _normalize_pages(self, pages: str) -> str:
        """Normalize page ranges/article numbers for consistent output."""
        if not pages:
            return pages

        normalized = re.sub(r'\s+', '', str(pages).strip())
        normalized = re.sub(r'(?<=\d)-(?=\d)', '--', normalized)
        normalized = re.sub(r'[–—]+', '--', normalized)
        return normalized

    def _score_crossref_candidate(self, ref: Dict[str, Any], item: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """Score a CrossRef candidate against the parsed reference."""
        crossref_title = ''
        if item.get('title'):
            crossref_title = item['title'][0] if isinstance(item['title'], list) else item['title']

        ref_title = self._normalize_text_for_match(ref.get('title', ''))
        normalized_crossref_title = self._normalize_text_for_match(crossref_title)
        title_similarity = difflib.SequenceMatcher(None, ref_title, normalized_crossref_title).ratio() if ref_title and normalized_crossref_title else 0.0

        ref_title_words = set(ref_title.split())
        crossref_title_words = set(normalized_crossref_title.split())
        title_overlap = len(ref_title_words & crossref_title_words) / max(len(ref_title_words), len(crossref_title_words), 1)

        crossref_journal = ''
        if item.get('container-title'):
            crossref_journal = item['container-title'][0] if isinstance(item['container-title'], list) else item['container-title']
        journal_similarity = difflib.SequenceMatcher(
            None,
            self._normalize_journal_name(ref.get('journal', '')),
            self._normalize_journal_name(crossref_journal),
        ).ratio() if ref.get('journal') and crossref_journal else 0.0

        ref_year = int(ref['year']) if str(ref.get('year', '')).isdigit() else None
        crossref_year = None
        if 'published-print' in item and item['published-print'].get('date-parts'):
            crossref_year = item['published-print']['date-parts'][0][0]
        elif 'published-online' in item and item['published-online'].get('date-parts'):
            crossref_year = item['published-online']['date-parts'][0][0]
        year_score = 0.0
        if ref_year and crossref_year:
            if ref_year == crossref_year:
                year_score = 1.0
            elif abs(ref_year - crossref_year) == 1:
                year_score = 0.5

        ref_surnames = self._extract_author_surnames(ref.get('authors', ''))
        crossref_surnames = self._extract_crossref_author_surnames(item)
        author_overlap = len(ref_surnames & crossref_surnames) / max(len(ref_surnames), len(crossref_surnames), 1) if ref_surnames and crossref_surnames else 0.0

        final_score = (title_similarity * 0.55) + (title_overlap * 0.15) + (author_overlap * 0.15) + (journal_similarity * 0.10) + (year_score * 0.05)
        return final_score, {
            'title_similarity': title_similarity,
            'title_overlap': title_overlap,
            'author_overlap': author_overlap,
            'journal_similarity': journal_similarity,
            'year_score': year_score,
        }

    def _is_crossref_candidate_acceptable(self, metrics: Dict[str, float]) -> bool:
        """Require strong title match plus supporting metadata before applying corrections."""
        title_similarity = metrics['title_similarity']
        title_overlap = metrics['title_overlap']
        author_overlap = metrics['author_overlap']
        journal_similarity = metrics['journal_similarity']
        year_score = metrics['year_score']

        strong_title = title_similarity >= 0.88 or (title_similarity >= 0.80 and title_overlap >= 0.75)
        support_signals = sum([
            author_overlap >= 0.5,
            journal_similarity >= 0.65,
            year_score >= 0.5,
        ])

        return strong_title and (support_signals >= 1 or title_similarity >= 0.94)

    def _split_authors_title_journal(self, clean_text: str, ref_data: Dict[str, Any]):
        """Split the remaining text into authors, title, and journal."""
        parts = [p.strip() for p in clean_text.split(',')]
        parts = [p for p in parts if p]

        if len(parts) >= 3:
            author_end_idx = 0
            for i, part in enumerate(parts):
                words = part.strip().split()
                has_initial = any('.' in w and len(w) <= 4 for w in words)
                is_short = len(words) <= 3
                if has_initial and is_short:
                    author_end_idx = i
                else:
                    break

            author_parts = parts[:author_end_idx + 1]
            remaining = parts[author_end_idx + 1:]

            if remaining:
                journal_start = None
                for idx in range(len(remaining) - 1, -1, -1):
                    candidate = ', '.join(remaining[idx:]).strip()
                    normalized_candidate = self._normalize_journal_name(candidate)
                    if normalized_candidate in self.journal_names:
                        journal_start = idx
                        break

                if journal_start is None:
                    journal_part = remaining[-1].strip()
                    title_parts = remaining[:-1]

                    if len(remaining) >= 2 and re.match(r'^(an|a|the)\b', remaining[-1].strip(), re.IGNORECASE):
                        prev_part = remaining[-2].strip()
                        if prev_part and prev_part[0].isupper():
                            journal_part = f"{prev_part}, {journal_part}"
                            title_parts = remaining[:-2]
                else:
                    journal_part = ', '.join(remaining[journal_start:]).strip()
                    title_parts = remaining[:journal_start]

                title_part = ', '.join(title_parts).strip()
                ref_data['authors'] = ', '.join(author_parts)
                if title_part:
                    ref_data['title'] = title_part
                if journal_part:
                    ref_data['journal'] = journal_part
                return

            ref_data['authors'] = ', '.join(author_parts)
            return

        if len(parts) == 2:
            ref_data['authors'] = parts[0].strip()
            if any(keyword in parts[1].lower() for keyword in ['journal', 'proceedings', 'conference']):
                ref_data['journal'] = parts[1].strip()
            else:
                ref_data['title'] = parts[1].strip()
        elif len(parts) == 1:
            ref_data['authors'] = parts[0].strip()
    
    def _load_author_patterns(self) -> List[str]:
        """Load common author name patterns for validation."""
        return [
            r'^[A-Z]\. [A-Z][a-z]+$',  # F. Last
            r'^[A-Z]\.[A-Z]\. [A-Z][a-z]+$',  # F.M. Last
            r'^[A-Z][a-z]+ [A-Z][a-z]+$',  # First Last
            r'^[A-Z]\. [A-Z]\. [A-Z][a-z]+$',  # F. M. Last
        ]
    
    async def process_reference_file(self, file_content: str, file_format: str = 'bibtex') -> ReferenceValidationResult:
        """
        Process and validate a reference file.
        
        Args:
            file_content: Content of the reference file
            file_format: Format of the file ('bibtex', 'bibitem', 'plain')
            
        Returns:
            ReferenceValidationResult with all corrections and validations
        """
        result = ReferenceValidationResult()
        result.processing_log.append(f"Starting validation of {file_format} reference file")
        
        try:
            # Step 1: Parse references from file
            references = await self._parse_references(file_content, file_format)
            result.original_count = len(references)
            result.processing_log.append(f"Parsed {len(references)} references")
            
            # Store original parsed references for accurate before/after comparison
            import copy
            result.original_references = {
                ref.get('original_key') or ref.get('key', f'ref_{i}'): copy.deepcopy(ref)
                for i, ref in enumerate(references)
            }
            
            # Step 2: Detect and remove duplicates
            unique_references = await self._remove_duplicates(references, result)
            result.processing_log.append(f"Removed {len(references) - len(unique_references)} duplicates")
            
            # Step 3: Format correction
            format_corrected = await self._correct_format(unique_references, result)
            result.processing_log.append(f"Applied format corrections to {len(result.format_corrections)} references")
            
            # Step 4: Spelling and capitalization correction
            spell_corrected = await self._correct_spelling_and_caps(format_corrected, result)
            result.processing_log.append(f"Applied spelling/capitalization corrections to {len(result.spelling_corrections)} references")
            
            # Step 5: Verify paper authenticity and data
            verified_references = await self._verify_papers(spell_corrected, result)
            result.processing_log.append(f"Verified {len(verified_references)} references, found {len(result.invalid_papers)} invalid")
            
            # Step 6: Generate final corrected references
            result.corrected_references = verified_references
            result.final_count = len(verified_references)
            
            result.processing_log.append(f"Validation complete: {result.original_count} → {result.final_count} references")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing reference file: {str(e)}")
            result.processing_log.append(f"Error: {str(e)}")
            raise e
    
    async def _parse_references(self, content: str, file_format: str) -> List[Dict[str, Any]]:
        """Parse references from file content based on format."""
        references = []
        
        if file_format.lower() == 'bibitem':
            # Parse \bibitem{key} format
            bibitem_pattern = r'\\bibitem\{([^}]+)\}\s*(.+?)(?=\\bibitem|\Z)'
            matches = re.findall(bibitem_pattern, content, re.DOTALL)
            
            for key, ref_text in matches:
                parsed_ref = await self._parse_bibitem_reference(key.strip(), ref_text.strip())
                if parsed_ref:
                    references.append(parsed_ref)
        
        elif file_format.lower() == 'bibtex':
            # Parse BibTeX format
            bibtex_pattern = r'@(\w+)\{([^,]+),\s*(.+?)\n\}'
            matches = re.findall(bibtex_pattern, content, re.DOTALL)
            
            for entry_type, key, fields in matches:
                parsed_ref = await self._parse_bibtex_reference(entry_type, key.strip(), fields)
                if parsed_ref:
                    references.append(parsed_ref)
        
        else:  # plain text format
            # Split by lines and try to parse each line as a reference
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            for i, line in enumerate(lines):
                parsed_ref = await self._parse_plain_reference(f"ref_{i+1}", line)
                if parsed_ref:
                    references.append(parsed_ref)
        
        return references
    
    async def _parse_bibitem_reference(self, key: str, ref_text: str) -> Optional[Dict[str, Any]]:
        """Parse a single bibitem reference with proper component extraction."""
        try:
            ref_data = {
                'key': key,
                'original_key': key,
                'original_text': ref_text,
                'format': 'bibitem'
            }
            
            # Clean the reference text
            clean_text = ref_text.strip()
            
            # Step 1: Extract DOI/URL first (at the end)
            doi_match = re.search(r'(https?://[^\s]+)\.?\s*$', clean_text)
            if doi_match:
                ref_data['doi'] = doi_match.group(1)
                clean_text = clean_text[:doi_match.start()].strip()
            
            # Step 2: Extract volume/issue/year/pages or article number from the bibliographic tail
            clean_text = self._extract_bibliographic_tail(clean_text, ref_data)

            # Step 3: Fallback year extraction when not captured in the tail
            if 'year' not in ref_data:
                year_matches = re.findall(r'\((\d{4})\)', clean_text)
                if year_matches:
                    ref_data['year'] = int(year_matches[-1])
                    year_pattern = r'\(' + str(year_matches[-1]) + r'\)'
                    clean_text = re.sub(year_pattern, '', clean_text, count=1, flags=re.IGNORECASE).strip()

            # Step 4: Fallback volume/issue extraction for unmatched patterns
            if 'volume' not in ref_data:
                volume_issue_match = re.search(r'\\textbf\{([^}]+)\}(?:\(([^)]+)\))?', clean_text)
                if volume_issue_match:
                    ref_data['volume'] = volume_issue_match.group(1).strip()
                    if volume_issue_match.group(2):
                        ref_data['issue'] = volume_issue_match.group(2).strip()
                    clean_text = re.sub(r'\\textbf\{[^}]+\}(?:\([^)]+\))?', '', clean_text).strip()

            # Step 5: Fallback page/article-number extraction
            if 'pages' not in ref_data:
                pages_match = re.search(r'\b(?:pp?\.\s*)?([A-Za-z]?\d+(?:[-–—]+\d+)?)\b\s*\.?\s*$', clean_text)
                if pages_match:
                    ref_data['pages'] = self._normalize_pages(pages_match.group(1))
                    clean_text = re.sub(r'\b(?:pp?\.\s*)?[A-Za-z]?\d+(?:[-–—]+\d+)?\b\s*\.?\s*$', '', clean_text).strip()

            # Step 6: Now parse Authors, Title, Journal from remaining text
            # Format: Authors, Title, Journal
            # Key insight: after removing year/volume/pages/doi, the structure is always:
            #   [author1, author2, ...], Title words possibly with commas, Journal Name
            # The journal is the LAST comma-segment that is NOT an author name.
            # Authors are identified by having initials (e.g. "S. Zafar", "M.H. Schultz").
            
            self._split_authors_title_journal(clean_text, ref_data)
            
            # Clean up extracted fields
            for field in ['authors', 'title', 'journal']:
                if field in ref_data:
                    # Remove extra whitespace and trailing punctuation
                    ref_data[field] = re.sub(r'\s+', ' ', ref_data[field]).strip(' .,;')

            if ref_data.get('title'):
                extracted_title, extracted_journal = self._extract_embedded_journal_from_title(ref_data['title'])
                ref_data['title'] = extracted_title
                if extracted_journal and not ref_data.get('journal'):
                    ref_data['journal'] = extracted_journal

            return ref_data
            
        except Exception as e:
            self.logger.warning(f"Failed to parse bibitem reference '{key}': {str(e)}")
            return None
    
    async def _parse_bibtex_reference(self, entry_type: str, key: str, fields: str) -> Optional[Dict[str, Any]]:
        """Parse a single BibTeX reference."""
        try:
            ref_data = {
                'key': key,
                'original_key': key,
                'entry_type': entry_type.lower(),
                'format': 'bibtex'
            }
            
            # Parse fields
            field_pattern = r'(\w+)\s*=\s*\{([^}]+)\}'
            field_matches = re.findall(field_pattern, fields)
            
            for field_name, field_value in field_matches:
                field_name = field_name.lower().strip()
                field_value = field_value.strip()
                
                if field_name == 'author':
                    ref_data['authors'] = field_value
                elif field_name == 'title':
                    ref_data['title'] = field_value
                elif field_name in ['journal', 'booktitle', 'venue']:
                    ref_data['journal'] = field_value
                elif field_name == 'year':
                    try:
                        ref_data['year'] = int(field_value)
                    except ValueError:
                        ref_data['year'] = field_value
                elif field_name == 'volume':
                    ref_data['volume'] = field_value
                elif field_name in ['number', 'issue']:
                    ref_data['issue'] = field_value
                elif field_name == 'pages':
                    ref_data['pages'] = field_value
                elif field_name == 'doi':
                    ref_data['doi'] = field_value
                elif field_name == 'url':
                    ref_data['url'] = field_value
            
            return ref_data
            
        except Exception as e:
            self.logger.warning(f"Failed to parse BibTeX reference '{key}': {str(e)}")
            return None
    
    async def _parse_plain_reference(self, key: str, ref_text: str) -> Optional[Dict[str, Any]]:
        """Parse a plain text reference."""
        try:
            ref_data = {
                'key': key,
                'original_key': key,
                'original_text': ref_text,
                'format': 'plain'
            }
            
            # Basic parsing - this is more heuristic
            # Try to identify year
            year_match = re.search(r'\((\d{4})\)', ref_text)
            if year_match:
                ref_data['year'] = int(year_match.group(1))
            
            # Try to identify DOI
            doi_match = re.search(r'https?://[^\s]+', ref_text)
            if doi_match:
                ref_data['doi'] = doi_match.group(0)
            
            # Store full text for manual processing
            ref_data['full_text'] = ref_text
            
            return ref_data
            
        except Exception as e:
            self.logger.warning(f"Failed to parse plain reference '{key}': {str(e)}")
            return None
    
    async def _remove_duplicates(self, references: List[Dict[str, Any]], result: ReferenceValidationResult) -> List[Dict[str, Any]]:
        """Remove duplicate references based on comprehensive similarity analysis."""
        unique_refs = []
        seen_signatures = []
        
        for ref in references:
            # Create comprehensive signature for duplicate detection
            signature = self._create_comprehensive_signature(ref)
            
            # Check for duplicates using multiple criteria
            is_duplicate = False
            duplicate_reason = ""
            
            for i, seen_sig in enumerate(seen_signatures):
                similarity_score, match_type = self._calculate_similarity(signature, seen_sig)
                
                if similarity_score > 0.85:  # 85% similarity threshold
                    is_duplicate = True
                    duplicate_reason = f"Similar to reference #{i+1} ({match_type}, similarity: {similarity_score:.2f})"
                    result.duplicates_removed.append({
                        'reference': ref,
                        'reason': duplicate_reason,
                        'signature': signature,
                        'similar_to_index': i,
                        'similarity_score': similarity_score
                    })
                    break
            
            if not is_duplicate:
                unique_refs.append(ref)
                seen_signatures.append(signature)
        
        return unique_refs
    
    def _create_comprehensive_signature(self, ref: Dict[str, Any]) -> Dict[str, str]:
        """Create a comprehensive signature for duplicate detection."""
        signature = {}
        
        # Normalize title
        if 'title' in ref and ref['title']:
            title = re.sub(r'[^\w\s]', '', ref['title'].lower())
            title = ' '.join(title.split())  # Normalize whitespace
            signature['title'] = title
        
        # Normalize authors
        if 'authors' in ref and ref['authors']:
            authors = ref['authors'].lower()
            # Extract last names only for comparison
            author_parts = [part.strip() for part in authors.split(',')]
            last_names = []
            for author in author_parts:
                words = author.split()
                if words:
                    last_names.append(words[-1])  # Last word is usually the last name
            signature['authors'] = '|'.join(sorted(last_names))
        
        # Add year
        if 'year' in ref and ref['year']:
            signature['year'] = str(ref['year'])
        
        # Add DOI (most reliable identifier)
        if 'doi' in ref and ref['doi']:
            doi = ref['doi'].lower()
            doi = doi.replace('https://doi.org/', '').replace('http://dx.doi.org/', '').replace('doi:', '')
            signature['doi'] = doi
        
        # Add journal
        if 'journal' in ref and ref['journal']:
            journal = re.sub(r'[^\w\s]', '', ref['journal'].lower())
            signature['journal'] = ' '.join(journal.split())
        
        return signature
    
    def _calculate_similarity(self, sig1: Dict[str, str], sig2: Dict[str, str]) -> Tuple[float, str]:
        """Calculate similarity between two signatures and return match type."""
        # DOI match is definitive
        if sig1.get('doi') and sig2.get('doi'):
            if sig1['doi'] == sig2['doi']:
                return 1.0, "DOI match"
        
        # Calculate weighted similarity
        weights = {'title': 0.4, 'authors': 0.3, 'year': 0.1, 'journal': 0.2}
        total_similarity = 0
        total_weight = 0
        match_components = []
        
        for component, weight in weights.items():
            if component in sig1 and component in sig2:
                if component == 'year':
                    # Exact match for year
                    similarity = 1.0 if sig1[component] == sig2[component] else 0.0
                else:
                    # Text similarity for other components
                    similarity = difflib.SequenceMatcher(None, sig1[component], sig2[component]).ratio()
                
                total_similarity += similarity * weight
                total_weight += weight
                
                if similarity > 0.8:
                    match_components.append(component)
        
        if total_weight > 0:
            final_similarity = total_similarity / total_weight
            match_type = f"{', '.join(match_components)} similarity" if match_components else "partial match"
            return final_similarity, match_type
        
        return 0.0, "no match"
    
    async def _correct_format(self, references: List[Dict[str, Any]], result: ReferenceValidationResult) -> List[Dict[str, Any]]:
        """Correct format issues in references."""
        corrected_refs = []
        
        for ref in references:
            corrections = []
            corrected_ref = ref.copy()
            
            # Validate and fix bibitem key
            key_correction = self._validate_and_fix_bibitem_key(corrected_ref)
            if key_correction:
                corrections.append(f"{key_correction['field']}: '{key_correction['before']}' → '{key_correction['after']}'")
            
            # Correct author format
            if 'authors' in ref:
                original_authors = ref['authors']
                corrected_authors = self._correct_author_format(original_authors)
                if corrected_authors != original_authors:
                    corrected_ref['authors'] = corrected_authors
                    corrections.append(f"Authors: '{original_authors}' → '{corrected_authors}'")
            
            # Correct title format
            if 'title' in ref:
                original_title = ref['title']
                corrected_title = self._correct_title_format(original_title)
                if corrected_title != original_title:
                    corrected_ref['title'] = corrected_title
                    corrections.append(f"Title: '{original_title}' → '{corrected_title}'")
            
            # Correct journal format
            if 'journal' in ref:
                original_journal = ref['journal']
                corrected_journal = self._correct_journal_format(original_journal)
                if corrected_journal != original_journal:
                    corrected_ref['journal'] = corrected_journal
                    corrections.append(f"Journal: '{original_journal}' → '{corrected_journal}'")
            
            if corrections:
                result.format_corrections.append({
                    'reference_key': corrected_ref.get('key', 'unknown'),
                    'corrections': corrections
                })
            
            corrected_refs.append(corrected_ref)
        
        return corrected_refs
    
    def _correct_author_format(self, authors: str) -> str:
        """Correct author name format to F.M. Last style without changing case incorrectly."""
        if not authors:
            return authors
        
        # Split multiple authors
        author_list = [a.strip() for a in authors.split(',')]
        corrected_authors = []
        
        for author in author_list:
            if not author:
                continue
            
            # Check if already in correct format (F.M. Last)
            if re.match(r'^[A-Z]\.[A-Z]*\.?\s+[A-Z][a-zA-Z]+', author):
                corrected_authors.append(author)
                continue
            
            # Try to parse and correct
            parts = author.split()
            if len(parts) >= 2:
                # Last part is surname (keep original case)
                surname = parts[-1]
                
                # First parts are given names
                given_names = parts[:-1]
                initials = []
                
                for name in given_names:
                    if len(name) == 1:
                        initials.append(f"{name.upper()}.")
                    elif len(name) > 1:
                        # If it's already an initial (like "A."), keep it
                        if name.endswith('.') and len(name) <= 3:
                            initials.append(name.upper())
                        else:
                            initials.append(f"{name[0].upper()}.")
                
                corrected_author = f"{''.join(initials)} {surname}"
                corrected_authors.append(corrected_author)
            else:
                corrected_authors.append(author)  # Keep as-is if can't parse
        
        return ', '.join(corrected_authors)
    
    def _correct_title_format(self, title: str) -> str:
        """Convert titles to sentence case while preserving proper names."""
        if not title:
            return title
        
        # Remove extra whitespace
        title = ' '.join(title.split())

        proper_nouns = {
            'wiener': 'Wiener',
            'sierpinski': 'Sierpinski',
            'sierpiński': 'Sierpiński',
            'plos': 'PLoS',
            'gmres': 'GMRES',
            'naacl': 'NAACL',
            'bert': 'BERT',
            'neurips': 'NeurIPS',
            'ieee': 'IEEE',
            'acm': 'ACM',
        }

        def fix_word(word: str, is_first: bool) -> str:
            prefix_match = re.match(r'^[([{"\']*', word)
            suffix_match = re.search(r'[\])}",;:.!?]*$', word)
            prefix = prefix_match.group(0) if prefix_match else ''
            suffix = suffix_match.group(0) if suffix_match else ''
            core_start = len(prefix)
            core_end = len(word) - len(suffix) if suffix else len(word)
            core = word[core_start:core_end]

            if not core:
                return word

            lower_core = core.lower()

            if lower_core in proper_nouns:
                fixed_core = proper_nouns[lower_core]
            elif core.isupper() and len(core) > 1 and lower_core in {'plos', 'gmres', 'naacl', 'bert', 'neurips', 'ieee', 'acm'}:
                fixed_core = core
            elif '-' in core:
                fixed_core = '-'.join(fix_word(part, is_first and idx == 0) for idx, part in enumerate(core.split('-')))
            elif is_first:
                fixed_core = lower_core.capitalize()
            else:
                fixed_core = lower_core

            return f"{prefix}{fixed_core}{suffix}"

        words = title.split()
        corrected_words = [fix_word(word, i == 0) for i, word in enumerate(words)]
        return ' '.join(corrected_words)
    
    def _correct_journal_format(self, journal: str) -> str:
        """Correct journal name format."""
        if not journal:
            return journal
        
        journal_lower = self._normalize_journal_name(journal)
        
        # Check against known journal names
        if journal_lower in self.journal_names:
            return self.journal_names[journal_lower]
        
        # Check near-exact matches only; avoid broad substring matches like
        # "Engineering Science and Technology" -> "Science".
        for known_journal_lower, correct_name in self.journal_names.items():
            similarity = difflib.SequenceMatcher(None, journal_lower, known_journal_lower).ratio()
            if similarity >= 0.92:
                return correct_name
        
        # Basic capitalization correction
        words = journal.split()
        corrected_words = []
        
        for word in words:
            if word.lower() in ['of', 'the', 'and', 'in', 'on', 'for', 'with']:
                corrected_words.append(word.lower())
            else:
                corrected_words.append(word.capitalize())
        
        return ' '.join(corrected_words)
    
    def _generate_correct_bibitem_key(self, authors_list, year):
        """Generate correct bibitem key based on authors and year."""
        if not authors_list or not year:
            return f"Unknown{year % 100 if year else '00'}"
        
        # Parse authors list
        if isinstance(authors_list, str):
            authors = [a.strip() for a in authors_list.split(',')]
        else:
            authors = authors_list
        
        # Get first two letters of last names
        key_parts = []
        
        if len(authors) >= 3:
            # Take first 3 authors
            for i in range(min(3, len(authors))):
                author = authors[i].strip()
                if author:
                    # Extract last name (last word)
                    parts = author.split()
                    if parts:
                        last_name = parts[-1]  # Get last word as last name
                        if len(last_name) >= 2:
                            # First letter caps, second letter lowercase
                            key_part = last_name[0].upper() + last_name[1].lower()
                            key_parts.append(key_part)
                        elif len(last_name) == 1:
                            # Only one letter, use it twice
                            key_parts.append(last_name[0].upper() + last_name[0].lower())
        else:
            # Take all authors (1 or 2)
            for author in authors:
                author = author.strip()
                if author:
                    parts = author.split()
                    if parts:
                        last_name = parts[-1]  # Get last word as last name
                        if len(last_name) >= 2:
                            # First letter caps, second letter lowercase
                            key_part = last_name[0].upper() + last_name[1].lower()
                            key_parts.append(key_part)
                        elif len(last_name) == 1:
                            # Only one letter, use it twice
                            key_parts.append(last_name[0].upper() + last_name[0].lower())
        
        # Add year (last 2 digits)
        year_suffix = year % 100 if year else 0
        
        key = "".join(key_parts) + f"{year_suffix:02d}"
        return key
    
    def _validate_and_fix_bibitem_key(self, ref_data):
        """Validate and fix bibitem key based on authors."""
        if 'authors' in ref_data and 'year' in ref_data:
            correct_key = self._generate_correct_bibitem_key(ref_data['authors'], ref_data['year'])
            original_key = ref_data.get('key', '')
            
            if original_key != correct_key:
                ref_data['key'] = correct_key
                return {
                    'field': 'Bibitem Key',
                    'before': original_key,
                    'after': correct_key
                }
        return None
        """Correct journal name format."""
        if not journal:
            return journal
        
        journal_lower = journal.lower().strip()
        
        # Check against known journal names
        if journal_lower in self.journal_names:
            return self.journal_names[journal_lower]
        
        # Check partial matches
        for known_journal_lower, correct_name in self.journal_names.items():
            if known_journal_lower in journal_lower or journal_lower in known_journal_lower:
                return correct_name
        
        # Basic capitalization correction
        words = journal.split()
        corrected_words = []
        
        for word in words:
            if word.lower() in ['of', 'the', 'and', 'in', 'on', 'for', 'with']:
                corrected_words.append(word.lower())
            else:
                corrected_words.append(word.capitalize())
        
        return ' '.join(corrected_words)
    
    async def _correct_spelling_and_caps(self, references: List[Dict[str, Any]], result: ReferenceValidationResult) -> List[Dict[str, Any]]:
        """Correct spelling and capitalization errors."""
        corrected_refs = []
        
        for ref in references:
            corrections = []
            corrected_ref = ref.copy()
            
            # Check title spelling
            if 'title' in ref:
                original_title = ref['title']
                corrected_title = self._correct_spelling(original_title)
                if corrected_title != original_title:
                    corrected_ref['title'] = corrected_title
                    corrections.append(f"Title spelling: '{original_title}' → '{corrected_title}'")
            
            if corrections:
                result.spelling_corrections.append({
                    'reference_key': ref.get('key', 'unknown'),
                    'corrections': corrections
                })
            
            corrected_refs.append(corrected_ref)
        
        return corrected_refs
    
    def _correct_spelling(self, text: str) -> str:
        """Basic spelling correction for common academic terms."""
        if not text:
            return text
        
        # Common misspellings in academic papers
        corrections = {
            'machien': 'machine',
            'learing': 'learning',
            'algoritm': 'algorithm',
            'anaylsis': 'analysis',
            'performace': 'performance',
            'clasification': 'classification',
            'optimiztion': 'optimization',
            'recogntion': 'recognition',
            'procesing': 'processing',
            'netowrk': 'network',
            'artifical': 'artificial',
            'inteligence': 'intelligence',
            'expermental': 'experimental',
            'comparision': 'comparison',
            'implemention': 'implementation',
            'evalution': 'evaluation'
        }
        
        corrected = text
        for wrong, right in corrections.items():
            corrected = re.sub(r'\b' + re.escape(wrong) + r'\b', right, corrected, flags=re.IGNORECASE)
        
        return corrected
    
    async def _verify_papers(self, references: List[Dict[str, Any]], result: ReferenceValidationResult) -> List[Dict[str, Any]]:
        """Verify paper authenticity and data accuracy using comprehensive CrossRef validation."""
        verified_refs = []
        
        for i, ref in enumerate(references):
            self.logger.info(f"Verifying paper {i+1}/{len(references)}: {ref.get('key', 'unknown')}")
            
            verification_result = await self._verify_single_paper(ref)
            
            result.verification_results.append({
                'reference_key': ref.get('key', 'unknown'),
                'verification': verification_result
            })
            
            if verification_result['is_valid']:
                # Create corrected reference with verified data
                corrected_ref = ref.copy()
                
                # Apply all verified data corrections
                if verification_result.get('verified_data'):
                    verified_data = verification_result['verified_data']
                    
                    # Update with verified information
                    if 'verified_title' in verified_data:
                        corrected_ref['title'] = verified_data['verified_title']
                    if 'verified_authors' in verified_data:
                        corrected_ref['authors'] = verified_data['verified_authors']
                    if 'verified_journal' in verified_data:
                        corrected_ref['journal'] = verified_data['verified_journal']
                    if 'verified_year' in verified_data:
                        corrected_ref['year'] = verified_data['verified_year']
                    if 'verified_volume' in verified_data:
                        corrected_ref['volume'] = verified_data['verified_volume']
                    if 'verified_issue' in verified_data:
                        corrected_ref['issue'] = verified_data['verified_issue']
                    if 'verified_pages' in verified_data:
                        corrected_ref['pages'] = verified_data['verified_pages']
                    if 'verified_doi' in verified_data:
                        corrected_ref['doi'] = verified_data['verified_doi']
                
                # Add metadata about verification
                corrected_ref['verification_method'] = verification_result.get('search_method', 'Unknown')
                corrected_ref['corrections_applied'] = len(verification_result.get('corrections_made', []))
                
                verified_refs.append(corrected_ref)
                
                # Log corrections made
                if verification_result.get('corrections_made'):
                    self.logger.info(f"Applied {len(verification_result['corrections_made'])} corrections to {ref.get('key', 'unknown')}")
                    for correction in verification_result['corrections_made']:
                        self.logger.debug(f"  - {correction}")
            else:
                # Paper is invalid - add to invalid list with detailed reason
                invalid_reasons = verification_result.get('issues_found', ['Unknown validation error'])
                result.invalid_papers.append({
                    'reference': ref,
                    'reason': '; '.join(invalid_reasons),
                    'search_attempts': verification_result.get('checks_performed', [])
                })
                
                self.logger.warning(f"Paper {ref.get('key', 'unknown')} marked as invalid: {'; '.join(invalid_reasons)}")
        
        self.logger.info(f"Verification complete: {len(verified_refs)}/{len(references)} papers validated successfully")
        return verified_refs
    
    async def _verify_single_paper(self, ref: Dict[str, Any]) -> Dict[str, Any]:
        """Verify a single paper using CrossRef API with comprehensive validation."""
        verification = {
            'is_valid': False,
            'checks_performed': [],
            'issues_found': [],
            'verified_data': {},
            'search_method': None,
            'corrections_made': []
        }
        
        crossref_data = None
        
        # Step 1: Try to find paper by DOI first
        if 'doi' in ref and ref['doi'] and requests:
            crossref_data = await self._search_by_doi(ref['doi'], verification)
        
        # Step 2: If DOI search failed, try title search
        if not crossref_data and 'title' in ref and ref['title']:
            crossref_data = await self._search_by_title(ref, verification)
        
        # Step 3: If paper found, validate and correct all information
        if crossref_data:
            await self._validate_and_correct_paper_data(ref, crossref_data, verification)
        else:
            verification['issues_found'].append('Paper not found in CrossRef database')
            verification['is_valid'] = False
        
        return verification
    
    async def _search_by_doi(self, doi: str, verification: Dict) -> Dict:
        """Search for paper by DOI."""
        try:
            # Clean DOI
            clean_doi = doi.strip()
            if clean_doi.startswith('https://doi.org/'):
                clean_doi = clean_doi.replace('https://doi.org/', '')
            elif clean_doi.startswith('http://dx.doi.org/'):
                clean_doi = clean_doi.replace('http://dx.doi.org/', '')
            elif clean_doi.startswith('doi:'):
                clean_doi = clean_doi.replace('doi:', '')
            
            # Query CrossRef API
            crossref_url = f"https://api.crossref.org/works/{clean_doi}"
            headers = {
                'User-Agent': 'Research System/1.0 (mailto:research@example.com)',
                'Accept': 'application/json'
            }
            
            self.logger.info(f"Searching paper by DOI: {clean_doi}")
            response = requests.get(crossref_url, headers=headers, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                work = data.get('message', {})
                verification['checks_performed'].append('DOI verification successful')
                verification['search_method'] = 'DOI'
                self.logger.info(f"Paper found by DOI: {work.get('title', ['Unknown'])[0] if work.get('title') else 'Unknown'}")
                return work
            elif response.status_code == 404:
                verification['issues_found'].append('DOI not found in CrossRef')
                self.logger.warning(f"DOI not found: {clean_doi}")
            else:
                verification['issues_found'].append(f'CrossRef API error for DOI: {response.status_code}')
                self.logger.warning(f"CrossRef API error for DOI {clean_doi}: {response.status_code}")
            
            # Small delay for API rate limiting
            await asyncio.sleep(0.2)
            
        except Exception as e:
            verification['issues_found'].append(f'DOI search failed: {str(e)}')
            self.logger.error(f"Error searching by DOI {doi}: {str(e)}")
        
        return None
    
    async def _search_by_title(self, ref: Dict[str, Any], verification: Dict) -> Dict:
        """Search for paper by title using CrossRef API with improved matching."""
        try:
            # Clean and prepare title for search
            title = ref.get('title', '')
            clean_title = title.strip()
            # Remove common academic formatting
            clean_title = re.sub(r'[^\w\s]', ' ', clean_title)
            clean_title = ' '.join(clean_title.split())  # Normalize whitespace
            
            # Try multiple search strategies
            search_queries = [
                clean_title,  # Full title
                ' '.join(clean_title.split()[:10]),  # First 10 words
                ' '.join(clean_title.split()[:6]),   # First 6 words
            ]
            
            for query in search_queries:
                if len(query.strip()) < 10:  # Skip very short queries
                    continue
                    
                # Query CrossRef API with title search
                search_url = "https://api.crossref.org/works"
                params = {
                    'query.title': query,
                    'rows': 10,  # Get more results for better matching
                    'sort': 'relevance'
                }
                headers = {
                    'User-Agent': 'Research System/1.0 (mailto:research@example.com)',
                    'Accept': 'application/json'
                }
                
                self.logger.info(f"Searching paper by title: {query[:50]}...")
                response = requests.get(search_url, params=params, headers=headers, timeout=15)
                
                if response.status_code == 200:
                    data = response.json()
                    items = data.get('message', {}).get('items', [])
                    
                    if items:
                        # Find best match by title similarity
                        best_match = None
                        best_similarity = 0
                        best_metrics = None
                        
                        for item in items:
                            if 'title' in item and item['title']:
                                combined_score, metrics = self._score_crossref_candidate(ref, item)
                                if combined_score > best_similarity:
                                    best_similarity = combined_score
                                    best_match = item
                                    best_metrics = metrics
                        
                        if best_match and best_metrics and self._is_crossref_candidate_acceptable(best_metrics):
                            verification['checks_performed'].append(
                                f"Title search successful (score: {best_similarity:.2f}, title: {best_metrics['title_similarity']:.2f}, authors: {best_metrics['author_overlap']:.2f}, journal: {best_metrics['journal_similarity']:.2f})"
                            )
                            verification['search_method'] = 'Title'
                            crossref_title = best_match['title'][0] if isinstance(best_match['title'], list) else best_match['title']
                            self.logger.info(f"Paper found by title: {crossref_title}")
                            return best_match

                        if best_match and best_metrics:
                            verification['issues_found'].append(
                                f"Best title match rejected (title={best_metrics['title_similarity']:.2f}, authors={best_metrics['author_overlap']:.2f}, journal={best_metrics['journal_similarity']:.2f}, year={best_metrics['year_score']:.2f})"
                            )
                 
                # Small delay between queries
                await asyncio.sleep(0.3)
            
            verification['issues_found'].append('No similar titles found in search results')
            self.logger.warning(f"No similar titles found for: {title}")
            
        except Exception as e:
            verification['issues_found'].append(f'Title search failed: {str(e)}')
            self.logger.error(f"Error searching by title {title}: {str(e)}")
        
        return None
    
    async def _validate_and_correct_paper_data(self, ref: Dict, crossref_data: Dict, verification: Dict):
        """Validate and correct all paper information against CrossRef data."""
        verification['is_valid'] = True
        corrections = []
        verified_data = {}
        
        # Check if this is the most recent version by comparing years
        ref_year = ref.get('year')
        crossref_year = None
        
        if 'published-print' in crossref_data and crossref_data['published-print'].get('date-parts'):
            crossref_year = crossref_data['published-print']['date-parts'][0][0]
        elif 'published-online' in crossref_data and crossref_data['published-online'].get('date-parts'):
            crossref_year = crossref_data['published-online']['date-parts'][0][0]
        
        # Only update DOI if CrossRef version is more recent or if no DOI exists
        should_update_doi = False
        if not ref.get('doi'):
            should_update_doi = True
        elif crossref_year and ref_year and crossref_year > ref_year:
            should_update_doi = True
            corrections.append(f"DOI updated to more recent version: {ref_year} → {crossref_year}")
        
        # Validate and correct title
        if 'title' in crossref_data and crossref_data['title']:
            crossref_title = crossref_data['title'][0] if isinstance(crossref_data['title'], list) else crossref_data['title']
            verified_data['verified_title'] = crossref_title
            
            if 'title' in ref and ref['title']:
                similarity = difflib.SequenceMatcher(None, 
                                                   ref['title'].lower(), 
                                                   crossref_title.lower()).ratio()
                if similarity < 0.9:  # If less than 90% similar, it's a correction
                    corrections.append(f"Title corrected: '{ref['title']}' → '{crossref_title}'")
            else:
                corrections.append(f"Title added: '{crossref_title}'")
        
        # Validate and correct authors
        if 'author' in crossref_data and crossref_data['author']:
            crossref_authors = []
            for author in crossref_data['author']:
                if 'given' in author and 'family' in author:
                    given = author['given']
                    family = author['family']
                    # Format as F.M. Last (initials first, then last name)
                    initials = '.'.join([name[0].upper() for name in given.split() if name]) + '.'
                    crossref_authors.append(f"{initials} {family}")
                elif 'family' in author:
                    crossref_authors.append(author['family'])
            
            if crossref_authors:
                verified_data['verified_authors'] = ', '.join(crossref_authors)
                
                if 'authors' in ref and ref['authors']:
                    if ref['authors'] != verified_data['verified_authors']:
                        corrections.append(f"Authors corrected: '{ref['authors']}' → '{verified_data['verified_authors']}'")
                else:
                    corrections.append(f"Authors added: '{verified_data['verified_authors']}'")
        
        # Validate and correct journal
        if 'container-title' in crossref_data and crossref_data['container-title']:
            crossref_journal = crossref_data['container-title'][0] if isinstance(crossref_data['container-title'], list) else crossref_data['container-title']
            verified_data['verified_journal'] = crossref_journal
            
            if 'journal' in ref and ref['journal']:
                if ref['journal'].lower() != crossref_journal.lower():
                    corrections.append(f"Journal corrected: '{ref['journal']}' → '{crossref_journal}'")
            else:
                corrections.append(f"Journal added: '{crossref_journal}'")
        
        # Validate and correct year (only if CrossRef version is more recent)
        if crossref_year:
            verified_data['verified_year'] = crossref_year
            
            if 'year' in ref and ref['year']:
                if int(ref['year']) != crossref_year and crossref_year > int(ref['year']):
                    corrections.append(f"Year updated to more recent: {ref['year']} → {crossref_year}")
            else:
                corrections.append(f"Year added: {crossref_year}")
        
        # Validate and correct volume
        if 'volume' in crossref_data and crossref_data['volume']:
            crossref_volume = str(crossref_data['volume']).strip()
            verified_data['verified_volume'] = crossref_volume
            
            if 'volume' in ref and ref['volume']:
                if str(ref['volume']).strip() != crossref_volume:
                    corrections.append(f"Volume corrected: {ref['volume']} → {crossref_volume}")
            else:
                corrections.append(f"Volume added: {crossref_volume}")
        
        # Validate and correct issue
        if 'issue' in crossref_data and crossref_data['issue']:
            crossref_issue = str(crossref_data['issue']).strip()
            verified_data['verified_issue'] = crossref_issue
            
            if 'issue' in ref and ref['issue']:
                if str(ref['issue']).strip() != crossref_issue:
                    corrections.append(f"Issue corrected: {ref['issue']} → {crossref_issue}")
            else:
                corrections.append(f"Issue added: {crossref_issue}")
        
        # Validate and correct pages/article number
        if 'page' in crossref_data and crossref_data['page']:
            crossref_pages = self._normalize_pages(crossref_data['page'])
            verified_data['verified_pages'] = crossref_pages
            
            if 'pages' in ref and ref['pages']:
                if str(ref['pages']).strip() != crossref_pages:
                    corrections.append(f"Pages corrected: {ref['pages']} → {crossref_pages}")
            else:
                corrections.append(f"Pages added: {crossref_pages}")
        elif 'article-number' in crossref_data and crossref_data['article-number']:
            crossref_article = self._normalize_pages(crossref_data['article-number'])
            verified_data['verified_pages'] = crossref_article
            
            if 'pages' in ref and ref['pages']:
                if str(ref['pages']).strip() != crossref_article:
                    corrections.append(f"Article number corrected: {ref['pages']} → {crossref_article}")
            else:
                corrections.append(f"Article number added: {crossref_article}")
        
        # Add DOI if should update
        if should_update_doi and 'DOI' in crossref_data and crossref_data['DOI']:
            crossref_doi = crossref_data['DOI'].strip()
            verified_data['verified_doi'] = f"https://doi.org/{crossref_doi}"
            
            if 'doi' in ref and ref['doi']:
                ref_doi = ref['doi'].replace('https://doi.org/', '').replace('http://dx.doi.org/', '')
                if ref_doi != crossref_doi:
                    corrections.append(f"DOI updated: {ref['doi']} → https://doi.org/{crossref_doi}")
            else:
                corrections.append(f"DOI added: https://doi.org/{crossref_doi}")
        
        verification['verified_data'] = verified_data
        verification['corrections_made'] = corrections
        
        if corrections:
            self.logger.info(f"Made {len(corrections)} corrections to paper data")
        else:
            self.logger.info("Paper data is already accurate")
    
    def generate_corrected_file(self, result: ReferenceValidationResult, output_format: str = 'bibitem') -> str:
        """Generate corrected reference file in specified format."""
        if output_format.lower() == 'bibitem':
            return self._generate_bibitem_format(result.corrected_references)
        elif output_format.lower() == 'bibtex':
            return self._generate_bibtex_format(result.corrected_references)
        else:
            return self._generate_plain_format(result.corrected_references)
    
    def _generate_bibitem_format(self, references: List[Dict[str, Any]]) -> str:
        """Generate references in bibitem format with verified data."""
        output_lines = []
        
        for ref in references:
            # Generate bibitem key
            key = ref.get('key', 'unknown')
            
            # Build citation with verified/corrected data
            parts = []
            
            # Authors (use verified data if available)
            authors = ref.get('authors', 'Unknown Author')
            parts.append(authors)
            
            # Title (use verified data if available)
            title = ref.get('title', 'Unknown Title')
            parts.append(title)
            
            # Journal (use verified data if available)
            journal = ref.get('journal', 'Unknown Journal')
            
            # Volume and issue (use verified data if available)
            volume = ref.get('volume')
            issue = ref.get('issue')
            
            journal_part = journal
            if volume:
                if issue:
                    journal_part += f" \\textbf{{{volume}}}({issue})"
                else:
                    journal_part += f" \\textbf{{{volume}}}"
            
            parts.append(journal_part)
            
            # Year (use verified data if available)
            year = ref.get('year', 'Unknown Year')
            parts.append(f"({year})")
            
            # Pages (use verified data if available)
            pages = ref.get('pages')
            if pages:
                parts.append(pages)
            
            # Combine parts
            citation = f"\\bibitem{{{key}}} {', '.join(parts[:3])}"
            if len(parts) > 3:
                citation += f" {' '.join(parts[3:])}"
            
            # Add DOI (use verified data if available)
            doi = ref.get('doi')
            if doi:
                if not doi.startswith('http'):
                    doi = f"https://doi.org/{doi}"
                citation += f". {doi}"
            
            output_lines.append(citation)
        
        return '\n\n'.join(output_lines)
    
    def _generate_bibtex_format(self, references: List[Dict[str, Any]]) -> str:
        """Generate references in BibTeX format with verified data."""
        output_lines = []
        
        for ref in references:
            key = ref.get('key', 'unknown')
            entry_type = ref.get('entry_type', 'article')
            
            lines = [f"@{entry_type}{{{key},"]
            
            # Add fields (use verified/corrected data)
            if 'authors' in ref:
                lines.append(f"  author = {{{ref['authors']}}},")
            
            if 'title' in ref:
                lines.append(f"  title = {{{ref['title']}}},")
            
            if 'journal' in ref:
                lines.append(f"  journal = {{{ref['journal']}}},")
            
            if 'year' in ref:
                lines.append(f"  year = {{{ref['year']}}},")
            
            if 'volume' in ref:
                lines.append(f"  volume = {{{ref['volume']}}},")
            
            if 'issue' in ref:
                lines.append(f"  number = {{{ref['issue']}}},")
            
            if 'pages' in ref:
                lines.append(f"  pages = {{{ref['pages']}}},")
            
            if 'doi' in ref:
                doi = ref['doi'].replace('https://doi.org/', '').replace('http://dx.doi.org/', '')
                lines.append(f"  doi = {{{doi}}},")
            
            lines.append("}")
            output_lines.append('\n'.join(lines))
        
        return '\n\n'.join(output_lines)
    
    def _generate_plain_format(self, references: List[Dict[str, Any]]) -> str:
        """Generate references in plain text format with verified data."""
        output_lines = []
        
        for i, ref in enumerate(references, 1):
            # Build plain citation with verified/corrected data
            authors = ref.get('authors', 'Unknown Author')
            title = ref.get('title', 'Unknown Title')
            journal = ref.get('journal', 'Unknown Journal')
            year = ref.get('year', 'Unknown Year')
            
            citation = f"{i}. {authors}. {title}. {journal}"
            
            volume = ref.get('volume')
            issue = ref.get('issue')
            
            if volume:
                if issue:
                    citation += f" {volume}({issue})"
                else:
                    citation += f" {volume}"
            
            citation += f" ({year})"
            
            pages = ref.get('pages')
            if pages:
                citation += f": {pages}"
            
            doi = ref.get('doi')
            if doi:
                if not doi.startswith('http'):
                    doi = f"https://doi.org/{doi}"
                citation += f". {doi}"
            
            output_lines.append(citation)
        
        return '\n\n'.join(output_lines)
    
    def generate_validation_report(self, result: ReferenceValidationResult) -> str:
        """Generate a comprehensive validation report with detailed corrections."""
        report_lines = []
        
        report_lines.append("# Reference Validation Report")
        report_lines.append("=" * 50)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Summary
        report_lines.append("## Summary")
        report_lines.append(f"- Original references: {result.original_count}")
        report_lines.append(f"- Duplicates removed: {len(result.duplicates_removed)}")
        report_lines.append(f"- Format corrections: {len(result.format_corrections)}")
        report_lines.append(f"- Spelling corrections: {len(result.spelling_corrections)}")
        report_lines.append(f"- Invalid papers found: {len(result.invalid_papers)}")
        report_lines.append(f"- Final valid references: {result.final_count}")
        
        # Calculate total corrections made
        total_corrections = 0
        for verification in result.verification_results:
            if verification['verification'].get('corrections_made'):
                total_corrections += len(verification['verification']['corrections_made'])
        
        report_lines.append(f"- Total data corrections: {total_corrections}")
        report_lines.append("")
        
        # Processing log
        report_lines.append("## Processing Log")
        for log_entry in result.processing_log:
            report_lines.append(f"- {log_entry}")
        report_lines.append("")
        
        # Duplicates removed
        if result.duplicates_removed:
            report_lines.append("## Duplicates Removed")
            for i, dup in enumerate(result.duplicates_removed, 1):
                report_lines.append(f"{i}. Key: {dup['reference'].get('key', 'unknown')}")
                report_lines.append(f"   Reason: {dup['reason']}")
                if 'similarity_score' in dup:
                    report_lines.append(f"   Similarity Score: {dup['similarity_score']:.2f}")
                report_lines.append("")
        
        # Format corrections
        if result.format_corrections:
            report_lines.append("## Format Corrections")
            for i, correction in enumerate(result.format_corrections, 1):
                report_lines.append(f"{i}. Reference: {correction['reference_key']}")
                for corr in correction['corrections']:
                    report_lines.append(f"   - {corr}")
                report_lines.append("")
        
        # Spelling corrections
        if result.spelling_corrections:
            report_lines.append("## Spelling Corrections")
            for i, correction in enumerate(result.spelling_corrections, 1):
                report_lines.append(f"{i}. Reference: {correction['reference_key']}")
                for corr in correction['corrections']:
                    report_lines.append(f"   - {corr}")
                report_lines.append("")
        
        # Paper verification and data corrections
        if result.verification_results:
            report_lines.append("## Paper Verification and Data Corrections")
            for i, verification in enumerate(result.verification_results, 1):
                report_lines.append(f"{i}. Reference: {verification['reference_key']}")
                
                ver_result = verification['verification']
                report_lines.append(f"   Status: {'✅ Valid' if ver_result['is_valid'] else '❌ Invalid'}")
                
                if ver_result.get('search_method'):
                    report_lines.append(f"   Search Method: {ver_result['search_method']}")
                
                if ver_result.get('checks_performed'):
                    report_lines.append("   Checks Performed:")
                    for check in ver_result['checks_performed']:
                        report_lines.append(f"   - {check}")
                
                if ver_result.get('corrections_made'):
                    report_lines.append("   Data Corrections Made:")
                    for correction in ver_result['corrections_made']:
                        report_lines.append(f"   - {correction}")
                
                if ver_result.get('issues_found'):
                    report_lines.append("   Issues Found:")
                    for issue in ver_result['issues_found']:
                        report_lines.append(f"   - {issue}")
                
                report_lines.append("")
        
        # Invalid papers
        if result.invalid_papers:
            report_lines.append("## Invalid Papers")
            for i, invalid in enumerate(result.invalid_papers, 1):
                report_lines.append(f"{i}. Key: {invalid['reference'].get('key', 'unknown')}")
                report_lines.append(f"   Reason: {invalid['reason']}")
                if 'search_attempts' in invalid:
                    report_lines.append(f"   Search Attempts: {', '.join(invalid['search_attempts'])}")
                report_lines.append("")
        
        # Validation statistics
        report_lines.append("## Validation Statistics")
        
        # Search method breakdown
        search_methods = {}
        for verification in result.verification_results:
            method = verification['verification'].get('search_method', 'Unknown')
            search_methods[method] = search_methods.get(method, 0) + 1
        
        if search_methods:
            report_lines.append("### Paper Discovery Methods:")
            for method, count in search_methods.items():
                report_lines.append(f"- {method}: {count} papers")
            report_lines.append("")
        
        # Correction type breakdown
        correction_types = {}
        for verification in result.verification_results:
            corrections = verification['verification'].get('corrections_made', [])
            for correction in corrections:
                if 'corrected:' in correction:
                    field = correction.split(' corrected:')[0]
                    correction_types[field] = correction_types.get(field, 0) + 1
                elif 'added:' in correction:
                    field = correction.split(' added:')[0]
                    correction_types[f"{field} (added)"] = correction_types.get(f"{field} (added)", 0) + 1
        
        if correction_types:
            report_lines.append("### Data Correction Breakdown:")
            for correction_type, count in sorted(correction_types.items()):
                report_lines.append(f"- {correction_type}: {count} corrections")
            report_lines.append("")
        
        return '\n'.join(report_lines)
