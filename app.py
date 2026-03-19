"""
Flask web frontend for the Autonomous Research Agent System.
"""
from flask import Flask, render_template, request, jsonify, send_file
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
import concurrent.futures

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.research_system import AutonomousResearchSystem
from src.agents.literature_builder_agent import LiteratureBuilderAgent

app = Flask(__name__)
app.config['SECRET_KEY'] = 'research-system-secret-key'

# Global system instance
research_system = None
executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

def get_system():
    """Get or create research system instance."""
    global research_system
    if research_system is None:
        research_system = AutonomousResearchSystem()
    return research_system


async def run_research_async(topic):
    """Run research asynchronously."""
    system = get_system()
    return await system.research(topic)


def run_research_sync(topic):
    """Run research synchronously in a new event loop."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        system = get_system()
        return loop.run_until_complete(system.research(topic))
    finally:
        loop.close()


def run_reference_validation(content, file_format, options):
    """Run reference validation in a separate thread."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        from src.agents.reference_validator import ReferenceValidator
        
        app.logger.info(f"Starting reference validation")
        validator = ReferenceValidator()
        result = loop.run_until_complete(validator.validate_references(content, file_format, options))
        
        app.logger.info(f"Reference validation completed. Valid: {len(result.valid_references)}, Invalid: {len(result.invalid_papers)}")
        
        # Convert to JSON-serializable format
        response = {
            'valid_references': [
                {
                    'key': ref.get('key', 'unknown'),
                    'authors': ref.get('authors', 'Unknown'),
                    'title': ref.get('title', 'Unknown'),
                    'journal': ref.get('journal', 'Unknown'),
                    'year': ref.get('year', 'Unknown'),
                    'volume': ref.get('volume', ''),
                    'issue': ref.get('issue', ''),
                    'pages': ref.get('pages', ''),
                    'doi': ref.get('doi', ''),
                    'original': ref.get('original', ''),
                    'corrected': ref.get('corrected', ''),
                    'corrections_made': ref.get('corrections_made', [])
                }
                for ref in result.valid_references
            ],
            'corrected_references': result.corrected_references,
            'duplicates_removed': result.duplicates_removed,
            'invalid_papers': result.invalid_papers,
            'corrections': [],
            'issues': []
        }
        
        # Process corrections for display
        corrections_by_ref = {}
        
        for ref in result.valid_references:
            ref_key = ref.get('key', 'unknown')
            
            if ref_key not in corrections_by_ref:
                corrections_by_ref[ref_key] = {
                    'reference_key': ref_key,
                    'title': ref.get('title', 'Unknown')[:80] + '...' if len(ref.get('title', '')) > 80 else ref.get('title', 'Unknown'),
                    'details': [],
                    'corrections_count': 0,
                    'processed_fields': set()
                }
            
            corrections_made = ref.get('corrections_made', [])
            
            for correction in corrections_made:
                field = correction.split(':')[0].strip() if ':' in correction else 'General'
                
                # Skip if we've already processed this field for this reference
                if field in corrections_by_ref[ref_key]['processed_fields']:
                    continue
                
                change = correction.split(':', 1)[1].strip() if ':' in correction else correction
                
                if change:
                    before = 'Not provided'
                    after = change
                    
                    if ' → ' in change:
                        before, after = change.split(' → ', 1)
                    
                    corrections_by_ref[ref_key]['details'].append({
                        'field': field,
                        'before': before.strip().strip("'\""),
                        'after': after.strip().strip("'\"")
                    })
                    corrections_by_ref[ref_key]['corrections_count'] += 1
                    corrections_by_ref[ref_key]['processed_fields'].add(field)
        
        # Clean up processed_fields from response
        for ref_key in corrections_by_ref:
            del corrections_by_ref[ref_key]['processed_fields']
        
        response['corrections'] = list(corrections_by_ref.values())
        
        # Process issues for display
        for duplicate in result.duplicates_removed:
            response['issues'].append({
                'type': 'Duplicate',
                'reference_key': duplicate['reference'].get('key', 'unknown'),
                'description': duplicate['reason']
            })
        
        for invalid in result.invalid_papers:
            response['issues'].append({
                'type': 'Invalid',
                'reference_key': invalid['reference'].get('key', 'unknown'),
                'description': invalid['reason']
            })
        
        app.logger.info(f"Reference validation response prepared successfully")
        return response
        
    except Exception as e:
        app.logger.error(f"Error in reference validation: {str(e)}")
        raise e
        
    finally:
        loop.close()


def run_literature_generation(topic: str, filters: dict):
    """Run literature generation in a separate thread."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Get research system and run research
        app.logger.info(f"Starting research for topic: {topic}")
        system = get_system()
        research_results = loop.run_until_complete(system.research(topic))
        
        app.logger.info(f"Research completed. Found {len(research_results.papers)} papers, {len(research_results.claims)} claims")
        
        # Generate literature
        app.logger.info("Starting literature generation...")
        literature_agent = LiteratureBuilderAgent()
        literature_document = loop.run_until_complete(literature_agent.process(research_results))
        
        app.logger.info(f"Literature generation completed. Generated {len(literature_document.sections)} sections")
        
        # Get statistics
        stats = literature_agent.get_literature_stats(literature_document)
        
        # Convert to JSON-serializable format
        result = {
            'topic': topic,
            'outline': {
                'title': literature_document.outline.title,
                'sections': literature_document.outline.sections,
                'total_papers': literature_document.outline.total_papers,
                'total_claims': literature_document.outline.total_claims,
                'date_range': literature_document.outline.date_range,
                'estimated_word_count': getattr(literature_document.outline, 'estimated_word_count', 0)
            },
            'sections': [
                {
                    'section_type': section.section_type,
                    'title': section.title,
                    'content': section.content,
                    'citations': section.citations,
                    'claim_ids': section.claim_ids,
                    'word_count': section.word_count
                }
                for section in literature_document.sections
            ],
            'bibliography': literature_document.bibliography,
            'metadata': literature_document.metadata,
            'stats': stats,
            'generated_at': literature_document.generated_at.isoformat()
        }
        
        app.logger.info(f"Literature generation result prepared successfully")
        return result
        
    except Exception as e:
        app.logger.error(f"Error in literature generation: {str(e)}")
        raise e
        
    finally:
        loop.close()


@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')


@app.route('/literature')
def literature_page():
    """Literature builder page."""
    return render_template('literature_builder.html')


@app.route('/validator')
def reference_validator_page():
    """Reference validator page."""
    return render_template('reference_validator.html')


@app.route('/corrections-details')
def corrections_details():
    """Corrections details page."""
    return render_template('corrections_details.html')


@app.route('/research', methods=['POST'])
def research():
    """Perform research on a topic."""
    try:
        data = request.get_json()
        topic = data.get('topic', '').strip()
        
        if not topic:
            return jsonify({'error': 'Topic is required'}), 400
        
        # Run research asynchronously
        try:
            # Try to get existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is running, we need to run in a thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_research_sync, topic)
                    results = future.result()
            else:
                results = loop.run_until_complete(run_research_async(topic))
        except RuntimeError:
            # No event loop exists, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                results = loop.run_until_complete(run_research_async(topic))
            finally:
                loop.close()
        
        # Convert results to JSON-serializable format
        response_data = {
            'topic': results.topic_map.main_topic,
            'generated_at': results.generated_at.isoformat(),
            'summary': {
                'papers_analyzed': results.total_papers_analyzed,
                'claims_extracted': results.total_claims_extracted,
                'contradictions_found': len(results.contradictions),
                'research_gaps_identified': len(results.research_gaps)
            },
            'topic_map': {
                'main_topic': results.topic_map.main_topic,
                'subtopics': results.topic_map.subtopics,
                'methods': results.topic_map.methods,
                'keywords': results.topic_map.keywords,
                'datasets': results.topic_map.datasets
            },
            'papers': [
                {
                    'title': paper.title,
                    'authors': paper.authors,
                    'year': paper.year,
                    'venue': paper.venue,
                    'relevance_score': paper.relevance_score,
                    'abstract': paper.abstract[:300] + '...' if len(paper.abstract) > 300 else paper.abstract,
                    'doi': paper.doi,
                    'arxiv_id': paper.arxiv_id,
                    'url': paper.url
                }
                for paper in results.papers
            ],
            'claims': [
                {
                    'statement': claim.statement,
                    'confidence': claim.confidence,
                    'metrics': claim.metrics,
                    'datasets': claim.datasets,
                    'conditions': claim.conditions
                }
                for claim in results.claims
            ],
            'contradictions': [
                {
                    'explanation': contradiction.explanation,
                    'type': contradiction.contradiction_type,
                    'severity': contradiction.severity
                }
                for contradiction in results.contradictions
            ],
            'research_gaps': [
                {
                    'description': gap.description,
                    'type': gap.gap_type,
                    'priority': gap.priority,
                    'potential_questions': gap.potential_questions
                }
                for gap in results.research_gaps
            ],
            'citations': [
                {
                    'paper_id': citation.paper_id,
                    'bibtex': citation.bibtex,
                    'apa': citation.apa,
                    'ieee': citation.ieee,
                    'mla': citation.mla
                }
                for citation in results.citations
            ]
        }
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"research_{timestamp}.json"
        filepath = Path("output") / filename
        
        with open(filepath, 'w') as f:
            json.dump(response_data, f, indent=2, default=str)
        
        response_data['download_url'] = f'/download/{filename}'
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/validate-references', methods=['POST'])
def validate_references():
    """Validate and correct uploaded reference file."""
    try:
        data = request.get_json()
        content = data.get('content', '').strip()
        file_format = data.get('format', 'bibtex')
        options = data.get('options', {})
        
        if not content:
            return jsonify({'error': 'No content provided'}), 400
        
        app.logger.info(f"Reference validation request: format={file_format}, length={len(content)}")
        
        # Run validation in a separate thread
        future = executor.submit(run_reference_validation, content, file_format, options)
        results = future.result(timeout=300)  # 5 minute timeout
        
        app.logger.info(f"Reference validation completed successfully")
        
        return jsonify(results)
        
    except concurrent.futures.TimeoutError:
        app.logger.error(f"Reference validation timed out")
        return jsonify({'error': 'Reference validation timed out. Please try with fewer references.'}), 500
    except Exception as e:
        app.logger.error(f"Error in reference validation: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Reference validation failed: {str(e)}'}), 500


@app.route('/generate-literature', methods=['POST'])
def generate_literature():
    """Generate structured literature from research results."""
    try:
        data = request.get_json()
        topic = data.get('topic', '').strip()
        filters = data.get('filters', {})
        
        if not topic:
            return jsonify({'error': 'Topic is required'}), 400
        
        app.logger.info(f"Literature generation request for topic: {topic}")
        
        # Run research and literature generation in a separate thread
        future = executor.submit(run_literature_generation, topic, filters)
        results = future.result(timeout=300)  # 5 minute timeout
        
        app.logger.info(f"Literature generated successfully for topic: {topic}")
        
        return jsonify(results)
        
    except concurrent.futures.TimeoutError:
        app.logger.error(f"Literature generation timed out for topic: {topic}")
        return jsonify({'error': 'Literature generation timed out. Please try a more specific topic.'}), 500
    except Exception as e:
        app.logger.error(f"Error in literature generation for topic '{topic}': {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Literature generation failed: {str(e)}'}), 500


@app.route('/download/<filename>')
def download_file(filename):
    """Download research results file."""
    try:
        filepath = Path("output") / filename
        if filepath.exists():
            return send_file(filepath, as_attachment=True)
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/history')
def history():
    """Get list of previous research results."""
    try:
        output_dir = Path("output")
        files = []
        
        if output_dir.exists():
            for file in output_dir.glob("research_*.json"):
                try:
                    with open(file, 'r') as f:
                        data = json.load(f)
                    
                    files.append({
                        'filename': file.name,
                        'topic': data.get('topic', 'Unknown'),
                        'generated_at': data.get('generated_at', ''),
                        'papers_count': len(data.get('papers', [])),
                        'claims_count': len(data.get('claims', [])),
                        'download_url': f'/download/{file.name}'
                    })
                except:
                    continue
        
        # Sort by date (newest first)
        files.sort(key=lambda x: x['generated_at'], reverse=True)
        
        return jsonify(files)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Create output directory
    Path("output").mkdir(exist_ok=True)
    
    print("🌐 Starting Research System Web Interface")
    print("📍 Open your browser to: http://localhost:5000")
    print("📚 Literature Builder: http://localhost:5000/literature")
    print("✓ Reference Validator: http://localhost:5000/validator")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
