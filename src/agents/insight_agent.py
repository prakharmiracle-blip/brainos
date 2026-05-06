"""
BrainOS — Insight Extraction Agent
Generates NotebookLM-style briefings from documents
"""

from typing import List, Dict, Any
import re
from datetime import datetime

from src.config import settings
from src.utils.logger import log
from src.utils.notebook_models import InsightResponse

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class InsightAgent:
    """
    Extract structured insights from document chunks.
    Similar to NotebookLM's briefing document feature.
    """
    
    def __init__(self):
        if settings.llm_provider == "openrouter" and OpenAI:
            self.client = OpenAI(
                api_key=settings.openrouter_api_key,
                base_url=settings.openrouter_base_url,
            )
        else:
            self.client = None
            log.warning("InsightAgent: OpenAI client not initialized")
    
    def extract(self, chunks: List[Dict[str, Any]]) -> InsightResponse:
        """
        Extract key insights from document chunks.
        
        Args:
            chunks: List of document chunks with content
            
        Returns:
            InsightResponse with structured insights
        """
        if not chunks:
            return InsightResponse(
                key_points=[],
                main_topics=[],
                important_dates=[],
                questions=[],
                summary="No content provided.",
                source_ids=[],
            )
        
        # Combine chunk content
        combined_text = "\n\n".join([
            chunk.get('content', '') for chunk in chunks[:20]  # Limit to first 20 chunks
        ])
        
        # Extract source IDs
        source_ids = list(set([
            chunk.get('metadata', {}).get('source_id', '')
            for chunk in chunks
            if chunk.get('metadata', {}).get('source_id')
        ]))
        
        # Generate insights using LLM
        if self.client:
            insights = self._generate_with_llm(combined_text)
        else:
            insights = self._generate_fallback(combined_text)
        
        insights['source_ids'] = source_ids
        
        return InsightResponse(**insights)
    
    def _generate_with_llm(self, text: str) -> Dict[str, Any]:
        """Use LLM to extract structured insights"""
        
        prompt = f"""Analyze the following content and extract structured insights:

CONTENT:
{text[:4000]}  # Limit token usage

Please provide:

1. KEY POINTS (3-5 most important takeaways)
2. MAIN TOPICS (3-5 central themes)
3. IMPORTANT DATES (any significant dates mentioned with context)
4. STUDY QUESTIONS (3-5 questions to test understanding)
5. BRIEF SUMMARY (2-3 sentences)

Format your response EXACTLY as:

KEY_POINTS:
- Point 1
- Point 2

MAIN_TOPICS:
- Topic 1
- Topic 2

IMPORTANT_DATES:
- Date: Context

QUESTIONS:
- Question 1
- Question 2

SUMMARY:
Summary text here
"""

        try:
            response = self.client.chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing documents and extracting key insights. Be concise and precise."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000,
            )
            
            result_text = response.choices[0].message.content
            return self._parse_llm_response(result_text)
            
        except Exception as e:
            log.error(f"LLM insight generation failed: {e}")
            return self._generate_fallback(text)
    
    def _parse_llm_response(self, text: str) -> Dict[str, Any]:
        """Parse structured LLM response"""
        
        result = {
            'key_points': [],
            'main_topics': [],
            'important_dates': [],
            'questions': [],
            'summary': '',
        }
        
        # Extract sections
        sections = {
            'KEY_POINTS': 'key_points',
            'MAIN_TOPICS': 'main_topics',
            'IMPORTANT_DATES': 'important_dates',
            'QUESTIONS': 'questions',
            'SUMMARY': 'summary',
        }
        
        current_section = None
        
        for line in text.split('\n'):
            line = line.strip()
            
            # Check for section headers
            for header, key in sections.items():
                if header in line:
                    current_section = key
                    break
            
            # Extract content
            if current_section and line.startswith('-'):
                item = line[1:].strip()
                if current_section == 'important_dates':
                    # Parse date format "Date: Context"
                    if ':' in item:
                        date, context = item.split(':', 1)
                        result[current_section].append({
                            'date': date.strip(),
                            'context': context.strip()
                        })
                elif current_section != 'summary':
                    result[current_section].append(item)
            elif current_section == 'summary' and line and not any(h in line for h in sections.keys()):
                result['summary'] += line + ' '
        
        result['summary'] = result['summary'].strip()
        
        return result
    
    def _generate_fallback(self, text: str) -> Dict[str, Any]:
        """Fallback: Extract basic insights without LLM"""
        
        # Simple extraction
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
        
        return {
            'key_points': sentences[:5] if sentences else ["No key points extracted"],
            'main_topics': self._extract_topics(text),
            'important_dates': self._extract_dates(text),
            'questions': [
                f"What are the main ideas in this content?",
                f"How does this relate to other topics?",
                f"What are the practical applications?"
            ],
            'summary': ' '.join(sentences[:3]) if sentences else "No summary available.",
        }
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract potential topics from text"""
        # Simple keyword extraction (in production, use NLP)
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        # Count frequency
        from collections import Counter
        common = Counter(words).most_common(5)
        return [word for word, count in common if count > 1]
    
    def _extract_dates(self, text: str) -> List[Dict[str, str]]:
        """Extract dates from text"""
        # Match common date formats
        date_pattern = r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4})\b'
        
        dates = []
        for match in re.finditer(date_pattern, text):
            date_str = match.group()
            # Get context (30 chars before and after)
            start = max(0, match.start() - 30)
            end = min(len(text), match.end() + 30)
            context = text[start:end].strip()
            
            dates.append({
                'date': date_str,
                'context': context
            })
        
        return dates[:5]  # Limit to 5 dates


# Global instance
insight_agent = InsightAgent()
