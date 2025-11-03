#!/usr/bin/env python3
"""
Advanced DOCX Translation Pipeline
Integrates Google Translate API with document format preservation

Author: Lukas Kurniawan
Purpose: Enterprise-level document translation system
"""

import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Third-party imports
from google.cloud import translate_v2 as translate
from python_docx import Document
import redis
import asyncio
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TranslationConfig:
    """Configuration for translation pipeline"""
    source_lang: str
    target_lang: str
    batch_size: int = 100
    max_workers: int = 5
    preserve_formatting: bool = True
    quality_threshold: float = 0.85
    cache_results: bool = True

class DocumentTranslationPipeline:
    """Enterprise-grade document translation pipeline"""
    
    def __init__(self, config: TranslationConfig):
        self.config = config
        self.translate_client = translate.Client()
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        self.translation_cache = {}
        
    def extract_text_elements(self, doc_path: str) -> List[Dict]:
        """Extract text elements while preserving document structure"""
        try:
            doc = Document(doc_path)
            elements = []
            
            for i, paragraph in enumerate(doc.paragraphs):
                if paragraph.text.strip():
                    elements.append({
                        'type': 'paragraph',
                        'index': i,
                        'text': paragraph.text,
                        'formatting': self._extract_formatting(paragraph)
                    })
            
            # Extract tables
            for table_idx, table in enumerate(doc.tables):
                for row_idx, row in enumerate(table.rows):
                    for cell_idx, cell in enumerate(row.cells):
                        if cell.text.strip():
                            elements.append({
                                'type': 'table_cell',
                                'table_index': table_idx,
                                'row_index': row_idx,
                                'cell_index': cell_idx,
                                'text': cell.text,
                                'formatting': None
                            })
            
            logger.info(f"Extracted {len(elements)} text elements from {doc_path}")
            return elements
            
        except Exception as e:
            logger.error(f"Error extracting text from {doc_path}: {str(e)}")
            raise
    
    def _extract_formatting(self, paragraph) -> Dict:
        """Extract formatting information from paragraph"""
        try:
            return {
                'bold': paragraph.runs[0].bold if paragraph.runs else False,
                'italic': paragraph.runs[0].italic if paragraph.runs else False,
                'font_size': paragraph.runs[0].font.size if paragraph.runs else None,
                'alignment': paragraph.alignment
            }
        except:
            return {}
    
    def translate_batch(self, texts: List[str]) -> List[str]:
        """Translate a batch of texts with caching"""
        translations = []
        untranslated_texts = []
        cache_keys = []
        
        # Check cache first
        for text in texts:
            cache_key = f"trans:{self.config.source_lang}:{self.config.target_lang}:{hash(text)}"
            cache_keys.append(cache_key)
            
            if self.config.cache_results:
                cached_result = self.redis_client.get(cache_key)
                if cached_result:
                    translations.append(cached_result.decode('utf-8'))
                    continue
            
            untranslated_texts.append(text)
        
        # Translate uncached texts
        if untranslated_texts:
            try:
                result = self.translate_client.translate(
                    untranslated_texts,
                    source_language=self.config.source_lang,
                    target_language=self.config.target_lang
                )
                
                # Store in cache
                for i, translation in enumerate(result):
                    translated_text = translation['translatedText']
                    translations.append(translated_text)
                    
                    if self.config.cache_results:
                        cache_key = cache_keys[len(translations) - 1]
                        self.redis_client.setex(
                            cache_key, 
                            86400,  # 24 hours
                            translated_text
                        )
                        
            except Exception as e:
                logger.error(f"Translation API error: {str(e)}")
                raise
        
        return translations
    
    def parallel_translate(self, elements: List[Dict]) -> List[Dict]:
        """Translate elements in parallel for better performance"""
        translated_elements = []
        
        # Group elements into batches
        batches = [elements[i:i + self.config.batch_size] 
                  for i in range(0, len(elements), self.config.batch_size)]
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_batch = {
                executor.submit(self._translate_batch_elements, batch): batch 
                for batch in batches
            }
            
            for future in as_completed(future_to_batch):
                batch = future_to_batch[future]
                try:
                    translated_batch = future.result()
                    translated_elements.extend(translated_batch)
                except Exception as e:
                    logger.error(f"Batch translation failed: {str(e)}")
                    raise
        
        return translated_elements
    
    def _translate_batch_elements(self, elements: List[Dict]) -> List[Dict]:
        """Translate a batch of elements"""
        texts = [elem['text'] for elem in elements]
        translations = self.translate_batch(texts)
        
        for elem, translation in zip(elements, translations):
            elem['translated_text'] = translation
            elem['translation_confidence'] = self._calculate_confidence(elem['text'], translation)
        
        return elements
    
    def _calculate_confidence(self, original: str, translated: str) -> float:
        """Calculate translation confidence score"""
        # Simple confidence calculation based on length ratio and character diversity
        if not original or not translated:
            return 0.0
        
        length_ratio = min(len(translated), len(original)) / max(len(translated), len(original))
        
        # Check for untranslated content (still in source language)
        if original == translated:
            return 0.5  # Possible untranslated content
        
        return min(length_ratio + 0.3, 1.0)
    
    def rebuild_document(self, translated_elements: List[Dict], output_path: str) -> bool:
        """Rebuild document with translations while preserving formatting"""
        try:
            # Create new document
            new_doc = Document()
            
            # Group elements by type
            paragraphs = [elem for elem in translated_elements if elem['type'] == 'paragraph']
            table_cells = [elem for elem in translated_elements if elem['type'] == 'table_cell']
            
            # Add translated paragraphs
            for elem in sorted(paragraphs, key=lambda x: x['index']):
                p = new_doc.add_paragraph(elem['translated_text'])
                
                # Apply formatting if preserved
                if self.config.preserve_formatting and elem.get('formatting'):
                    self._apply_formatting(p, elem['formatting'])
            
            # Handle tables (simplified for demo)
            if table_cells:
                self._rebuild_tables(new_doc, table_cells)
            
            new_doc.save(output_path)
            logger.info(f"Document saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error rebuilding document: {str(e)}")
            return False
    
    def _apply_formatting(self, paragraph, formatting: Dict):
        """Apply formatting to paragraph"""
        try:
            if formatting.get('bold'):
                paragraph.runs[0].bold = True
            if formatting.get('italic'):
                paragraph.runs[0].italic = True
            if formatting.get('font_size'):
                paragraph.runs[0].font.size = formatting['font_size']
        except:
            pass  # Ignore formatting errors
    
    def _rebuild_tables(self, doc, table_cells: List[Dict]):
        """Rebuild tables with translated content"""
        # Simplified table reconstruction
        # In production, this would maintain exact table structure
        table = doc.add_table(rows=1, cols=1)
        for cell_elem in table_cells:
            # Add translated cell content
            # This is a simplified version
            pass
    
    def translate_document(self, input_path: str, output_path: str) -> Dict:
        """Main translation pipeline"""
        start_time = datetime.now()
        
        try:
            logger.info(f"Starting translation: {input_path} -> {output_path}")
            
            # Step 1: Extract text elements
            elements = self.extract_text_elements(input_path)
            
            # Step 2: Translate in parallel
            translated_elements = self.parallel_translate(elements)
            
            # Step 3: Quality check
            low_quality_count = sum(1 for elem in translated_elements 
                                  if elem.get('translation_confidence', 0) < self.config.quality_threshold)
            
            # Step 4: Rebuild document
            success = self.rebuild_document(translated_elements, output_path)
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            result = {
                'success': success,
                'processing_time': processing_time,
                'total_elements': len(elements),
                'low_quality_translations': low_quality_count,
                'average_confidence': sum(elem.get('translation_confidence', 0) 
                                        for elem in translated_elements) / len(translated_elements),
                'source_language': self.config.source_lang,
                'target_language': self.config.target_lang
            }
            
            logger.info(f"Translation completed in {processing_time:.2f} seconds")
            return result
            
        except Exception as e:
            logger.error(f"Translation pipeline failed: {str(e)}")
            return {'success': False, 'error': str(e)}

# Example usage
if __name__ == "__main__":
    config = TranslationConfig(
        source_lang='en',
        target_lang='id',
        batch_size=50,
        max_workers=3,
        preserve_formatting=True,
        quality_threshold=0.85
    )
    
    pipeline = DocumentTranslationPipeline(config)
    result = pipeline.translate_document(
        'input_document.docx',
        'translated_document.docx'
    )
    
    print(f"Translation result: {result}")