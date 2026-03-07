#!/usr/bin/env python3
"""
Convert Markdown Documentation to RAG-ready JSONL Format
=========================================================

This script extracts content from markdown files and converts them into
structured JSON format suitable for RAG (Retrieval-Augmented Generation).

Usage:
    python convert_md_to_rag.py --input_dir ../ --output_dir ../data/processed
"""

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class MarkdownToRAGConverter:
    """Converts markdown files to RAG-ready JSON format"""
    
    def __init__(self, base_url: str = "http://wdndev.github.io/llm_interview_note"):
        self.base_url = base_url
        self.doc_counter = 0
        self.qa_counter = 0
        
    def extract_title(self, content: str) -> Optional[str]:
        """Extract title from markdown content"""
        # Try h1 first
        h1_match = re.search(r'^#\s+(.+?)$', content, re.MULTILINE)
        if h1_match:
            return h1_match.group(1).strip()
        
        # Try h2
        h2_match = re.search(r'^##\s+(.+?)$', content, re.MULTILINE)
        if h2_match:
            return h2_match.group(1).strip()
        
        return None
    
    def extract_sections(self, content: str) -> List[Dict[str, str]]:
        """Extract sections from markdown content"""
        sections = []
        
        # Split by headers (h2 and h3)
        pattern = r'^(#{2,3})\s+(.+?)$'
        matches = list(re.finditer(pattern, content, re.MULTILINE))
        
        for i, match in enumerate(matches):
            level = len(match.group(1))
            title = match.group(2).strip()
            start_pos = match.end()
            
            # Get content until next header
            if i < len(matches) - 1:
                end_pos = matches[i + 1].start()
            else:
                end_pos = len(content)
            
            section_content = content[start_pos:end_pos].strip()
            
            if section_content:
                sections.append({
                    'level': level,
                    'title': title,
                    'content': section_content
                })
        
        return sections
    
    def extract_code_blocks(self, content: str) -> List[str]:
        """Extract code blocks from content"""
        pattern = r'```[\w]*\n(.*?)```'
        matches = re.findall(pattern, content, re.DOTALL)
        return [match.strip() for match in matches]
    
    def extract_keywords(self, content: str) -> List[str]:
        """Extract important keywords from content"""
        keywords = set()
        
        # Extract words in bold or italic
        bold_italic = re.findall(r'\*\*(.+?)\*\*|\*(.+?)\*|__(.+?)__|_(.+?)_', content)
        for groups in bold_italic:
            for word in groups:
                if word:
                    keywords.add(word.strip())
        
        # Extract technical terms (CamelCase or snake_case)
        technical = re.findall(r'\b[A-Z][a-zA-Z0-9]+(?:[A-Z][a-zA-Z0-9]+)*\b|\b\w+_\w+\b', content)
        keywords.update([t for t in technical if len(t) > 2])
        
        return list(keywords)[:20]  # Limit to top 20
    
    def categorize_content(self, file_path: str) -> Tuple[str, str]:
        """Determine category and subcategory from file path"""
        parts = Path(file_path).parts
        
        category = ""
        subcategory = ""
        
        for part in parts:
            if part.startswith(('01.', '02.', '03.', '04.', '05.', '06.', '07.', '08.', '09.', '10.')):
                category = part
            elif not part.endswith('.md') and part != 'README.md':
                subcategory = part
        
        return category, subcategory
    
    def infer_difficulty(self, content: str, title: str) -> str:
        """Infer difficulty level based on content"""
        content_lower = content.lower()
        title_lower = title.lower() if title else ""
        
        # Beginner indicators
        beginner_keywords = ['基础', '概念', '简介', '入门', 'basic', 'introduction', '什么是']
        if any(kw in title_lower or kw in content_lower[:200] for kw in beginner_keywords):
            return "beginner"
        
        # Advanced indicators
        advanced_keywords = ['优化', '原理', '源码', '实现', '深入', 'advanced', 'optimization', 'implementation']
        if any(kw in title_lower or kw in content_lower[:200] for kw in advanced_keywords):
            return "advanced"
        
        return "intermediate"
    
    def extract_questions(self, content: str) -> List[str]:
        """Extract questions from content"""
        questions = []
        
        # Pattern for numbered questions
        pattern = r'(?:^|\n)(?:\d+[\.\、]|\*\s*)\s*(.+?\?)'
        matches = re.findall(pattern, content)
        questions.extend([q.strip() for q in matches])
        
        # Pattern for header questions
        header_pattern = r'^#{2,4}\s+(.+?\?)$'
        header_matches = re.findall(header_pattern, content, re.MULTILINE)
        questions.extend([q.strip() for q in header_matches])
        
        return questions[:10]  # Limit to top 10
    
    def is_qa_content(self, content: str, title: str) -> bool:
        """Determine if content is Q&A format"""
        if not title:
            return False
        
        title_lower = title.lower()
        content_lower = content.lower()
        
        # Check for Q&A indicators
        qa_indicators = [
            '面试题', '问题', 'interview', 'question', 'q&a', 'faq',
            '题目', '问答'
        ]
        
        if any(ind in title_lower for ind in qa_indicators):
            return True
        
        # Check if content has Q&A structure
        question_count = len(re.findall(r'(?:^|\n)(?:\d+[\.\、]|\*\s*).+?\?', content))
        return question_count >= 3
    
    def convert_to_document(self, file_path: str, content: str, relative_path: str) -> Dict:
        """Convert markdown to document format"""
        self.doc_counter += 1
        
        title = self.extract_title(content)
        category, subcategory = self.categorize_content(file_path)
        difficulty = self.infer_difficulty(content, title or "")
        keywords = self.extract_keywords(content)
        questions = self.extract_questions(content)
        code_blocks = self.extract_code_blocks(content)
        
        # Build URL
        url_path = relative_path.replace('.md', '').replace('\\', '/')
        url = f"{self.base_url}/{url_path}"
        
        # Remove markdown formatting for clean content
        clean_content = re.sub(r'```[\w]*\n.*?```', '[CODE]', content, flags=re.DOTALL)
        clean_content = re.sub(r'!\[.*?\]\(.*?\)', '[IMAGE]', clean_content)
        clean_content = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', clean_content)
        clean_content = re.sub(r'[*_]{1,2}(.+?)[*_]{1,2}', r'\1', clean_content)
        
        doc_id = f"doc_{category.split('.')[0] if '.' in category else 'misc'}_{self.doc_counter:04d}"
        
        return {
            "id": doc_id,
            "category": category,
            "subcategory": subcategory,
            "title": title or "Untitled",
            "content": clean_content.strip(),
            "questions": questions,
            "keywords": keywords,
            "difficulty": difficulty,
            "source_file": relative_path,
            "url": url,
            "last_updated": datetime.now().isoformat(),
            "metadata": {
                "word_count": len(content),
                "has_code": len(code_blocks) > 0,
                "has_images": '[IMAGE]' in clean_content,
                "references": []
            }
        }
    
    def convert_to_qa(self, file_path: str, content: str, relative_path: str) -> List[Dict]:
        """Convert markdown to Q&A format"""
        qa_pairs = []
        
        category, subcategory = self.categorize_content(file_path)
        title = self.extract_title(content)
        sections = self.extract_sections(content)
        
        # Extract Q&A from sections
        for section in sections:
            section_title = section['title']
            section_content = section['content']
            
            # Check if section is a question
            if '?' in section_title or any(kw in section_title.lower() for kw in ['什么', '如何', '为什么', 'what', 'how', 'why']):
                self.qa_counter += 1
                
                # Extract key points
                key_points = []
                bullet_pattern = r'(?:^|\n)[-*]\s+(.+?)(?=\n|$)'
                bullets = re.findall(bullet_pattern, section_content)
                key_points.extend([b.strip() for b in bullets[:5]])
                
                # Extract code examples
                code_examples = self.extract_code_blocks(section_content)
                
                # Create short answer (first paragraph)
                paragraphs = section_content.split('\n\n')
                short_answer = paragraphs[0].strip() if paragraphs else ""
                short_answer = re.sub(r'\[.*?\]\(.*?\)', '', short_answer)
                short_answer = re.sub(r'[*_]{1,2}', '', short_answer)
                
                # Keywords
                keywords = self.extract_keywords(section_title + " " + section_content)
                
                # Related topics
                related_topics = [s['title'] for s in sections if s['title'] != section_title][:5]
                
                qa_id = f"qa_{category.split('.')[0] if '.' in category else 'misc'}_{self.qa_counter:04d}"
                
                qa_pair = {
                    "id": qa_id,
                    "category": category,
                    "subcategory": subcategory,
                    "difficulty": self.infer_difficulty(section_content, section_title),
                    "question": section_title,
                    "short_answer": short_answer[:200] if short_answer else "See detailed answer",
                    "detailed_answer": section_content.strip(),
                    "key_points": key_points,
                    "code_examples": code_examples,
                    "related_topics": related_topics,
                    "keywords": keywords,
                    "source_file": relative_path,
                    "url": f"{self.base_url}/{relative_path.replace('.md', '')}",
                    "status": "verified"
                }
                
                qa_pairs.append(qa_pair)
        
        return qa_pairs
    
    def process_directory(self, input_dir: str, output_dir: str):
        """Process all markdown files in directory"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all markdown files
        md_files = list(input_path.rglob('*.md'))
        
        # Exclude certain files
        excluded = ['README.md', '_sidebar.md', '_navbar.md', 'index.md']
        md_files = [f for f in md_files if f.name not in excluded]
        
        print(f"Found {len(md_files)} markdown files")
        
        documents = []
        qa_pairs = []
        
        for md_file in md_files:
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if len(content.strip()) < 100:  # Skip very short files
                    continue
                
                relative_path = str(md_file.relative_to(input_path))
                print(f"Processing: {relative_path}")
                
                # Convert to document
                doc = self.convert_to_document(str(md_file), content, relative_path)
                documents.append(doc)
                
                # Try to extract Q&A if applicable
                title = self.extract_title(content)
                if self.is_qa_content(content, title or ""):
                    qa_list = self.convert_to_qa(str(md_file), content, relative_path)
                    qa_pairs.extend(qa_list)
                    print(f"  Extracted {len(qa_list)} Q&A pairs")
            
            except Exception as e:
                print(f"Error processing {md_file}: {e}")
        
        # Save documents
        doc_output = output_path / 'all_documents.jsonl'
        with open(doc_output, 'w', encoding='utf-8') as f:
            for doc in documents:
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
        
        print(f"\nSaved {len(documents)} documents to {doc_output}")
        
        # Save Q&A pairs
        if qa_pairs:
            qa_output = output_path / 'all_qa_pairs.jsonl'
            with open(qa_output, 'w', encoding='utf-8') as f:
                for qa in qa_pairs:
                    f.write(json.dumps(qa, ensure_ascii=False) + '\n')
            
            print(f"Saved {len(qa_pairs)} Q&A pairs to {qa_output}")
        
        # Save summary
        summary = {
            "total_documents": len(documents),
            "total_qa_pairs": len(qa_pairs),
            "categories": list(set(d['category'] for d in documents)),
            "generated_at": datetime.now().isoformat()
        }
        
        summary_output = output_path / 'dataset_summary.json'
        with open(summary_output, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"\nDataset Summary:")
        print(f"  Documents: {summary['total_documents']}")
        print(f"  Q&A Pairs: {summary['total_qa_pairs']}")
        print(f"  Categories: {len(summary['categories'])}")


def main():
    parser = argparse.ArgumentParser(description='Convert Markdown to RAG-ready JSONL')
    parser.add_argument('--input-dir', type=str, default='../',
                       help='Input directory containing markdown files')
    parser.add_argument('--output-dir', type=str, default='../data/processed',
                       help='Output directory for JSONL files')
    parser.add_argument('--base-url', type=str,
                       default='http://wdndev.github.io/llm_interview_note',
                       help='Base URL for documentation')
    
    args = parser.parse_args()
    
    converter = MarkdownToRAGConverter(base_url=args.base_url)
    converter.process_directory(args.input_dir, args.output_dir)
    
    print("\n✓ Conversion complete!")


if __name__ == '__main__':
    main()
