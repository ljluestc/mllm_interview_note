#!/usr/bin/env python3
"""
RAG Engine for LLM Interview Notes
===================================

This module provides a complete RAG (Retrieval-Augmented Generation) system
for semantic search over LLM interview documentation.

Features:
- Embedding generation with multilingual support
- Vector similarity search
- Reranking for improved relevance
- Answer generation with source attribution

Usage:
    from rag_engine import RAGEngine
    
    rag = RAGEngine()
    rag.load_data('../data/processed')
    results = rag.search("什么是attention机制?", top_k=5)
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import warnings

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    warnings.warn("sentence-transformers not installed. Install with: pip install sentence-transformers")

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    warnings.warn("faiss not installed. Using numpy fallback. Install with: pip install faiss-cpu")


class RAGEngine:
    """Complete RAG system for LLM interview documentation"""
    
    def __init__(
        self,
        model_name: str = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        device: str = 'cpu'
    ):
        """
        Initialize RAG engine
        
        Args:
            model_name: Sentence transformer model for embeddings
            device: Device for model inference ('cpu' or 'cuda')
        """
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError("sentence-transformers required. Install with: pip install sentence-transformers")
        
        self.model = SentenceTransformer(model_name, device=device)
        self.documents = []
        self.qa_pairs = []
        self.doc_embeddings = None
        self.qa_embeddings = None
        self.doc_index = None
        self.qa_index = None
        self.device = device
        
        print(f"RAG Engine initialized with model: {model_name}")
        print(f"Device: {device}")
    
    def load_data(self, data_dir: str):
        """
        Load documents and Q&A pairs from processed data directory
        
        Args:
            data_dir: Path to directory containing JSONL files
        """
        data_path = Path(data_dir)
        
        # Load documents
        doc_file = data_path / 'all_documents.jsonl'
        if doc_file.exists():
            with open(doc_file, 'r', encoding='utf-8') as f:
                self.documents = [json.loads(line) for line in f]
            print(f"Loaded {len(self.documents)} documents")
        
        # Load Q&A pairs
        qa_file = data_path / 'all_qa_pairs.jsonl'
        if qa_file.exists():
            with open(qa_file, 'r', encoding='utf-8') as f:
                self.qa_pairs = [json.loads(line) for line in f]
            print(f"Loaded {len(self.qa_pairs)} Q&A pairs")
        
        if not self.documents and not self.qa_pairs:
            raise ValueError(f"No data found in {data_dir}")
    
    def generate_embeddings(self, batch_size: int = 32, save_to: Optional[str] = None):
        """
        Generate embeddings for all documents and Q&A pairs
        
        Args:
            batch_size: Batch size for encoding
            save_to: Optional directory to save embeddings
        """
        print("\nGenerating embeddings...")
        
        # Generate document embeddings
        if self.documents:
            print(f"Encoding {len(self.documents)} documents...")
            doc_texts = [
                f"{doc['title']} {doc['content'][:500]}"  # Use title + first 500 chars
                for doc in self.documents
            ]
            self.doc_embeddings = self.model.encode(
                doc_texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            print(f"Document embeddings shape: {self.doc_embeddings.shape}")
        
        # Generate Q&A embeddings
        if self.qa_pairs:
            print(f"\nEncoding {len(self.qa_pairs)} Q&A pairs...")
            qa_texts = [
                f"{qa['question']} {qa['short_answer']}"
                for qa in self.qa_pairs
            ]
            self.qa_embeddings = self.model.encode(
                qa_texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            print(f"Q&A embeddings shape: {self.qa_embeddings.shape}")
        
        # Save embeddings if requested
        if save_to:
            self.save_embeddings(save_to)
    
    def build_index(self):
        """Build FAISS index for fast similarity search"""
        if HAS_FAISS:
            print("\nBuilding FAISS indices...")
            
            # Build document index
            if self.doc_embeddings is not None:
                self.doc_index = faiss.IndexFlatIP(self.doc_embeddings.shape[1])
                # Normalize embeddings for cosine similarity
                faiss.normalize_L2(self.doc_embeddings)
                self.doc_index.add(self.doc_embeddings)
                print(f"Document index: {self.doc_index.ntotal} vectors")
            
            # Build Q&A index
            if self.qa_embeddings is not None:
                self.qa_index = faiss.IndexFlatIP(self.qa_embeddings.shape[1])
                faiss.normalize_L2(self.qa_embeddings)
                self.qa_index.add(self.qa_embeddings)
                print(f"Q&A index: {self.qa_index.ntotal} vectors")
        else:
            print("\nFAISS not available, using numpy fallback")
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        search_type: str = 'both',
        min_score: float = 0.0
    ) -> List[Dict]:
        """
        Search for relevant documents/Q&A pairs
        
        Args:
            query: Search query
            top_k: Number of results to return
            search_type: 'documents', 'qa', or 'both'
            min_score: Minimum similarity score threshold
        
        Returns:
            List of search results with scores
        """
        if self.doc_embeddings is None and self.qa_embeddings is None:
            raise ValueError("No embeddings available. Run generate_embeddings() first")
        
        # Encode query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        
        results = []
        
        # Search documents
        if search_type in ['documents', 'both'] and self.doc_embeddings is not None:
            doc_results = self._search_collection(
                query_embedding,
                self.doc_embeddings,
                self.documents,
                self.doc_index,
                top_k,
                'document'
            )
            results.extend(doc_results)
        
        # Search Q&A pairs
        if search_type in ['qa', 'both'] and self.qa_embeddings is not None:
            qa_results = self._search_collection(
                query_embedding,
                self.qa_embeddings,
                self.qa_pairs,
                self.qa_index,
                top_k,
                'qa'
            )
            results.extend(qa_results)
        
        # Sort by score and apply threshold
        results = [r for r in results if r['score'] >= min_score]
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return results[:top_k]
    
    def _search_collection(
        self,
        query_embedding: np.ndarray,
        embeddings: np.ndarray,
        collection: List[Dict],
        index: Optional[object],
        top_k: int,
        result_type: str
    ) -> List[Dict]:
        """Internal method to search a collection"""
        if HAS_FAISS and index is not None:
            # Use FAISS
            faiss.normalize_L2(query_embedding)
            scores, indices = index.search(query_embedding, min(top_k, len(collection)))
            scores = scores[0]
            indices = indices[0]
        else:
            # Use numpy fallback
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            scores = np.dot(embeddings_norm, query_norm.T).flatten()
            indices = np.argsort(scores)[::-1][:top_k]
            scores = scores[indices]
        
        results = []
        for idx, score in zip(indices, scores):
            if idx < len(collection):  # Ensure valid index
                item = collection[idx].copy()
                item['score'] = float(score)
                item['result_type'] = result_type
                results.append(item)
        
        return results
    
    def rerank(self, query: str, results: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Rerank results using cross-encoder for better relevance
        
        Args:
            query: Original query
            results: Initial search results
            top_k: Number of top results to return after reranking
        
        Returns:
            Reranked results
        """
        # Simple reranking based on keyword matching
        # For production, use a cross-encoder model
        query_words = set(query.lower().split())
        
        for result in results:
            # Get text content
            if result['result_type'] == 'document':
                text = f"{result.get('title', '')} {result.get('content', '')}"
            else:
                text = f"{result.get('question', '')} {result.get('detailed_answer', '')}"
            
            text_words = set(text.lower().split())
            
            # Calculate keyword overlap
            overlap = len(query_words & text_words) / max(len(query_words), 1)
            
            # Combine with similarity score
            result['rerank_score'] = 0.7 * result['score'] + 0.3 * overlap
        
        # Sort by rerank score
        results.sort(key=lambda x: x.get('rerank_score', x['score']), reverse=True)
        
        return results[:top_k]
    
    def generate_answer(self, query: str, top_k: int = 3) -> Dict:
        """
        Generate answer for query using RAG
        
        Args:
            query: User question
            top_k: Number of context documents to use
        
        Returns:
            Dict with answer and sources
        """
        # Search for relevant content
        results = self.search(query, top_k=top_k * 2, search_type='both')
        
        # Rerank
        results = self.rerank(query, results, top_k=top_k)
        
        # Build context from top results
        context_parts = []
        sources = []
        
        for i, result in enumerate(results):
            if result['result_type'] == 'qa':
                context_parts.append(f"[来源 {i+1}] {result['question']}\n{result['detailed_answer']}")
                sources.append({
                    'type': 'Q&A',
                    'title': result['question'],
                    'url': result.get('url', ''),
                    'score': result['score']
                })
            else:
                context_parts.append(f"[来源 {i+1}] {result['title']}\n{result['content'][:300]}...")
                sources.append({
                    'type': 'Document',
                    'title': result['title'],
                    'url': result.get('url', ''),
                    'score': result['score']
                })
        
        context = "\n\n".join(context_parts)
        
        # For now, return context (in production, pass to LLM for generation)
        return {
            'query': query,
            'context': context,
            'sources': sources,
            'answer': "基于以上检索到的内容，请使用LLM生成答案。"  # Placeholder
        }
    
    def save_embeddings(self, output_dir: str):
        """Save embeddings to disk"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if self.doc_embeddings is not None:
            np.save(output_path / 'doc_embeddings.npy', self.doc_embeddings)
            print(f"Saved document embeddings to {output_path / 'doc_embeddings.npy'}")
        
        if self.qa_embeddings is not None:
            np.save(output_path / 'qa_embeddings.npy', self.qa_embeddings)
            print(f"Saved Q&A embeddings to {output_path / 'qa_embeddings.npy'}")
    
    def load_embeddings(self, input_dir: str):
        """Load embeddings from disk"""
        input_path = Path(input_dir)
        
        doc_emb_file = input_path / 'doc_embeddings.npy'
        if doc_emb_file.exists():
            self.doc_embeddings = np.load(doc_emb_file)
            print(f"Loaded document embeddings: {self.doc_embeddings.shape}")
        
        qa_emb_file = input_path / 'qa_embeddings.npy'
        if qa_emb_file.exists():
            self.qa_embeddings = np.load(qa_emb_file)
            print(f"Loaded Q&A embeddings: {self.qa_embeddings.shape}")


def main():
    """Example usage"""
    print("=" * 80)
    print("LLM Interview Notes - RAG Engine Demo")
    print("=" * 80)
    
    # Initialize RAG engine
    rag = RAGEngine()
    
    # Load data
    rag.load_data('../data/processed')
    
    # Generate embeddings
    rag.generate_embeddings(save_to='../data/embeddings')
    
    # Build index
    rag.build_index()
    
    # Example queries
    queries = [
        "什么是attention机制?",
        "如何进行模型微调?",
        "什么是LoRA?",
        "解释Transformer架构"
    ]
    
    print("\n" + "=" * 80)
    print("Running example queries...")
    print("=" * 80)
    
    for query in queries:
        print(f"\n查询: {query}")
        print("-" * 80)
        
        results = rag.search(query, top_k=3)
        
        for i, result in enumerate(results):
            print(f"\n[{i+1}] Score: {result['score']:.4f}")
            print(f"Type: {result['result_type']}")
            
            if result['result_type'] == 'qa':
                print(f"Question: {result['question']}")
                print(f"Answer: {result['short_answer'][:150]}...")
            else:
                print(f"Title: {result['title']}")
                print(f"Content: {result['content'][:150]}...")


if __name__ == '__main__':
    main()
