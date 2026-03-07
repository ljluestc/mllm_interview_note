# RAG Dataset Schema Documentation

## Overview

This document describes the schema for RAG (Retrieval-Augmented Generation) datasets in the LLM Interview Note repository.

## Dataset Structure

### 1. Document Schema (JSONL format)

Each line in the JSONL file represents a single document with the following structure:

```json
{
  "id": "string",                    // Unique identifier (e.g., "llm_basics_001")
  "category": "string",              // Main category (e.g., "大语言模型基础", "Transformer")
  "subcategory": "string",           // Subcategory (e.g., "attention", "分词")
  "title": "string",                 // Document title
  "content": "string",               // Full content text
  "questions": ["string"],           // List of related questions
  "keywords": ["string"],            // Important keywords for retrieval
  "difficulty": "string",            // "beginner" | "intermediate" | "advanced"
  "source_file": "string",           // Original markdown file path
  "url": "string",                   // Online documentation URL
  "last_updated": "string",          // ISO 8601 timestamp
  "metadata": {
    "word_count": "integer",
    "has_code": "boolean",
    "has_images": "boolean",
    "references": ["string"]
  }
}
```

### 2. Q&A Schema (JSONL format)

For interview question-answer pairs:

```json
{
  "id": "string",                    // Unique identifier (e.g., "qa_attention_001")
  "category": "string",              // Main category
  "subcategory": "string",           // Subcategory
  "difficulty": "string",            // "beginner" | "intermediate" | "advanced"
  "question": "string",              // The interview question
  "short_answer": "string",          // 1-2 sentence summary
  "detailed_answer": "string",       // Comprehensive answer
  "key_points": ["string"],          // Main concepts as bullet points
  "code_examples": ["string"],       // Code snippets if applicable
  "related_topics": ["string"],      // Related concepts
  "keywords": ["string"],            // Search keywords
  "source_file": "string",           // Original markdown file
  "url": "string",                   // Documentation URL
  "status": "string"                 // "verified" | "draft" | "needs_review"
}
```

### 3. Embedding Schema (Binary format)

Embeddings are stored in separate files with metadata:

```
data/embeddings/
  ├── documents.npy           # Document embeddings (numpy array)
  ├── documents_meta.json     # Document metadata with IDs
  ├── qa.npy                  # Q&A embeddings
  └── qa_meta.json            # Q&A metadata
```

## Categories

### Main Categories

1. **01.大语言模型基础** - LLM Basics
   - 语言模型 (Language Models)
   - 分词与词向量 (Tokenization & Word Embeddings)
   - NLP基础 (NLP Fundamentals)
   - 深度学习 (Deep Learning)

2. **02.大语言模型架构** - LLM Architecture
   - Transformer模型
   - 注意力机制 (Attention Mechanisms)
   - BERT
   - LLaMA系列
   - ChatGLM系列
   - MoE

3. **03.训练数据集** - Training Datasets

4. **04.分布式训练** - Distributed Training
   - 数据并行 (Data Parallelism)
   - 流水线并行 (Pipeline Parallelism)
   - 张量并行 (Tensor Parallelism)
   - DeepSpeed
   - Megatron

5. **05.有监督微调** - Supervised Fine-tuning
   - Prompting
   - Adapter Tuning
   - LoRA
   - 实战案例 (Practical Cases)

6. **06.推理** - Inference
   - vLLM
   - TGI (Text Generation Inference)
   - FasterTransformer
   - TensorRT-LLM
   - 推理优化技术

7. **07.强化学习** - Reinforcement Learning
   - RLHF
   - PPO
   - DPO

8. **08.检索增强RAG** - Retrieval-Augmented Generation
   - RAG技术
   - Agent技术

9. **09.大语言模型评估** - LLM Evaluation
   - 评测方法
   - 幻觉问题

10. **10.大语言模型应用** - LLM Applications
    - 思维链 (Chain-of-Thought)
    - LangChain

## Difficulty Levels

- **beginner**: Basic concepts, definitions, simple explanations
- **intermediate**: Technical details, implementation concepts, comparisons
- **advanced**: Deep technical knowledge, optimization, research-level content

## File Naming Conventions

### Documents
```
documents_{category}_{index}.jsonl
```
Example: `documents_transformer_001.jsonl`

### Q&A Pairs
```
qa_{category}_{index}.jsonl
```
Example: `qa_attention_001.jsonl`

### Combined Datasets
```
all_documents.jsonl
all_qa_pairs.jsonl
```

## Usage Examples

### Loading Data

```python
import json

# Load Q&A pairs
with open('data/processed/all_qa_pairs.jsonl', 'r', encoding='utf-8') as f:
    qa_pairs = [json.loads(line) for line in f]

# Filter by category
attention_qa = [qa for qa in qa_pairs if qa['subcategory'] == 'attention']
```

### Creating Embeddings

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

# Embed questions
questions = [qa['question'] for qa in qa_pairs]
embeddings = model.encode(questions, convert_to_numpy=True)
```

### Vector Search

```python
import numpy as np
from numpy.linalg import norm

def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

# Find similar questions
query_embedding = model.encode(["什么是attention机制?"])
similarities = [cosine_similarity(query_embedding[0], emb) for emb in embeddings]
top_k = np.argsort(similarities)[-5:][::-1]
```

## Quality Standards

1. **Completeness**: All required fields must be present
2. **Accuracy**: Technical information must be correct and up-to-date
3. **Clarity**: Answers should be clear and well-structured
4. **Consistency**: Use consistent terminology across documents
5. **Traceability**: Link back to source files for verification

## Version Control

Dataset versions are tracked using git tags:
- `v1.0.0` - Initial RAG dataset
- `v1.1.0` - Added new categories
- `v1.2.0` - Enhanced Q&A pairs

## Contact & Contributions

For questions or contributions, please open an issue or PR in the repository.
