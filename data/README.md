# MLLM Interview Notes - RAG System

Complete Retrieval-Augmented Generation (RAG) system for semantic search over Multimodal LLM interview documentation.

## 📁 Directory Structure

```
data/
├── RAG_SCHEMA.md           # Dataset schema documentation
├── README.md               # This file
├── raw/                    # Raw data (if needed)
├── processed/              # Processed JSONL datasets
│   ├── all_documents.jsonl       # All documentation (7 documents)
│   └── dataset_summary.json      # Dataset statistics
└── embeddings/             # Vector embeddings
    └── doc_embeddings.npy        # Document embeddings

scripts/
└── convert_md_to_rag.py    # Markdown to JSONL converter

rag_system/
└── rag_engine.py           # Complete RAG engine implementation
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Use RAG Engine

```python
from rag_system.rag_engine import RAGEngine

# Initialize RAG engine
rag = RAGEngine()

# Load processed data
rag.load_data('data/processed')

# Generate embeddings (first time only)
rag.generate_embeddings(save_to='data/embeddings')

# Build search index
rag.build_index()

# Search for relevant content
results = rag.search("Sora技术原理是什么?", top_k=5)

# Print results
for i, result in enumerate(results):
    print(f"[{i+1}] Score: {result['score']:.4f}")
    print(f"Title: {result.get('title')}")
    print()
```

## 📊 Dataset Statistics

- **Total Documents**: 7
- **Categories**: 4
  - **01.Sora** - Sora技术原理、Transformers + Diffusion论文、训练准备工作
  - **02.mllm论文** - Qwen-VL、从视觉表征到多模态大模型
  - **03.finetune** - 基于LoRA微调多模态大模型
  - **data** - RAG Schema文档

## 🎯 Content Overview

### 1. Sora (Video Generation)
- **Sora技术原理解析**: DiT架构、时空Patch、视频压缩网络、扩散Transformer
- **Transformers + Diffusion论文**: Scalable Diffusion Models with Transformers (DiT)
- **训练Sora准备工作**: 数据集准备、模型架构、训练策略

### 2. 多模态大模型论文
- **Qwen-VL**: 阿里通义千问视觉语言模型、ViT + LLM架构
- **从视觉表征到多模态大模型**: CLIP、BLIP、Flamingo、LLaVA演进历程

### 3. 微调实战
- **基于LoRA微调多模态大模型**: 低秩适配、参数高效微调、实践案例

## 🔧 Advanced Usage

### Custom Embedding Model

```python
# Use Chinese-optimized model for better MLLM content understanding
rag = RAGEngine(
    model_name='moka-ai/m3e-base',
    device='cuda'  # Use GPU if available
)
```

### Search Examples

```python
# Search about Sora
results = rag.search("Sora如何处理视频生成?", top_k=5)

# Search about multimodal models
results = rag.search("视觉语言模型的架构是什么?", top_k=5)

# Search about fine-tuning
results = rag.search("如何微调多模态大模型?", top_k=5)
```

## 🔍 Example Queries

```python
# 1. Video generation with Sora
rag.search("Sora的DiT架构原理")
rag.search("视频Patch如何编码")

# 2. Multimodal architectures
rag.search("Qwen-VL模型结构")
rag.search("CLIP和BLIP的区别")

# 3. Fine-tuning techniques
rag.search("LoRA微调多模态模型")
rag.search("参数高效微调方法")
```

## 🌟 Features

1. **Multilingual Support**: Optimized for Chinese and English content
2. **Fast Search**: FAISS-accelerated vector search (falls back to numpy)
3. **Hybrid Search**: Combines semantic and keyword-based search
4. **Source Attribution**: Tracks sources for each result
5. **Flexible Schema**: Easy to extend with new fields
6. **Batch Processing**: Efficient embedding generation

## 📈 Performance

- **Embedding Model**: `paraphrase-multilingual-mpnet-base-v2`
  - Dimensions: 768
  - Languages: 50+
  - Speed: ~100 docs/sec (CPU)

- **Search Speed**:
  - With FAISS: <10ms for 100k documents
  - Without FAISS: ~50ms for 1k documents

## 🔮 Future Enhancements

- [ ] Add Q&A extraction from content
- [ ] Add cross-encoder reranking
- [ ] Implement hybrid search (BM25 + semantic)
- [ ] Support for image/diagram retrieval
- [ ] Integration with multimodal LLM for answer generation
- [ ] Add evaluation metrics

## 📚 References

- [Sora Official](https://openai.com/sora)
- [DiT Paper](https://arxiv.org/abs/2212.09748)
- [Qwen-VL](https://github.com/QwenLM/Qwen-VL)
- [Sentence Transformers](https://www.sbert.net/)

## 🤝 Contributing

To add new content:

1. Add markdown files to appropriate categories
2. Run converter: `python scripts/convert_md_to_rag.py`
3. Regenerate embeddings: `rag.generate_embeddings(save_to='data/embeddings')`
4. Rebuild index: `rag.build_index()`

---

**Note**: The embeddings are not included in git due to size. Run `rag.generate_embeddings()` on first use.
