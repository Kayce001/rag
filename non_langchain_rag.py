#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轻量级RAG检索重排序系统
去除LangChain依赖，使用纯Python实现
适用于已预处理的TXT文档
"""
#nice
import os
import pickle
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import warnings
import torch
import numpy as np
from pathlib import Path
import hashlib

# 本地向量存储
import faiss

# VLLM和Transformers
try:
    from vllm import LLM, SamplingParams
    from vllm.inputs.data import TokensPrompt
    from transformers import AutoTokenizer
    print("✅ 成功导入所有必需库")
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请安装: pip install faiss-cpu vllm transformers torch")
    exit(1)

warnings.filterwarnings("ignore")


@dataclass
class Document:
    """轻量级文档类，替代LangChain的Document"""
    page_content: str
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class RAGConfig:
    """RAG系统配置类"""
    embed_model_path: str
    rerank_model_path: str
    txt_data_path: str
    vector_store_path: str = "./vector_store"
    embed_top_k: int = 10
    final_top_k: int = 5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class TxtDocumentLoader:
    """轻量级TXT文档加载器"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
    
    def load_documents(self) -> List[Document]:
        """加载所有TXT文档（包括子目录）"""
        documents = []
        
        if self.data_path.is_file() and self.data_path.suffix == '.txt':
            txt_files = [self.data_path]
        elif self.data_path.is_dir():
            # 递归查找所有子目录中的txt文件
            txt_files = list(self.data_path.rglob("*.txt"))
        else:
            raise ValueError(f"无效的数据路径: {self.data_path}")
        
        print(f"📁 发现 {len(txt_files)} 个TXT文件（包括子目录）")
        
        for txt_file in txt_files:
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                
                if content:
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": str(txt_file),
                            "filename": txt_file.name,
                            "file_size": len(content),
                            "file_hash": hashlib.md5(content.encode()).hexdigest()
                        }
                    )
                    documents.append(doc)
                    print(f"  ✅ 加载: {txt_file.name} ({len(content)} 字符)")
                else:
                    print(f"  ⚠️  跳过空文件: {txt_file.name}")
                    
            except Exception as e:
                print(f"  ❌ 加载失败: {txt_file.name} - {str(e)}")
        
        print(f"📚 总共加载 {len(documents)} 个有效文档")
        return documents


class QwenEmbeddings:
    """Qwen嵌入模型"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        print(f"📦 加载嵌入模型: {model_path}")
        self.model = LLM(model=model_path, task="embed")
        print("✅ 嵌入模型加载完成")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入文档列表"""
        if not texts:
            return []
        
        # 为文档添加指令前缀
        prefixed_texts = [
            f"Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: {text}"
            for text in texts
        ]
        
        print(f"🔄 正在向量化 {len(texts)} 个文档...")
        outputs = self.model.embed(prefixed_texts)
        embeddings = [output.outputs.embedding for output in outputs]
        print(f"✅ 文档向量化完成")
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """嵌入查询文本"""
        prefixed_text = f"Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: {text}"
        outputs = self.model.embed([prefixed_text])
        return outputs[0].outputs.embedding


class QwenReranker:
    """Qwen重排序器"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        print(f"📦 加载重排序模型: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = LLM(
            model=model_path,
            max_model_len=8192,
            trust_remote_code=True,
            gpu_memory_utilization=0.8
        )
        self._setup_reranker()
        print("✅ 重排序模型加载完成")
    
    def _setup_reranker(self):
        """设置重排序器参数"""
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)
        
        self.yes_token = self.tokenizer("yes", add_special_tokens=False).input_ids[0]
        self.no_token = self.tokenizer("no", add_special_tokens=False).input_ids[0]
        
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            logprobs=20,
            allowed_token_ids=[self.yes_token, self.no_token]
        )
    
    def _create_rerank_prompt(self, query: str, document: str, instruction: str) -> List[dict]:
        """创建重排序提示词"""
        return [
            {
                "role": "system",
                "content": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
            },
            {
                "role": "user",
                "content": f"<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {document}"
            }
        ]
    
    def score(self, query: str, document: str, 
              instruction: str = "Given a web search query, retrieve relevant passages that answer the query") -> float:
        """计算重排序评分"""
        messages = [self._create_rerank_prompt(query, document, instruction)]
        
        tokenized = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False
        )[0]
        
        full_tokens = tokenized + self.suffix_tokens
        token_prompt = TokensPrompt(prompt_token_ids=full_tokens)
        
        outputs = self.model.generate([token_prompt], self.sampling_params)
        logprobs = outputs[0].outputs[0].logprobs[-1]
        
        yes_logprob = logprobs.get(self.yes_token, type('obj', (object,), {'logprob': -10})()).logprob
        no_logprob = logprobs.get(self.no_token, type('obj', (object,), {'logprob': -10})()).logprob
        
        yes_prob = np.exp(yes_logprob)
        no_prob = np.exp(no_logprob)
        
        total_prob = yes_prob + no_prob
        score = yes_prob / total_prob if total_prob > 0 else 0.0
        
        return float(score)


class VectorStore:
    """轻量级向量存储"""
    
    def __init__(self, embeddings: QwenEmbeddings):
        self.embeddings = embeddings
        self.documents = []
        self.index = None
        self.dimension = None
    
    def add_documents(self, documents: List[Document]):
        """添加文档到向量存储"""
        if not documents:
            return
        
        # 提取文档内容
        texts = [doc.page_content for doc in documents]
        
        # 获取嵌入向量
        embeddings = self.embeddings.embed_documents(texts)
        
        # 初始化FAISS索引
        if self.index is None:
            self.dimension = len(embeddings[0])
            self.index = faiss.IndexFlatIP(self.dimension)  # 使用内积相似度
            print(f"📊 创建FAISS索引，维度: {self.dimension}")
        
        # 标准化向量并添加到索引
        embeddings_array = np.array(embeddings, dtype=np.float32)
        # L2标准化，使内积等价于余弦相似度
        faiss.normalize_L2(embeddings_array)
        
        self.index.add(embeddings_array)
        self.documents.extend(documents)
        
        print(f"📊 向量索引更新完成，总文档数: {len(self.documents)}")
    
    def similarity_search_with_score(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
        """相似度搜索"""
        if self.index is None or len(self.documents) == 0:
            return []
        
        # 查询向量化
        query_embedding = self.embeddings.embed_query(query)
        query_vector = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_vector)  # 标准化
        
        # 搜索
        k = min(k, len(self.documents))  # 确保k不超过文档总数
        scores, indices = self.index.search(query_vector, k)
        
        # 构建结果
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):  # 确保索引有效
                results.append((self.documents[idx], float(score)))
        
        return results
    
    def save_local(self, path: str):
        """保存向量存储到本地"""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 保存FAISS索引
        if self.index is not None:
            faiss.write_index(self.index, str(save_path / "index.faiss"))
        
        # 保存文档和元数据
        docs_data = []
        for doc in self.documents:
            docs_data.append({
                'page_content': doc.page_content,
                'metadata': doc.metadata
            })
        
        with open(save_path / "documents.pkl", 'wb') as f:
            pickle.dump(docs_data, f)
        
        # 保存配置信息
        config_info = {
            'dimension': self.dimension,
            'num_documents': len(self.documents)
        }
        with open(save_path / "config.json", 'w', encoding='utf-8') as f:
            json.dump(config_info, f, ensure_ascii=False, indent=2)
        
        print(f"💾 向量存储已保存到: {save_path}")
    
    def load_local(self, path: str) -> bool:
        """从本地加载向量存储"""
        load_path = Path(path)
        
        # 检查必要文件是否存在
        required_files = ['index.faiss', 'documents.pkl', 'config.json']
        for file in required_files:
            if not (load_path / file).exists():
                print(f"❌ 缺少文件: {file}")
                return False
        
        try:
            # 加载配置
            with open(load_path / "config.json", 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 加载FAISS索引
            self.index = faiss.read_index(str(load_path / "index.faiss"))
            self.dimension = config['dimension']
            
            # 加载文档
            with open(load_path / "documents.pkl", 'rb') as f:
                docs_data = pickle.load(f)
            
            self.documents = []
            for doc_data in docs_data:
                doc = Document(
                    page_content=doc_data['page_content'],
                    metadata=doc_data['metadata']
                )
                self.documents.append(doc)
            
            print(f"📂 向量存储加载成功: {len(self.documents)} 个文档，维度 {self.dimension}")
            return True
            
        except Exception as e:
            print(f"❌ 向量存储加载失败: {e}")
            return False


class RAGRetriever:
    """RAG检索器"""
    
    def __init__(self, vector_store: VectorStore, reranker: QwenReranker, 
                 embed_top_k: int = 10, final_top_k: int = 5):
        self.vector_store = vector_store
        self.reranker = reranker
        self.embed_top_k = embed_top_k
        self.final_top_k = final_top_k
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """检索相关文档"""
        print(f"\n🔍 开始RAG检索: '{query}'")
        print("="*60)
        
        # 第一步：向量检索
        print(f"📊 执行向量检索，获取top-{self.embed_top_k}文档...")
        embed_docs = self.vector_store.similarity_search_with_score(query, k=self.embed_top_k)
        
        if not embed_docs:
            print("⚠️  没有找到相关文档")
            return []
        
        print(f"✅ 向量检索完成，获得 {len(embed_docs)} 个候选文档")
        
        # 第二步：重排序
        print(f"🎯 执行重排序...")
        rerank_results = []
        
        for i, (doc, embed_score) in enumerate(embed_docs):
            rerank_score = self.reranker.score(query, doc.page_content)
            rerank_results.append((doc, embed_score, rerank_score))
            print(f"  文档 {i+1}/{len(embed_docs)}: 向量={embed_score:.3f}, 重排序={rerank_score:.3f}")
        
        # 按重排序评分排序
        rerank_results.sort(key=lambda x: x[2], reverse=True)
        
        # 返回最终结果
        final_docs = []
        for i, (doc, embed_score, rerank_score) in enumerate(rerank_results[:self.final_top_k]):
            doc.metadata.update({
                'embed_score': embed_score,
                'rerank_score': rerank_score,
                'final_rank': i + 1
            })
            final_docs.append(doc)
        
        print(f"✅ 重排序完成，返回top-{len(final_docs)}结果")
        return final_docs


class LightweightRAGSystem:
    """轻量级RAG系统"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.embeddings = None
        self.reranker = None
        self.vector_store = None
        self.retriever = None
        
        self._initialize_components()
    
    def _initialize_components(self):
        """初始化组件"""
        print("🚀 初始化轻量级RAG系统...")
        
        # 初始化嵌入模型
        self.embeddings = QwenEmbeddings(self.config.embed_model_path)
        
        # 初始化重排序器
        self.reranker = QwenReranker(self.config.rerank_model_path)
        
        # 初始化向量存储
        self.vector_store = VectorStore(self.embeddings)
        
        print("✅ 组件初始化完成")
    
    def _get_data_hash(self) -> str:
        """获取数据目录的哈希值，用于检测数据是否变化"""
        data_path = Path(self.config.txt_data_path)
        
        if data_path.is_file():
            with open(data_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        
        # 目录情况：递归计算所有txt文件的综合哈希
        hash_md5 = hashlib.md5()
        txt_files = sorted(data_path.rglob("*.txt"))  # 递归查找并排序确保一致性
        
        for txt_file in txt_files:
            try:
                with open(txt_file, 'rb') as f:
                    hash_md5.update(f.read())
                hash_md5.update(str(txt_file).encode())  # 包含文件名
            except Exception as e:
                print(f"警告：无法读取文件 {txt_file}: {e}")
        
        return hash_md5.hexdigest()
    
    def _check_vector_store_validity(self) -> bool:
        """检查向量存储是否有效且最新"""
        vector_path = Path(self.config.vector_store_path)
        hash_file = vector_path / "data_hash.txt"
        
        if not hash_file.exists():
            return False
        
        try:
            with open(hash_file, 'r') as f:
                stored_hash = f.read().strip()
            
            current_hash = self._get_data_hash()
            return stored_hash == current_hash
        
        except Exception:
            return False
    
    def build_vector_store(self, force_rebuild: bool = False) -> VectorStore:
        """构建或加载向量存储"""
        vector_path = Path(self.config.vector_store_path)
        
        # 检查是否需要重建
        need_rebuild = force_rebuild or not self._check_vector_store_validity()
        
        if not need_rebuild and vector_path.exists():
            print("📂 检测到有效的向量存储缓存...")
            if self.vector_store.load_local(str(vector_path)):
                print("✅ 向量存储加载成功，跳过重新构建")
                return self.vector_store
            else:
                print("⚠️  向量存储加载失败，将重新构建")
                need_rebuild = True
        
        if need_rebuild:
            print("🏗️  构建新的向量存储...")
            
            # 加载文档
            loader = TxtDocumentLoader(self.config.txt_data_path)
            documents = loader.load_documents()
            
            if not documents:
                raise ValueError("没有找到有效的文档")
            
            # 直接使用原始文档（因为已经预处理好了）
            print(f"📄 使用预处理文档: {len(documents)} 个文档")
            
            # 构建向量存储
            print("🔨 开始向量化文档...")
            self.vector_store.add_documents(documents)
            
            # 保存向量存储
            self.vector_store.save_local(str(vector_path))
            
            # 保存数据哈希
            hash_file = vector_path / "data_hash.txt"
            with open(hash_file, 'w') as f:
                f.write(self._get_data_hash())
            
            print("✅ 向量存储构建并保存完成")
        
        return self.vector_store
    
    def create_retriever(self) -> RAGRetriever:
        """创建检索器"""
        if not self.vector_store or len(self.vector_store.documents) == 0:
            raise ValueError("向量存储未初始化，请先调用build_vector_store()")
        
        self.retriever = RAGRetriever(
            vector_store=self.vector_store,
            reranker=self.reranker,
            embed_top_k=self.config.embed_top_k,
            final_top_k=self.config.final_top_k
        )
        
        print("🎯 RAG检索器创建完成")
        return self.retriever
    
    def search(self, query: str) -> List[Dict]:
        """执行检索"""
        if not self.retriever:
            raise ValueError("检索器未初始化，请先调用create_retriever()")
        
        docs = self.retriever.get_relevant_documents(query)
        
        results = []
        for doc in docs:
            results.append({
                'rank': doc.metadata.get('final_rank', 0),
                'content': doc.page_content,
                'source': doc.metadata.get('source', 'unknown'),
                'filename': doc.metadata.get('filename', 'unknown'),
                'embed_score': doc.metadata.get('embed_score', 0.0),
                'rerank_score': doc.metadata.get('rerank_score', 0.0),
                'relevance': self._get_relevance_label(doc.metadata.get('rerank_score', 0.0))
            })
        
        return results
    
    def _get_relevance_label(self, score: float) -> str:
        """获取相关性标签"""
        if score > 0.7:
            return "🔥 高度相关"
        elif score > 0.4:
            return "📄 中等相关"
        else:
            return "❓ 低相关"


def demo():
    """演示函数"""
    print("🚀 轻量级RAG系统演示（无LangChain依赖）")
    print("="*60)
    
    # 配置参数
    config = RAGConfig(
        embed_model_path="/mnt/e/qwenrag/embed",     # 修改为你的嵌入模型路径
        rerank_model_path="/mnt/e/qwenrag/rerank",   # 修改为你的重排序模型路径
        txt_data_path="/mnt/e/qwenrag/Rag_System/data/processed/LiHua-World",  # 支持递归查找子目录中的TXT文件
        vector_store_path="/mnt/e/qwenrag/Rag_System/data/vectors",
        embed_top_k=6,
        final_top_k=3
    )
    
    try:
        # 初始化系统
        rag_system = LightweightRAGSystem(config)
        
        # 构建向量存储（只有数据变化时才重新构建）
        rag_system.build_vector_store(force_rebuild=False)
        
        # 创建检索器
        rag_system.create_retriever()
        
        # 测试查询
        test_queries = [
            "gym workout plan",
            "RAG system implementation",
            "epic photos"
        ]
        
        for query in test_queries:
            print(f"\n{'='*60}")
            print(f"🔍 查询: {query}")
            print(f"{'='*60}")
            
            try:
                results = rag_system.search(query)
                
                if results:
                    print(f"\n🏆 检索结果 (共{len(results)}条):")
                    for result in results:
                        print(f"\n{result['rank']}. {result['relevance']}")
                        print(f"   文件: {result['filename']}")
                        print(f"   向量相似度: {result['embed_score']:.3f}")
                        print(f"   重排序评分: {result['rerank_score']:.3f}")
                        print(f"   内容预览: {result['content'][:200]}...")
                else:
                    print("❌ 没有找到相关结果")
                    
            except Exception as e:
                print(f"❌ 查询失败: {str(e)}")
    
    except Exception as e:
        print(f"❌ 系统初始化失败: {str(e)}")


if __name__ == "__main__":
    demo()

