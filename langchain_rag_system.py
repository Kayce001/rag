#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于LangChain的RAG检索重排序系统
使用LangChain框架重构原有系统，提供更好的扩展性和维护性
"""

import os
import json
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import warnings
import torch
import numpy as np
from pathlib import Path
import hashlib

# LangChain核心组件
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.pydantic_v1 import Field

# LangChain文档加载器
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# LangChain向量存储
from langchain_community.vectorstores import FAISS

# VLLM和Transformers
try:
    from vllm import LLM, SamplingParams
    from vllm.inputs.data import TokensPrompt
    from transformers import AutoTokenizer
    print("✅ 成功导入所有必需库")
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请安装: pip install langchain langchain-community faiss-cpu vllm transformers torch")
    exit(1)

warnings.filterwarnings("ignore")


@dataclass
class RAGConfig:
    """RAG系统配置类"""
    embed_model_path: str
    rerank_model_path: str
    txt_data_path: str
    vector_store_path: str = "./vector_store"
    embed_top_k: int = 10
    final_top_k: int = 5
    chunk_size: int = 1000
    chunk_overlap: int = 200
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class QwenEmbeddings(Embeddings):
    """
    继承LangChain的Embeddings基类的Qwen嵌入模型
    提供标准的LangChain嵌入接口
    """
    
    def __init__(self, model_path: str):
        super().__init__()
        self.model_path = model_path
        print(f"📦 加载嵌入模型: {model_path}")
        self.model = LLM(model=model_path, task="embed")
        print("✅ 嵌入模型加载完成")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入文档列表 - LangChain标准接口"""
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
        """嵌入查询文本 - LangChain标准接口"""
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


class RerankRetriever(BaseRetriever):
    """
    继承LangChain BaseRetriever的重排序检索器
    实现标准的LangChain检索器接口
    """
    
    vector_store: FAISS = Field(description="向量存储")
    reranker: QwenReranker = Field(description="重排序器")
    embed_top_k: int = Field(default=10, description="向量检索返回数量")
    final_top_k: int = Field(default=5, description="最终返回数量")
    
    class Config:
        arbitrary_types_allowed = True
    
    def _get_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """获取相关文档 - LangChain标准接口"""
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
            # 更新文档元数据
            doc.metadata.update({
                'embed_score': embed_score,
                'rerank_score': rerank_score,
                'final_rank': i + 1
            })
            final_docs.append(doc)
        
        print(f"✅ 重排序完成，返回top-{len(final_docs)}结果")
        return final_docs


class LangChainRAGSystem:
    """基于LangChain的RAG系统"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.embeddings = None
        self.reranker = None
        self.vector_store = None
        self.retriever = None
        self.text_splitter = None
        
        self._initialize_components()
    
    def _initialize_components(self):
        """初始化组件"""
        print("🚀 初始化LangChain RAG系统...")
        
        # 初始化嵌入模型
        self.embeddings = QwenEmbeddings(self.config.embed_model_path)
        
        # 初始化重排序器
        self.reranker = QwenReranker(self.config.rerank_model_path)
        
        # 初始化文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        
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
    
    def load_documents(self) -> List[Document]:
        """使用LangChain加载文档"""
        data_path = Path(self.config.txt_data_path)
        
        if data_path.is_file() and data_path.suffix == '.txt':
            # 单个文件
            loader = TextLoader(str(data_path), encoding='utf-8')
            docs = loader.load()
        elif data_path.is_dir():
            # 目录加载器，支持递归查找
            loader = DirectoryLoader(
                str(data_path),
                glob="**/*.txt",  # 递归查找所有txt文件
                loader_cls=TextLoader,
                loader_kwargs={'encoding': 'utf-8'},
                show_progress=True
            )
            docs = loader.load()
        else:
            raise ValueError(f"无效的数据路径: {data_path}")
        
        print(f"📁 使用LangChain加载了 {len(docs)} 个文档")
        
        # 文档分割（如果需要）
        if self.config.chunk_size > 0:
            print(f"📄 开始文档分割，块大小: {self.config.chunk_size}, 重叠: {self.config.chunk_overlap}")
            docs = self.text_splitter.split_documents(docs)
            print(f"📄 分割完成，总共 {len(docs)} 个文档块")
        
        return docs
    
    def build_vector_store(self, force_rebuild: bool = False) -> FAISS:
        """构建或加载向量存储"""
        vector_path = Path(self.config.vector_store_path)
        
        # 检查是否需要重建
        need_rebuild = force_rebuild or not self._check_vector_store_validity()
        
        if not need_rebuild and vector_path.exists():
            print("📂 检测到有效的向量存储缓存...")
            try:
                self.vector_store = FAISS.load_local(
                    str(vector_path), 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print("✅ 向量存储加载成功，跳过重新构建")
                return self.vector_store
            except Exception as e:
                print(f"⚠️  向量存储加载失败: {e}，将重新构建")
                need_rebuild = True
        
        if need_rebuild:
            print("🏗️  构建新的向量存储...")
            
            # 使用LangChain加载文档
            documents = self.load_documents()
            
            if not documents:
                raise ValueError("没有找到有效的文档")
            
            # 使用LangChain FAISS构建向量存储
            print("🔨 开始向量化文档...")
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            
            # 保存向量存储
            vector_path.mkdir(parents=True, exist_ok=True)
            self.vector_store.save_local(str(vector_path))
            
            # 保存数据哈希
            hash_file = vector_path / "data_hash.txt"
            with open(hash_file, 'w') as f:
                f.write(self._get_data_hash())
            
            print("✅ 向量存储构建并保存完成")
        
        return self.vector_store
    
    def create_retriever(self) -> RerankRetriever:
        """创建检索器"""
        if not self.vector_store:
            raise ValueError("向量存储未初始化，请先调用build_vector_store()")
        
        self.retriever = RerankRetriever(
            vector_store=self.vector_store,
            reranker=self.reranker,
            embed_top_k=self.config.embed_top_k,
            final_top_k=self.config.final_top_k
        )
        
        print("🎯 LangChain RAG检索器创建完成")
        return self.retriever
    
    def search(self, query: str) -> List[Dict]:
        """执行检索"""
        if not self.retriever:
            raise ValueError("检索器未初始化，请先调用create_retriever()")
        
        # 使用LangChain标准接口
        docs = self.retriever.get_relevant_documents(query)
        
        results = []
        for doc in docs:
            results.append({
                'rank': doc.metadata.get('final_rank', 0),
                'content': doc.page_content,
                'source': doc.metadata.get('source', 'unknown'),
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
    
    def get_retriever_as_langchain_tool(self):
        """获取可用于LangChain Agent的检索工具"""
        from langchain.tools.retriever import create_retriever_tool
        
        retriever_tool = create_retriever_tool(
            self.retriever,
            "rag_retriever",
            "这是一个RAG检索工具，可以根据查询检索相关文档。用于回答需要查找知识库信息的问题。"
        )
        
        return retriever_tool


def demo():
    """演示函数"""
    print("🚀 LangChain RAG系统演示")
    print("="*60)
    
    # 配置参数
    config = RAGConfig(
        embed_model_path="/mnt/e/qwenrag/embed",     # 修改为你的嵌入模型路径
        rerank_model_path="/mnt/e/qwenrag/rerank",   # 修改为你的重排序模型路径
        txt_data_path="/mnt/e/qwenrag/Rag_System/data/processed/LiHua-World",  # 支持递归查找子目录中的TXT文件
        vector_store_path="/mnt/e/qwenrag/Rag_System/data/vectors_langchain",
        embed_top_k=6,
        final_top_k=3,
        chunk_size=1000,  # 启用文档分割
        chunk_overlap=200
    )
    
    try:
        # 初始化系统
        rag_system = LangChainRAGSystem(config)
        
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
                        print(f"   来源: {Path(result['source']).name}")
                        print(f"   向量相似度: {result['embed_score']:.3f}")
                        print(f"   重排序评分: {result['rerank_score']:.3f}")
                        print(f"   内容预览: {result['content'][:200]}...")
                else:
                    print("❌ 没有找到相关结果")
                    
            except Exception as e:
                print(f"❌ 查询失败: {str(e)}")
        
        # 演示获取LangChain工具
        print(f"\n{'='*60}")
        print("🛠️  获取LangChain检索工具")
        print(f"{'='*60}")
        retriever_tool = rag_system.get_retriever_as_langchain_tool()
        print(f"✅ 工具创建完成: {retriever_tool.name}")
        print(f"   描述: {retriever_tool.description}")
    
    except Exception as e:
        print(f"❌ 系统初始化失败: {str(e)}")


if __name__ == "__main__":
    demo()