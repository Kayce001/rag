#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整RAG检索重排序系统
结合Embedding检索 + Reranker重排序
适合初学者学习的简化版本
"""

import math
import warnings
from typing import List, Tuple, Dict
import torch

# 忽略警告信息
warnings.filterwarnings("ignore")

try:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs.data import TokensPrompt
    print("✓ 成功导入所有必需库")
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请安装: pip install vllm transformers torch")
    exit(1)


class SimpleRAGSystem:
    """简化版RAG检索重排序系统"""
    
    def __init__(self, embed_model_path: str, rerank_model_path: str):
        """
        初始化RAG系统
        
        Args:
            embed_model_path: 嵌入模型路径
            rerank_model_path: 重排序模型路径
        """
        print("🚀 初始化RAG系统...")
        
        # 1. 加载嵌入模型
        print(f"📦 正在加载嵌入模型: {embed_model_path}")
        self.embed_model = LLM(model=embed_model_path, task="embed")
        
        # 2. 加载重排序模型
        print(f"📦 正在加载重排序模型: {rerank_model_path}")
        self.rerank_tokenizer = AutoTokenizer.from_pretrained(
            rerank_model_path, 
            trust_remote_code=True
        )
        self.rerank_model = LLM(
            model=rerank_model_path,
            max_model_len=8192,
            trust_remote_code=True,
            gpu_memory_utilization=0.8
        )
        
        # 3. 设置重排序参数
        self._setup_reranker()
        
        print("✅ RAG系统初始化完成")
    
    def _setup_reranker(self):
        """设置重排序器参数"""
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.suffix_tokens = self.rerank_tokenizer.encode(self.suffix, add_special_tokens=False)
        
        # 获取yes/no对应的token ID
        self.yes_token = self.rerank_tokenizer("yes", add_special_tokens=False).input_ids[0]
        self.no_token = self.rerank_tokenizer("no", add_special_tokens=False).input_ids[0]
        
        # 设置采样参数
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            logprobs=20,
            allowed_token_ids=[self.yes_token, self.no_token]
        )
    
    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        """创建详细指令"""
        return f'Instruct: {task_description}\nQuery:{query}'
    
    def embed_texts(self, texts: List[str]) -> torch.Tensor:
        """
        对文本进行嵌入编码
        
        Args:
            texts: 文本列表
            
        Returns:
            嵌入向量张量
        """
        outputs = self.embed_model.embed(texts)
        embeddings = torch.tensor([o.outputs.embedding for o in outputs])
        return embeddings
    
    def retrieve_by_embedding(self, query: str, documents: List[str], 
                            task: str = "Given a web search query, retrieve relevant passages that answer the query",
                            top_k: int = 10) -> List[Tuple[str, float]]:
        """
        使用嵌入向量进行初步检索
        
        Args:
            query: 查询文本
            documents: 文档列表
            task: 任务描述
            top_k: 返回top-k个结果
            
        Returns:
            按相似度排序的(文档, 相似度)列表
        """
        print(f"🔍 使用嵌入向量检索 top-{top_k} 文档...")
        
        # 准备输入文本
        detailed_query = self.get_detailed_instruct(task, query)
        input_texts = [detailed_query] + documents
        
        # 获取嵌入向量
        embeddings = self.embed_texts(input_texts)
        
        # 计算相似度
        query_embedding = embeddings[0:1]  # 查询向量
        doc_embeddings = embeddings[1:]    # 文档向量
        
        # 计算余弦相似度
        similarity_scores = (query_embedding @ doc_embeddings.T).squeeze()
        
        # 创建结果列表
        results = []
        for i, (doc, score) in enumerate(zip(documents, similarity_scores)):
            results.append((doc, float(score)))
        
        # 按相似度排序并返回top-k
        results.sort(key=lambda x: x[1], reverse=True)
        top_results = results[:top_k]
        
        print(f"✅ 嵌入检索完成，返回 {len(top_results)} 个结果")
        return top_results
    
    def create_rerank_prompt(self, instruction: str, query: str, document: str) -> List[dict]:
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
    
    def rerank_score(self, query: str, document: str, 
                    instruction: str = "Given a web search query, retrieve relevant passages that answer the query") -> float:
        """
        计算重排序评分
        
        Args:
            query: 查询文本
            document: 文档文本
            instruction: 任务指令
            
        Returns:
            重排序评分 (0-1之间)
        """
        # 创建提示词
        messages = [self.create_rerank_prompt(instruction, query, document)]
        
        # 转换为tokens
        tokenized = self.rerank_tokenizer.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=False
        )[0]
        
        # 添加后缀并创建输入
        full_tokens = tokenized + self.suffix_tokens
        token_prompt = TokensPrompt(prompt_token_ids=full_tokens)
        
        # 模型推理
        outputs = self.rerank_model.generate([token_prompt], self.sampling_params)
        
        # 获取logprobs
        logprobs = outputs[0].outputs[0].logprobs[-1]
        
        # 计算yes和no的概率
        yes_logprob = logprobs.get(self.yes_token, type('obj', (object,), {'logprob': -10})()).logprob
        no_logprob = logprobs.get(self.no_token, type('obj', (object,), {'logprob': -10})()).logprob
        
        yes_prob = math.exp(yes_logprob)
        no_prob = math.exp(no_logprob)
        
        # 归一化得到最终评分
        total_prob = yes_prob + no_prob
        score = yes_prob / total_prob if total_prob > 0 else 0.0
        
        return score
    
    def rerank_documents(self, query: str, documents: List[Tuple[str, float]], 
                        instruction: str = "Given a web search query, retrieve relevant passages that answer the query") -> List[Tuple[str, float, float]]:
        """
        对文档进行重排序
        
        Args:
            query: 查询文本
            documents: (文档, 嵌入相似度)列表
            instruction: 任务指令
            
        Returns:
            (文档, 嵌入相似度, 重排序评分)列表，按重排序评分排序
        """
        print(f"🎯 正在对 {len(documents)} 个文档进行重排序...")
        
        results = []
        for i, (doc, embed_score) in enumerate(documents):
            rerank_score = self.rerank_score(query, doc, instruction)
            results.append((doc, embed_score, rerank_score))
            print(f"  文档 {i+1}/{len(documents)}: 嵌入={embed_score:.3f}, 重排序={rerank_score:.3f}")
        
        # 按重排序评分排序
        results.sort(key=lambda x: x[2], reverse=True)
        
        print("✅ 重排序完成")
        return results
    
    def search(self, query: str, documents: List[str], 
              task: str = "Given a web search query, retrieve relevant passages that answer the query",
              embed_top_k: int = 10, final_top_k: int = 5) -> List[Dict]:
        """
        完整的RAG检索流程
        
        Args:
            query: 查询文本
            documents: 文档库
            task: 任务描述
            embed_top_k: 嵌入检索返回的文档数量
            final_top_k: 最终返回的文档数量
            
        Returns:
            最终排序结果列表
        """
        print(f"\n🔍 开始RAG检索: '{query}'")
        print(f"📚 文档库大小: {len(documents)}")
        print("="*60)
        
        # 第一步：嵌入向量检索
        embed_results = self.retrieve_by_embedding(query, documents, task, embed_top_k)
        
        # 第二步：重排序
        rerank_results = self.rerank_documents(query, embed_results, task)
        
        # 第三步：返回最终结果
        final_results = []
        for i, (doc, embed_score, rerank_score) in enumerate(rerank_results[:final_top_k]):
            final_results.append({
                'rank': i + 1,
                'document': doc,
                'embed_score': embed_score,
                'rerank_score': rerank_score,
                'relevance': '🔥 高度相关' if rerank_score > 0.7 else '📄 中等相关' if rerank_score > 0.4 else '❓ 低相关'
            })
        
        return final_results


def demo():
    """演示函数"""
    print("🚀 完整RAG检索重排序系统演示")
    print("="*60)
    
    # 请修改为你的模型路径
    EMBED_MODEL_PATH = "/mnt/e/qwenrag/embed"      # 嵌入模型路径
    RERANK_MODEL_PATH = "/mnt/e/qwenrag/rerank"    # 重排序模型路径
    
    # 初始化RAG系统
    rag_system = SimpleRAGSystem(EMBED_MODEL_PATH, RERANK_MODEL_PATH)
    
    # 测试数据
    query = "What is the capital of China?"
    documents = [
        "The capital of China is Beijing. It is located in northern China and serves as the political center.",
        "China is a large country in East Asia with over 1.4 billion people and rich cultural heritage.",
        "Beijing is the capital and political center of China, known for its historical landmarks.",
        "Shanghai is the most populous city in China and a major financial hub.",
        "The Great Wall of China is a famous tourist attraction that stretches across northern China.",
        "Guangzhou is a major city in southern China known for its trade and commerce.",
        "Xi'an is an ancient city in China, famous for the Terracotta Warriors.",
        "Shenzhen is a modern city in China, known for its technology industry.",
        "Chengdu is the capital of Sichuan province, famous for pandas and spicy food.",
        "Hangzhou is known for its beautiful West Lake and as the headquarters of Alibaba."
    ]
    
    # 执行完整RAG检索
    results = rag_system.search(
        query=query,
        documents=documents,
        embed_top_k=6,  # 嵌入检索返回前6个
        final_top_k=3   # 最终返回前3个
    )
    
    # 显示结果
    print(f"\n🏆 最终检索结果:")
    print("="*60)
    for result in results:
        print(f"\n{result['rank']}. {result['relevance']}")
        print(f"   嵌入相似度: {result['embed_score']:.3f}")
        print(f"   重排序评分: {result['rerank_score']:.3f}")
        print(f"   文档内容: {result['document']}")


if __name__ == "__main__":
    demo()