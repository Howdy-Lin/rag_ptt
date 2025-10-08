"""
對話式 RAG 系統：使用 ComplementaryRetriever + TAIDE LLM (4-bit 量化版本)
支援多輪對話與上下文記憶
顯示檢索到的參考文章
"""

import os
from typing import List, Dict, Optional, Tuple
from complementary_retriever import ComplementaryRetriever
from transformers import AutoTokenizer, AutoModelForCausalLM
from complementary_retriever import ComplementaryRetriever, BM25Index
#from transformers import BitsAndBytesConfig
import torch
import sys
import os
from dotenv import load_dotenv
load_dotenv()
sys.modules['__main__'].BM25Index = BM25Index

class ConversationalRAG:
    """對話式 RAG 系統"""

    def __init__(
        self,
        faiss_index_path: str,
        metadata_path: str,
        bm25_index_path: str,
        hf_token: str,
        model_name: str = "taide/TAIDE-LX-7B-Chat"
        # use_4bit: bool = True  # ← 刪除這個參數
    ):
        # 初始化檢索器
        print("=" * 60)
        print("初始化檢索器...")
        print("=" * 60)
        self.retriever = ComplementaryRetriever(
            faiss_index_path=faiss_index_path,
            metadata_path=metadata_path,
            bm25_index_path=bm25_index_path
        )
        
        # 初始化 TAIDE LLM
        print("\n" + "=" * 60)
        print("載入 TAIDE 模型（無量化）...")
        print("=" * 60)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用裝置: {self.device}")
        
        # 載入 Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=hf_token
        )
        
        # 載入模型（無量化）
        print("模型配置:")
        print("  - 精度: float16" if self.device == "cuda" else "  - 精度: float32")
        print("  - 設備分配: auto")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=hf_token,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        print("\n✓ TAIDE 模型載入完成")
        
        # 顯示 GPU 記憶體使用情況
        if self.device == "cuda":
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"✓ GPU 記憶體使用: {allocated:.2f} GB (已配置) / {reserved:.2f} GB (已保留)")
        print()
        
        # 對話歷史
        self.conversation_history = []
        self.max_history_turns = 5

    def _detect_query_type(self, query: str) -> str:
        """判斷查詢類型"""
        temporal_keywords = ['第一個', '誰先', '最早', '起源', '開始', '何時']
        if any(kw in query for kw in temporal_keywords):
            return 'temporal'
        return 'general'

    def _format_context(self, results: List[Dict], query_type: str) -> str:
        """格式化檢索結果為 Context"""
        if not results:
            return "（未找到相關資料）"

        parts = []

        if query_type == 'temporal':
            results_sorted = sorted(results, key=lambda x: x.get('push_time', ''))
            parts.append("以下是按時間排序的相關討論：\n")

            for i, r in enumerate(results_sorted[:10], 1):
                parts.append(
                    f"{i}. [{r['push_time']}] {r['user_id']}\n"
                    f"   {r['combined_text'][:150]}\n"
                )
        else:
            parts.append("以下是相關的討論內容：\n")

            for i, r in enumerate(results[:15], 1):
                source_label = {
                    'bm25_high': '精確',
                    'bm25_med': '部分',
                    'faiss_high': '相關',
                    'faiss_low': '參考'
                }.get(r.get('source', ''), '參考')

                parts.append(f"{i}. [{source_label}] {r['combined_text'][:120]}\n")

        return "\n".join(parts)

    def _format_retrieved_documents(self, results: List[Dict], query_type: str) -> str:
        """格式化檢索文檔供使用者查看（完整版本）"""
        if not results:
            return "未檢索到相關文章"

        parts = ["\n" + "=" * 60]
        parts.append("📚 檢索到的參考文章（LLM 閱讀的內容）")
        parts.append("=" * 60 + "\n")

        if query_type == 'temporal':
            results_sorted = sorted(results, key=lambda x: x.get('push_time', ''))
            results_to_show = results_sorted[:10]
        else:
            results_to_show = results[:15]

        for i, r in enumerate(results_to_show, 1):
            # 來源標籤
            source_label = {
                'bm25_high': '🎯 精確匹配',
                'bm25_med': '📌 部分匹配',
                'faiss_high': '🔗 高度相關',
                'faiss_low': '📎 參考相關'
            }.get(r.get('source', ''), '📎 參考')

            # 文章資訊
            parts.append(f"【文章 {i}】 {source_label}")
            parts.append(f"作者: {r.get('user_id', 'Unknown')}")
            parts.append(f"時間: {r.get('push_time', 'Unknown')}")

            # 相似度分數（如果有）
            if 'score' in r:
                parts.append(f"相似度: {r['score']:.4f}")

            parts.append(f"\n內容:")
            parts.append(f"{r['combined_text']}")
            parts.append("\n" + "-" * 60 + "\n")

        return "\n".join(parts)

    def _build_prompt(
            self,
            query: str,
            context: str,
            query_type: str,
            use_history: bool = True  # ← 這個參數可以保留但不會使用
    ) -> str:
        """建構對話式 Prompt"""

        system_prompt = """你是 PTT Baseball 板的專家助手，專門回答棒球相關問題。
    你的回答風格：
    - 親切自然，像朋友聊天
    - 根據參考資料回答，不編造資訊
    - 如果不確定，誠實說明
    - 可以適度加入自己的見解
    - 回答簡潔清楚，不要過於冗長"""

        # ← 刪除對話歷史部分
        # history_section = ""
        # if use_history and self.conversation_history:
        #     history_section = f"\n{self._format_conversation_history()}\n"

        context_section = f"""
    【參考資料】
    {context}
    """

        if query_type == 'temporal':
            special_instruction = "\n請明確指出第一個提到的人和時間。"
        else:
            special_instruction = ""

        # ← 移除 history_section
        prompt = f"""[INST] <<SYS>>
    {system_prompt}
    <</SYS>>

    {context_section}

    【使用者問題】
    {query}{special_instruction}
    [/INST]"""

        return prompt

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """使用 TAIDE 生成回答"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "[/INST]" in response:
            response = response.split("[/INST]")[-1].strip()

        return response

    def chat(
        self,
        question: str,
        top_k: int = 20,
        bm25_high_threshold: float = 16.0,
        bm25_med_threshold: float = 12.0,
        faiss_high_threshold: float = 0.90,
        faiss_low_threshold: float = 0.80,
        use_retrieval: bool = True,
        use_history: bool = True,
        show_retrieved_docs: bool = True  # 新增：是否顯示檢索文檔
    ) -> Tuple[str, Optional[str], Optional[List[Dict]]]:
        """
        對話式查詢（主要介面）

        Args:
            question: 使用者問題
            use_retrieval: 是否使用 RAG 檢索
            use_history: 是否使用對話歷史
            show_retrieved_docs: 是否返回檢索到的文檔

        Returns:
            Tuple[answer, retrieved_docs_text, results]:
                - answer: LLM 生成的回答
                - retrieved_docs_text: 格式化的檢索文檔文字（供顯示）
                - results: 原始檢索結果列表
        """
        # 判斷查詢類型
        query_type = self._detect_query_type(question)

        # 檢索相關文檔
        context = ""
        results = None
        retrieved_docs_text = None

        if use_retrieval:
            results = self.retriever.search(
                query=question,
                top_k=top_k,
                bm25_high_threshold=bm25_high_threshold,
                bm25_med_threshold=bm25_med_threshold,
                faiss_high_threshold=faiss_high_threshold,
                faiss_low_threshold=faiss_low_threshold
            )

            if results:
                context = self._format_context(results, query_type)

                # 格式化檢索文檔供使用者查看
                if show_retrieved_docs:
                    retrieved_docs_text = self._format_retrieved_documents(results, query_type)

        # 建構 Prompt
        prompt = self._build_prompt(question, context, query_type, use_history)

        # 生成回答
        answer = self.generate(prompt)

        return answer, retrieved_docs_text, results

    def clear_history(self):
        """清空對話歷史"""
        print("對話記憶功能已停用")


def interactive_chat():
    """互動式對話"""

    # 初始化系統
    rag = ConversationalRAG(
        faiss_index_path="faiss_index_part1/ptt_baseball_part1.index",
        metadata_path="faiss_index_part1/ptt_baseball_metadata_part1.pkl",
        bm25_index_path="faiss_index_part1/bm25_index.pkl",
        hf_token=os.environ.get("HUGGINGFACE_TOKEN")
    )

    print("\n" + "=" * 60)
    print("PTT Baseball 對話助手（輸入 'q' 退出）")
    print("=" * 60)
    print("\n指令：")
    print("  q      - 退出")
    print("  clear  - 清空對話歷史")
    print("  help   - 顯示幫助")
    print("  docs   - 切換是否顯示檢索文檔\n")

    show_docs = True  # 預設顯示檢索文檔

    while True:
        question = input("\n你: ").strip()

        if question.lower() == 'q':
            print("\n再見！")
            break

        if question.lower() == 'clear':
            rag.clear_history()
            continue

        if question.lower() == 'docs':
            show_docs = not show_docs
            status = "開啟" if show_docs else "關閉"
            print(f"\n已{status}檢索文檔顯示")
            continue

        if question.lower() == 'help':
            print("\n你可以問我：")
            print("  - 球員相關問題（王建民表現如何？）")
            print("  - 時序問題（Wang Soto 誰先提出？）")
            print("  - 一般討論（大谷翔平評價如何？）")
            print("  - 也可以純聊天（不一定要棒球相關）")
            continue

        if not question:
            continue

        try:
            # 生成回答（取得三個返回值）
            answer, retrieved_docs, results = rag.chat(
                question,
                show_retrieved_docs=show_docs
            )

            # 先顯示 LLM 回答
            print(f"\n🤖 助手回答:")
            print("=" * 60)
            print(answer)
            print("=" * 60)

            # 顯示檢索到的文檔（如果啟用）
            if show_docs and retrieved_docs:
                print(retrieved_docs)

            # 顯示統計資訊
            if results:
                print(f"\n📊 檢索統計: 共 {len(results)} 篇文章")

        except Exception as e:
            print(f"\n錯誤：{e}")
            import traceback
            traceback.print_exc()


def demo_conversation():
    """示範對話"""

    rag = ConversationalRAG(
        faiss_index_path="faiss_index_part1/ptt_baseball_part1.index",
        metadata_path="faiss_index_part1/ptt_baseball_metadata_part1.pkl",
        bm25_index_path="faiss_index_part1/bm25_index.pkl",
        hf_token="token"
    )

    conversation = [
        "王建民表現如何？",
        "他跟郭泓志比呢？",
        "那大谷翔平呢？",
    ]

    print("\n" + "=" * 60)
    print("示範對話")
    print("=" * 60)

    for question in conversation:
        print(f"\n使用者: {question}")

        # 取得回答和檢索文檔
        answer, retrieved_docs, results = rag.chat(question, show_retrieved_docs=True)

        print(f"\n🤖 助手: {answer}")
        # 顯示檢索的文章
        if retrieved_docs:
            print(retrieved_docs)

        input("\n按 Enter 繼續...")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo_conversation()
    else:
        interactive_chat()