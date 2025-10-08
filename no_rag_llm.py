"""
純 LLM 對話系統：使用 TAIDE LLM（無 RAG）
支援多輪對話與上下文記憶
作為 RAG 系統的對照組
"""

from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import os
from dotenv import load_dotenv
load_dotenv()
class ConversationalLLM:
    """純 LLM 對話系統（無 RAG 檢索）"""

    def __init__(
            self,
            hf_token: str,
            model_name: str = "taide/TAIDE-LX-7B-Chat",
            use_4bit: bool = True
    ):
        print("=" * 60)
        print("載入 TAIDE 模型（無 RAG 版本）...")
        if use_4bit:
            print("使用 4-bit 量化模式")
        print("=" * 60)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用裝置: {self.device}")

        # 載入 Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=hf_token
        )

        # 配置 4-bit 量化
        if use_4bit and self.device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

            print("量化配置:")
            print("  - 4-bit 量化: 啟用")
            print("  - 量化類型: NF4")
            print("  - 雙重量化: 啟用")
            print("  - 計算精度: float16")

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=hf_token,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=hf_token,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto"
            )

        print("✓ TAIDE 模型載入完成")

        # 顯示記憶體使用情況
        if self.device == "cuda":
            print(f"✓ GPU 記憶體使用: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
        print()

        # 對話歷史
        self.conversation_history = []
        self.max_history_turns = 5

    def _format_conversation_history(self) -> str:
        """格式化對話歷史"""
        if not self.conversation_history:
            return ""

        history_text = "先前的對話：\n"
        for turn in self.conversation_history[-self.max_history_turns:]:
            history_text += f"使用者：{turn['user']}\n"
            history_text += f"助手：{turn['assistant']}\n\n"

        return history_text

    def _build_prompt(
            self,
            query: str,
            use_history: bool = True
    ) -> str:
        """建構對話式 Prompt（無 RAG 檢索）"""

        # 系統提示（針對棒球領域）
        system_prompt = """你是一個友善的棒球專家助手。
你的回答風格：
- 親切自然，像朋友聊天
- 根據你的棒球知識回答問題
- 如果不確定或不知道，誠實說明
- 可以分享你對棒球的見解和看法
- 回答簡潔清楚，不要過於冗長"""

        # 對話歷史
        history_section = ""
        if use_history and self.conversation_history:
            history_section = f"\n{self._format_conversation_history()}\n"

        # 組合 Prompt（無參考資料部分）
        prompt = f"""[INST] <<SYS>>
{system_prompt}
<</SYS>>

{history_section}
【使用者問題】
{query}
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

        # 移除 prompt 部分
        if "[/INST]" in response:
            response = response.split("[/INST]")[-1].strip()

        return response

    def chat(
            self,
            question: str,
            use_history: bool = True
    ) -> str:
        """
        對話式查詢（無 RAG 版本）

        Args:
            question: 使用者問題
            use_history: 是否使用對話歷史

        Returns:
            LLM 生成的回答
        """
        # 建構 Prompt（無檢索）
        prompt = self._build_prompt(question, use_history)

        # 生成回答
        answer = self.generate(prompt)

        # 更新對話歷史
        self.conversation_history.append({
            'user': question,
            'assistant': answer
        })

        # 只保留最近 N 輪
        if len(self.conversation_history) > self.max_history_turns:
            self.conversation_history = self.conversation_history[-self.max_history_turns:]

        return answer

    def clear_history(self):
        """清空對話歷史"""
        self.conversation_history = []
        print("對話歷史已清空")


def interactive_chat():
    """互動式對話（無 RAG）"""

    # 初始化系統
    llm = ConversationalLLM(
        hf_token=os.environ.get("HUGGINGFACE_TOKEN"),
        use_4bit=False
    )

    print("\n" + "=" * 60)
    print("PTT Baseball 對話助手 - 對照組（無 RAG）")
    print("僅使用 LLM 本身的知識回答問題")
    print("=" * 60)
    print("\n指令：")
    print("  q      - 退出")
    print("  clear  - 清空對話歷史")
    print("  help   - 顯示幫助\n")

    while True:
        question = input("\n你: ").strip()

        if question.lower() == 'q':
            print("\n再見！")
            break

        if question.lower() == 'clear':
            llm.clear_history()
            continue

        if question.lower() == 'help':
            print("\n你可以問我：")
            print("  - 球員相關問題（王建民表現如何？）")
            print("  - 棒球規則和知識")
            print("  - 一般棒球討論")
            print("  - 也可以純聊天")
            print("\n注意：此版本不會檢索 PTT 資料，僅依靠模型本身知識")
            continue

        if not question:
            continue

        try:
            # 生成回答
            answer = llm.chat(question)

            # 顯示回答
            print(f"\n助手: {answer}")

        except Exception as e:
            print(f"\n錯誤：{e}")
            import traceback
            traceback.print_exc()


def demo_conversation():
    """示範對話（無 RAG）"""

    llm = ConversationalLLM(
        hf_token="token",
        use_4bit=False
    )

    conversation = [
        "王建民表現如何？",
        "他跟郭泓志比呢？",
        "那大谷翔平呢？",
    ]

    print("\n" + "=" * 60)
    print("示範對話（無 RAG 版本）")
    print("=" * 60)

    for question in conversation:
        print(f"\n使用者: {question}")
        answer = llm.chat(question)
        print(f"助手: {answer}")
        input("\n按 Enter 繼續...")


def compare_with_rag():
    """
    比較實驗建議：同時運行 RAG 和 無RAG 版本
    記錄以下指標：
    1. 回答準確度
    2. 回答相關性
    3. 事實性錯誤數量
    4. 回答時間
    5. 使用者滿意度
    """
    print("""
    ═══════════════════════════════════════════════════════════
    對照實驗建議
    ═══════════════════════════════════════════════════════════

    測試問題分類：

    1. 事實性問題（可驗證）
       - "王建民在 2023 年的表現如何？"
       - "PTT 上第一個提到 Soto 的是誰？"

    2. 意見性問題（主觀評價）
       - "大谷翔平是史上最強球員嗎？"
       - "鄉民們怎麼看待這次交易？"

    3. 時序性問題（需要精確時間資訊）
       - "誰先提出 Wang Soto？"
       - "這個梗是什麼時候開始的？"

    4. 一般知識問題（不需特定資料）
       - "三振是什麼意思？"
       - "棒球規則怎麼算？"

    評估指標：
    ✓ 準確性：答案是否正確
    ✓ 相關性：是否回答了問題
    ✓ 完整性：資訊是否充足
    ✓ 可信度：是否有來源支持
    ✓ 流暢度：語言表達品質

    預期差異：
    - RAG 版本：在事實性、時序性問題上更準確，有具體來源
    - 無 RAG 版本：在一般知識問題上足夠，但缺乏特定資訊

    ═══════════════════════════════════════════════════════════
    """)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "demo":
            demo_conversation()
        elif sys.argv[1] == "compare":
            compare_with_rag()
    else:
        interactive_chat()