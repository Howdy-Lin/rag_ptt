"""
互補法混合檢索器：BM25 精確 + FAISS 語義
使用原始分數 + 匹配質量檢查
"""

import os
import pickle
import faiss
import numpy as np
import jieba
from rank_bm25 import BM25Okapi
from typing import List, Dict, Tuple
from langchain_huggingface import HuggingFaceEmbeddings
import torch

EMBEDDING_MODEL_NAME = "BAAI/bge-base-zh-v1.5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class BM25Index:
    """BM25 索引管理器"""

    def __init__(self, metadata: List[Dict]):
        print("建立 BM25 索引...")
        self.metadata = metadata

        self.corpus = []
        for meta in metadata:
            text = meta.get('combined_text', '')
            tokens = list(jieba.cut(text))
            self.corpus.append(tokens)

        self.bm25 = BM25Okapi(self.corpus)
        print(f"✓ BM25 完成 ({len(self.corpus):,} 筆)")

    def search(self, query: str, top_k: int = 50) -> List[Tuple[int, float]]:
        """返回原始 BM25 分數（不正規化）"""
        query_tokens = list(jieba.cut(query))
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [(int(idx), float(scores[idx]))
                for idx in top_indices if scores[idx] > 0]

    @staticmethod
    def save(bm25, path: str):
        with open(path, 'wb') as f:
            pickle.dump(bm25, f)
        print(f"✓ BM25 索引已儲存: {path}")

    @staticmethod
    def load(path: str):
        with open(path, 'rb') as f:
            return pickle.load(f)


class ComplementaryRetriever:
    """互補法檢索器（原始分數版本）"""

    def __init__(
            self,
            faiss_index_path: str,
            metadata_path: str,
            bm25_index_path: str = None,
            use_gpu_faiss: bool = False  # ← 新增參數
    ):
        print(f"載入 FAISS: {faiss_index_path}")
        self.faiss_index = faiss.read_index(faiss_index_path)

        print(f"載入 Metadata: {metadata_path}")
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)

        if bm25_index_path and os.path.exists(bm25_index_path):
            print(f"載入 BM25: {bm25_index_path}")
            self.bm25_index = BM25Index.load(bm25_index_path)
        else:
            self.bm25_index = BM25Index(self.metadata)
            if bm25_index_path:
                BM25Index.save(self.bm25_index, bm25_index_path)

        print(f"載入模型: {EMBEDDING_MODEL_NAME}")
        self.emb = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )
        print("✓ 初始化完成\n")

    def _faiss_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """FAISS 搜尋（返回原始分數）"""
        prefixed = f"passage: {query}"
        
        # --- 主要改動開始 ---
        # 1. 使用新的 embed_query() 方法來生成向量 (它會回傳一個 Python list)
        embedding_list = self.emb.embed_query(prefixed)
    
        # 2. 將 list 轉換成 FAISS 需要的 NumPy array 格式 (2D array)
        vec = np.array(embedding_list, dtype=np.float32).reshape(1, -1)
        # --- 主要改動結束 ---
    
        distances, indices = self.faiss_index.search(vec, top_k)
    
        return [(int(idx), float(d))
                for idx, d in zip(indices[0], distances[0]) if idx >= 0]

    def _bm25_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """BM25 搜尋（返回原始分數）"""
        return self.bm25_index.search(query, top_k)

    def _is_good_match(self, query: str, doc_text: str) -> bool:
        """
        判斷是否為有意義的匹配

        過濾掉只匹配到常見字的情況
        """
        query_tokens = set(jieba.cut(query))
        doc_tokens = set(jieba.cut(doc_text))
        matched_tokens = query_tokens & doc_tokens

        if not matched_tokens:
            return False

        # 匹配率：匹配的詞數 / 查詢詞數
        match_ratio = len(matched_tokens) / len(query_tokens)

        # 如果匹配超過 50% 的查詢詞，認為是好的匹配
        if match_ratio >= 0.5:
            return True

        # 常見字列表（這些字匹配意義不大）
        common_words = {
            '的', '是', '在', '和', '了', '有', '人', '我', '也', '就',
            '不', '都', '這', '個', '他', '你', '來', '說', '到', '著',
            '王', '李', '張', '陳', '林', '黃', '劉', '吳', '周', '徐'
        }

        # 檢查匹配的詞中，有多少不是常見字
        meaningful_matched = matched_tokens - common_words

        # 如果只匹配到常見字，認為不是好的匹配
        if len(meaningful_matched) == 0:
            return False

        # 如果匹配到至少一個有意義的詞，認為是好的匹配
        return True

    def search(
            self,
            query: str,
            top_k: int = 20,
            bm25_high_threshold: float = 16.0,
            bm25_med_threshold: float = 12.0,
            faiss_high_threshold: float = 0.90,
            faiss_low_threshold: float = 0.80
    ) -> List[Dict]:
        """
        互補法檢索（單 Part）- 改進版

        改進流程：
        1. 搜尋單一 part，收集候選
        2. 按照互補策略分層篩選
        3. 每層內部按分數排序
        4. 按優先級順序合併
        """
        # 搜尋（取較多結果以確保不遺漏）
        search_size = top_k * 2

        faiss_results = self._faiss_search(query, search_size)  # ← 改：單 Part 搜尋
        bm25_results = self._bm25_search(query, search_size)  # ← 改：單 Part 搜尋

        # 先排序
        bm25_results_sorted = sorted(bm25_results, key=lambda x: x[1], reverse=True)  # ← 改：x[1] 是分數
        faiss_results_sorted = sorted(faiss_results, key=lambda x: x[1], reverse=True)  # ← 改：x[1] 是分數

        # ========== 關鍵改動：分層收集 ==========
        tier1_bm25_high = []  # BM25 高分（最優先）
        tier2_faiss_high = []  # FAISS 高分
        tier3_bm25_med = []  # BM25 中分
        tier4_faiss_low = []  # FAISS 低分（補充）

        seen = set()  # ← 改：只存 idx，不是 (part_num, local_idx)

        # 階段 1：收集 BM25 高分
        for idx, score in bm25_results_sorted:  # ← 改：只有 idx 和 score
            if score > bm25_high_threshold:
                if idx in seen:  # ← 改：直接用 idx
                    continue

                meta = self.metadata[idx].copy()  # ← 改：直接存取 metadata
                doc_text = meta.get('combined_text', '')

                if self._is_good_match(query, doc_text):
                    meta['score'] = score
                    meta['source'] = 'bm25_high'
                    meta['reason'] = f'精確匹配 (BM25={score:.2f})'
                    meta['priority'] = 1
                    tier1_bm25_high.append(meta)
                    seen.add(idx)  # ← 改：只加 idx

        # 階段 2：收集 FAISS 高分
        for idx, score in faiss_results_sorted:  # ← 改：只有 idx 和 score
            if score > faiss_high_threshold:
                if idx in seen:  # ← 改：直接用 idx
                    continue

                meta = self.metadata[idx].copy()  # ← 改：直接存取 metadata
                meta['score'] = score
                meta['source'] = 'faiss_high'
                meta['reason'] = f'高度相關 (FAISS={score:.3f})'
                meta['priority'] = 2
                tier2_faiss_high.append(meta)
                seen.add(idx)  # ← 改：只加 idx

        # 階段 3：收集 BM25 中分
        for idx, score in bm25_results_sorted:  # ← 改：只有 idx 和 score
            if score > bm25_med_threshold:
                if idx in seen:  # ← 改：直接用 idx
                    continue

                meta = self.metadata[idx].copy()  # ← 改：直接存取 metadata
                doc_text = meta.get('combined_text', '')

                if self._is_good_match(query, doc_text):
                    meta['score'] = score
                    meta['source'] = 'bm25_med'
                    meta['reason'] = f'部分匹配 (BM25={score:.2f})'
                    meta['priority'] = 3
                    tier3_bm25_med.append(meta)
                    seen.add(idx)  # ← 改：只加 idx

        # 階段 4：收集 FAISS 補充
        for idx, score in faiss_results_sorted:  # ← 改：只有 idx 和 score
            if score > faiss_low_threshold:
                if idx in seen:  # ← 改：直接用 idx
                    continue

                meta = self.metadata[idx].copy()  # ← 改：直接存取 metadata
                meta['score'] = score
                meta['source'] = 'faiss_low'
                meta['reason'] = f'相關 (FAISS={score:.3f})'
                meta['priority'] = 4
                tier4_faiss_low.append(meta)
                seen.add(idx)  # ← 改：只加 idx

        # ========== 關鍵改動：層內排序 ==========
        tier1_bm25_high.sort(key=lambda x: x['score'], reverse=True)
        tier2_faiss_high.sort(key=lambda x: x['score'], reverse=True)
        tier3_bm25_med.sort(key=lambda x: x['score'], reverse=True)
        tier4_faiss_low.sort(key=lambda x: x['score'], reverse=True)

        # ========== 關鍵改動：按優先級合併 ==========
        final = []
        final.extend(tier1_bm25_high)
        final.extend(tier2_faiss_high)
        final.extend(tier3_bm25_med)
        final.extend(tier4_faiss_low)

        # 輸出調試資訊（移除 Part 資訊）
        print(f"\n🔍 檢索結果統計:")
        print(f"  Tier 1 (BM25 高分): {len(tier1_bm25_high)} 篇")
        print(f"  Tier 2 (FAISS 高分): {len(tier2_faiss_high)} 篇")
        print(f"  Tier 3 (BM25 中分): {len(tier3_bm25_med)} 篇")
        print(f"  Tier 4 (FAISS 補充): {len(tier4_faiss_low)} 篇")
        print(f"  總計: {len(final)} 篇")

        if final:
            print(f"\n📊 前 5 筆結果:")
            for i, r in enumerate(final[:5], 1):
                # ← 改：移除 Part 資訊
                print(f"  {i}. {r['source']:12} | 分數: {r['score']:.3f}")

        return final[:top_k]


def build_bm25_index(metadata_path: str, output_path: str):
    """建立 BM25 索引"""
    print("建立 BM25 索引...\n")

    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)

    bm25 = BM25Index(metadata)
    BM25Index.save(bm25, output_path)

    print("\n完成！")


if __name__ == "__main__":
    # 建立 BM25 索引
    build_bm25_index(
        metadata_path="faiss_index_part1/ptt_baseball_metadata_part1.pkl",
        output_path="faiss_index_part1/bm25_index.pkl"
    )