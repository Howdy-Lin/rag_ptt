# vectorize_faiss.py
# FAISS 版本 - 適配 PTT Baseball 資料庫結構

try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass

import os
import time
import sqlite3
import threading
import queue
import pickle
from typing import List, Dict
from collections import deque

import numpy as np
import torch
import faiss
from tqdm import tqdm
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# ==================== 分批設定 ====================
ENABLE_BATCH_MODE = True
BATCH_NUMBER = 9  # 手動改：1, 2, 3, ..., 11
DOCS_PER_BATCH = 2000000
BATCH_OFFSET = (BATCH_NUMBER - 1) * DOCS_PER_BATCH if ENABLE_BATCH_MODE else 0

# 記憶體優化：分段儲存 metadata
METADATA_FLUSH_SIZE = 50000  # 每 5 萬筆寫一次硬碟（降低記憶體）

# ==================== 設定 ====================
DB_NAME = "ptt_baseball.db"
OUTPUT_DIR = f"faiss_index_part{BATCH_NUMBER}" if ENABLE_BATCH_MODE else "faiss_index"
INDEX_NAME = f"ptt_baseball_part{BATCH_NUMBER}.index" if ENABLE_BATCH_MODE else "ptt_baseball.index"
METADATA_NAME = f"ptt_baseball_metadata_part{BATCH_NUMBER}.pkl" if ENABLE_BATCH_MODE else "ptt_baseball_metadata.pkl"

os.makedirs(OUTPUT_DIR, exist_ok=True)

EMBEDDING_MODEL_NAME = "BAAI/bge-base-zh-v1.5"
EMBEDDING_DIM = 768

DB_READ_BATCH_SIZE = 4096
EMBED_BATCH_SIZE = 2048
FLUSH_EVERY_DOCS = 8192
WRITE_BATCH_SIZE = 5000

READ_QUEUE_MAXSIZE = 2
WRITE_QUEUE_MAXSIZE = 4

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
MAX_SEQ_LEN = 128

USE_GPU = True
FINAL_INDEX_TYPE = "HNSW"


# ----------------- 速度監控器 -----------------
class SpeedMonitor:
    def __init__(self, window_size=10):
        self.speeds = deque(maxlen=window_size)
        self.last_count = 0
        self.last_time = time.time()

    def update(self, current_count):
        now = time.time()
        elapsed = now - self.last_time
        if elapsed > 0:
            speed = (current_count - self.last_count) / elapsed
            self.speeds.append(speed)
        self.last_count = current_count
        self.last_time = now

    def get_avg_speed(self):
        if not self.speeds:
            return 0
        return sum(self.speeds) / len(self.speeds)

    def is_slowing_down(self, threshold=0.3):
        if len(self.speeds) < 5:
            return False
        recent = list(self.speeds)[-3:]
        older = list(self.speeds)[:-3]
        if not older:
            return False
        avg_recent = sum(recent) / len(recent)
        avg_older = sum(older) / len(older)
        return avg_recent < avg_older * (1 - threshold)


# ----------------- 工具函數 -----------------
def _clean_text(text: str) -> str:
    if text is None:
        return ""
    t = str(text).strip()
    return " ".join(t.split())


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    return cur.fetchone() is not None


def get_total_count(db_name: str) -> int:
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    if ENABLE_BATCH_MODE:
        cursor.execute("SELECT COUNT(*) FROM comments")
        db_total = cursor.fetchone()[0]
        start = BATCH_OFFSET
        end = min(BATCH_OFFSET + DOCS_PER_BATCH, db_total)
        total = max(0, end - start)
    else:
        cursor.execute("SELECT COUNT(*) FROM comments")
        total = cursor.fetchone()[0]

    conn.close()
    return total


# ----------------- DB 讀取 -----------------
def stream_from_db(batch_size: int, pbar: tqdm):
    """從資料庫讀取並串流處理評論資料"""
    if not os.path.exists(DB_NAME):
        print(f"找不到 SQLite 檔案：{DB_NAME}")
        return

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # SQLite 優化
    cursor.execute("PRAGMA synchronous = OFF;")
    cursor.execute("PRAGMA journal_mode = OFF;")
    cursor.execute("PRAGMA temp_store = MEMORY;")
    cursor.execute("PRAGMA cache_size = -200000;")
    cursor.execute("PRAGMA mmap_size = 2147483648;")
    cursor.execute("PRAGMA read_uncommitted = ON;")

    try:
        if not _table_exists(conn, "comments"):
            print("資料表 comments 不存在。")
            return

        # 檢查 articles 表
        has_articles = _table_exists(conn, "articles")

        # 建構查詢 - 根據你的資料庫結構
        if has_articles:
            query = """
                SELECT 
                    c.comment_id,
                    c.content,
                    c.article_id,
                    c.push_time,
                    c.user_id,
                    c.push_tag,
                    a.title,
                    a.author,
                    a.post_time,
                    a.url
                FROM comments c
                LEFT JOIN articles a ON c.article_id = a.article_id
            """
        else:
            query = """
                SELECT 
                    comment_id,
                    content,
                    article_id,
                    push_time,
                    user_id,
                    push_tag,
                    NULL as title,
                    NULL as author,
                    NULL as post_time,
                    NULL as url
                FROM comments
            """

        if ENABLE_BATCH_MODE:
            query += " LIMIT ? OFFSET ?"
            cursor.execute(query, (DOCS_PER_BATCH, BATCH_OFFSET))
        else:
            cursor.execute(query)

        processed = 0
        while True:
            if ENABLE_BATCH_MODE and processed >= DOCS_PER_BATCH:
                break

            rows = cursor.fetchmany(batch_size)
            if not rows:
                break

            texts, metas = [], []
            for row in rows:
                (comment_id, content, article_id, push_time, user_id,
                 push_tag, title, author, post_time, url) = row

                comment_text = _clean_text(content)
                if not comment_text:
                    continue

                # 清理標題
                title_text = _clean_text(title) if title else ""

                # 融合標題和內容
                if title_text:
                    combined_text = f"[{title_text}] {comment_text}"
                else:
                    combined_text = comment_text

                texts.append(combined_text)  # ← 向量化融合後的文字

                metas.append({
                    "doc_id": f"comment_{comment_id}",
                    "comment_id": str(comment_id),
                    "article_id": str(article_id),
                    "comment_content": comment_text,
                    "combined_text": combined_text,  # ← 加這行！
                    "push_time": str(push_time) if push_time else None,
                    "user_id": str(user_id) if user_id else None,
                    "push_tag": str(push_tag) if push_tag else None,
                    "article_title": str(title) if title else None,
                    "article_author": str(author) if author else None,
                    "article_post_time": str(post_time) if post_time else None,
                    "article_url": str(url) if url else None,
                })

            processed += len(rows)
            pbar.update(len(rows))
            if texts:
                yield texts, metas
    finally:
        conn.close()


# ----------------- Embedding 模型 -----------------
def set_effective_max_seq_length(emb, max_len: int = MAX_SEQ_LEN):
    ok = False
    if hasattr(emb, "client") and hasattr(emb.client, "max_seq_length"):
        try:
            emb.client.max_seq_length = max_len
            ok = True
        except Exception:
            pass
    if hasattr(emb, "client") and hasattr(emb.client, "tokenizer"):
        tok = emb.client.tokenizer
        try:
            tok.model_max_length = max_len
            ok = True
        except Exception:
            pass
    if ok:
        print(f"已設定 max_seq_length={max_len}")


def build_embedding_model(embed_batch_size: int) -> HuggingFaceBgeEmbeddings:
    print(f"初始化嵌入模型 '{EMBEDDING_MODEL_NAME}' - device={DEVICE}, dtype={DTYPE}, batch={embed_batch_size}")
    print("載入模型...")

    print("   1/3 載入模型權重...")
    emb = HuggingFaceBgeEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": embed_batch_size,
            "show_progress_bar": False,
            "num_workers": 4,
        },
        #query_instruction="query: ",
        #embed_instruction="passage: ",
    )

    print("   2/3 移動模型到 GPU...")
    try:
        emb.client._target_device = torch.device(DEVICE)
        emb.client.auto_model.to(dtype=DTYPE, device=DEVICE)
        print("模型載入完成")
    except Exception as e:
        print(f"警告：模型移動到 GPU 失敗 - {e}")

    set_effective_max_seq_length(emb, max_len=MAX_SEQ_LEN)

    print("   3/3 Warmup GPU...")
    try:
        _ = emb.client.encode(
            ["passage: warmup " * 20],
            batch_size=min(8, embed_batch_size),
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        print("Warmup 完成")
    except Exception:
        pass
    return emb


# ----------------- FAISS Index -----------------
def create_faiss_index(dimension: int, use_gpu: bool = False) -> faiss.Index:
    print(f"建立 FAISS Flat Index (dim={dimension})")
    index = faiss.IndexFlatIP(dimension)

    if use_gpu and faiss.get_num_gpus() > 0:
        print("使用 GPU 加速 FAISS")
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    return index


def convert_to_hnsw(flat_index: faiss.Index, dimension: int) -> faiss.Index:
    print("\n轉換為 HNSW Index...")

    if isinstance(flat_index, faiss.GpuIndex):
        flat_index = faiss.index_gpu_to_cpu(flat_index)

    n = flat_index.ntotal
    print(f"重建索引中... ({n:,} 筆資料)")

    with tqdm(total=n, desc="提取向量", unit="docs") as pbar:
        vectors = np.zeros((n, dimension), dtype=np.float32)
        batch_size = 10000
        for i in range(0, n, batch_size):
            j = min(i + batch_size, n)
            vectors[i:j] = flat_index.reconstruct_n(i, j - i)
            pbar.update(j - i)

    M = 32
    ef_construction = 200

    hnsw_index = faiss.IndexHNSWFlat(dimension, M)
    hnsw_index.hnsw.efConstruction = ef_construction

    print("寫入 HNSW 索引...")
    with tqdm(total=n, desc="建立 HNSW", unit="docs") as pbar:
        batch_size = 10000
        for i in range(0, n, batch_size):
            j = min(i + batch_size, n)
            hnsw_index.add(vectors[i:j])
            pbar.update(j - i)

    return hnsw_index


# ----------------- Pipeline -----------------
def reader_thread_fn(read_q: queue.Queue, total_count: int):
    pbar = tqdm(total=total_count, desc="讀取DB", unit="rows", position=0)
    try:
        for texts, metas in stream_from_db(DB_READ_BATCH_SIZE, pbar):
            read_q.put((texts, metas))
    finally:
        pbar.close()
        read_q.put(None)


def _embed_and_enqueue(write_q: queue.Queue, emb: HuggingFaceBgeEmbeddings,
                       texts: List[str], metas: List[Dict]):
    t0 = time.perf_counter()

    prefixed = [f"passage: {t}" for t in texts]
    vecs: np.ndarray = emb.client.encode(
        prefixed,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    if DEVICE == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    took = time.perf_counter() - t0
    write_q.put((vecs.astype(np.float32), metas, took))


def embedder_thread_fn(read_q: queue.Queue, write_q: queue.Queue,
                       emb: HuggingFaceBgeEmbeddings, total_count: int):
    buffer_texts = []
    buffer_metas = []

    pbar = tqdm(total=total_count, desc="Embedding", unit="docs", position=1)
    speed_monitor = SpeedMonitor(window_size=10)
    processed = 0

    while True:
        item = read_q.get()

        if item is None:
            if buffer_texts:
                _embed_and_enqueue(write_q, emb, buffer_texts, buffer_metas)
                pbar.update(len(buffer_texts))
            pbar.close()
            write_q.put(None)
            break

        texts, metas = item
        buffer_texts.extend(texts)
        buffer_metas.extend(metas)

        if len(buffer_texts) >= FLUSH_EVERY_DOCS:
            _embed_and_enqueue(write_q, emb, buffer_texts, buffer_metas)
            pbar.update(len(buffer_texts))
            processed += len(buffer_texts)

            speed_monitor.update(processed)
            avg_speed = speed_monitor.get_avg_speed()
            pbar.set_postfix({"速度": f"{avg_speed:.0f} docs/s"})

            if speed_monitor.is_slowing_down():
                tqdm.write("警告：Embedding 速度下降中...")

            buffer_texts = []
            buffer_metas = []


def writer_thread_fn(write_q: queue.Queue, index: faiss.Index, total_count: int):
    """寫入 FAISS（邊寫邊存 metadata）"""
    pbar = tqdm(total=total_count, desc="寫入FAISS", unit="docs", position=2)
    speed_monitor = SpeedMonitor(window_size=10)
    written = 0
    embed_times = []

    # 分段儲存 metadata
    metadata_buffer = []
    metadata_chunk_idx = 0

    while True:
        item = write_q.get()
        if item is None:
            # 存剩餘的 metadata
            if metadata_buffer:
                metadata_chunk_idx += 1
                chunk_path = os.path.join(OUTPUT_DIR, f"metadata_chunk_{metadata_chunk_idx}.pkl")
                with open(chunk_path, "wb") as f:
                    pickle.dump(metadata_buffer, f)
                tqdm.write(f"已存 chunk {metadata_chunk_idx}: {len(metadata_buffer):,} 筆")

            pbar.close()
            if embed_times:
                avg_time = sum(embed_times) / len(embed_times)
                print(f"\n平均 Embedding 時間：{avg_time:.2f}s / batch")
            break

        vecs, metas, embed_time = item
        embed_times.append(embed_time)

        n = len(vecs)
        t0 = time.perf_counter()

        # FAISS 寫入
        index.add(vecs)
        metadata_buffer.extend(metas)

        # 定期存檔釋放記憶體
        if len(metadata_buffer) >= METADATA_FLUSH_SIZE:
            metadata_chunk_idx += 1
            chunk_path = os.path.join(OUTPUT_DIR, f"metadata_chunk_{metadata_chunk_idx}.pkl")
            with open(chunk_path, "wb") as f:
                pickle.dump(metadata_buffer, f)
            tqdm.write(f"已存 chunk {metadata_chunk_idx}: {len(metadata_buffer):,} 筆")
            metadata_buffer = []

        took = time.perf_counter() - t0
        written += n
        pbar.update(n)

        speed_monitor.update(written)
        avg_speed = speed_monitor.get_avg_speed()
        write_speed = n / max(took, 1e-6)

        pbar.set_postfix({
            "寫入": f"{write_speed:.0f} docs/s",
            "平均": f"{avg_speed:.0f} docs/s"
        })

        if speed_monitor.is_slowing_down():
            tqdm.write("警告：寫入速度下降中...")


# ----------------- 主流程 -----------------
def vectorize_pipeline():
    print("正在統計資料總數...")
    total_count = get_total_count(DB_NAME)

    if ENABLE_BATCH_MODE:
        print(f"\n{'=' * 60}")
        print(f"分批模式 - 批次 {BATCH_NUMBER}")
        print(f"  偏移：{BATCH_OFFSET:,}")
        print(f"  本批數量：{total_count:,}")
        print(f"  輸出目錄：{OUTPUT_DIR}")
        print(f"{'=' * 60}\n")
    else:
        print(f"資料庫總筆數：{total_count:,}\n")

    emb = build_embedding_model(EMBED_BATCH_SIZE)
    index = create_faiss_index(EMBEDDING_DIM, use_gpu=USE_GPU)

    print(f"FAISS Index 建立完成\n")

    read_q = queue.Queue(maxsize=READ_QUEUE_MAXSIZE)
    write_q = queue.Queue(maxsize=WRITE_QUEUE_MAXSIZE)

    reader = threading.Thread(target=reader_thread_fn, args=(read_q, total_count), daemon=True)
    embedder = threading.Thread(target=embedder_thread_fn, args=(read_q, write_q, emb, total_count), daemon=True)
    writer = threading.Thread(target=writer_thread_fn, args=(write_q, index, total_count), daemon=True)

    t0 = time.perf_counter()
    reader.start()
    embedder.start()
    writer.start()

    print("開始處理...\n")

    reader.join()
    embedder.join()
    writer.join()

    elapsed = time.perf_counter() - t0
    total_vectors = index.ntotal

    print("\n" + "=" * 60)
    print(f"向量化完成！共 {total_vectors:,} 筆")
    print(f"總耗時：{elapsed / 60:.1f} 分鐘")
    print(f"平均速度：{total_vectors / elapsed:,.0f} docs/s")
    print("=" * 60)

    # 轉換為 HNSW
    if FINAL_INDEX_TYPE == "HNSW" and total_vectors > 0:
        final_index = convert_to_hnsw(index, EMBEDDING_DIM)
    else:
        final_index = index
        if isinstance(final_index, faiss.GpuIndex):
            final_index = faiss.index_gpu_to_cpu(final_index)

    # 儲存索引
    index_path = os.path.join(OUTPUT_DIR, INDEX_NAME)
    print(f"\n儲存 FAISS Index 到：{index_path}")
    faiss.write_index(final_index, index_path)

    # 合併所有 metadata chunks
    print(f"\n合併 Metadata chunks...")
    all_metadata = []
    chunk_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.startswith("metadata_chunk_")])

    for chunk_file in tqdm(chunk_files, desc="合併 chunks"):
        chunk_path = os.path.join(OUTPUT_DIR, chunk_file)
        with open(chunk_path, "rb") as f:
            chunk_data = pickle.load(f)
            all_metadata.extend(chunk_data)
        os.remove(chunk_path)

    metadata_path = os.path.join(OUTPUT_DIR, METADATA_NAME)
    print(f"儲存完整 Metadata 到：{metadata_path}")
    with open(metadata_path, "wb") as f:
        pickle.dump(all_metadata, f)

    print("\n" + "=" * 60)
    print("完成！")
    print(f"   - Index: {index_path}")
    print(f"   - Metadata: {metadata_path}")
    print(f"   - 總筆數: {len(all_metadata):,}")

    # 清理記憶體
    del all_metadata
    del final_index
    import gc
    gc.collect()

    print("=" * 60)


if __name__ == "__main__":
    vectorize_pipeline()