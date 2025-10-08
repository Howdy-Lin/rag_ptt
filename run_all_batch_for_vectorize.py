#!/usr/bin/env python3
# run_all_batches.py
# 自動執行所有批次的向量化

import os
import sys
import sqlite3
import subprocess
import time
from datetime import datetime

# ==================== 設定 ====================
DB_NAME = "ptt_baseball.db"
DOCS_PER_BATCH = 2000000
VECTORIZE_SCRIPT = "vectorize_faiss.py"


# ==================== 計算總批次數 ====================
def get_total_batches(db_name: str, docs_per_batch: int) -> int:
    """計算需要多少個批次"""
    if not os.path.exists(db_name):
        print(f"錯誤：找不到資料庫 {db_name}")
        sys.exit(1)

    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM comments")
    total_comments = cursor.fetchone()[0]
    conn.close()

    total_batches = (total_comments + docs_per_batch - 1) // docs_per_batch

    print(f"資料庫統計:")
    print(f"  總評論數: {total_comments:,}")
    print(f"  每批數量: {docs_per_batch:,}")
    print(f"  需要批次: {total_batches}")
    print()

    return total_batches, total_comments


# ==================== 執行單一批次 ====================
def run_batch(batch_number: int) -> bool:
    """
    執行單一批次

    Args:
        batch_number: 批次編號 (1-based)

    Returns:
        bool: 是否成功
    """
    print(f"\n{'=' * 60}")
    print(f"開始執行批次 {batch_number}")
    print(f"{'=' * 60}\n")

    start_time = time.time()

    # 修改 vectorize_faiss.py 中的 BATCH_NUMBER
    try:
        with open(VECTORIZE_SCRIPT, 'r', encoding='utf-8') as f:
            content = f.read()

        # 替換 BATCH_NUMBER
        import re
        content = re.sub(
            r'BATCH_NUMBER = \d+',
            f'BATCH_NUMBER = {batch_number}',
            content
        )

        # 寫回檔案
        with open(VECTORIZE_SCRIPT, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"已更新 BATCH_NUMBER = {batch_number}")
    except Exception as e:
        print(f"錯誤：更新 BATCH_NUMBER 失敗 - {e}")
        return False

    # 執行 vectorize_faiss.py
    try:
        result = subprocess.run(
            [sys.executable, VECTORIZE_SCRIPT],
            check=True,
            capture_output=False,  # 顯示即時輸出
            text=True
        )

        elapsed = time.time() - start_time
        print(f"\n批次 {batch_number} 完成！耗時: {elapsed / 60:.1f} 分鐘")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n錯誤：批次 {batch_number} 執行失敗 - {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n警告：批次 {batch_number} 被使用者中斷")
        return False


# ==================== 檢查批次是否已完成 ====================
def is_batch_completed(batch_number: int) -> bool:
    """檢查批次是否已經完成"""
    index_dir = f"faiss_index_part{batch_number}"
    index_file = os.path.join(index_dir, f"ptt_baseball_part{batch_number}.index")
    metadata_file = os.path.join(index_dir, f"ptt_baseball_metadata_part{batch_number}.pkl")

    return os.path.exists(index_file) and os.path.exists(metadata_file)


# ==================== 主程式 ====================
def main():
    print("=" * 60)
    print("批次向量化自動執行器")
    print("=" * 60)
    print()

    # 檢查腳本是否存在
    if not os.path.exists(VECTORIZE_SCRIPT):
        print(f"錯誤：找不到 {VECTORIZE_SCRIPT}")
        sys.exit(1)

    # 計算總批次數
    total_batches, total_comments = get_total_batches(DB_NAME, DOCS_PER_BATCH)

    # 詢問是否繼續
    print(f"即將執行 {total_batches} 個批次")
    response = input("是否繼續？(y/n): ").strip().lower()
    if response != 'y':
        print("已取消")
        sys.exit(0)

    # 記錄開始時間
    global_start = time.time()
    completed_batches = []
    failed_batches = []
    skipped_batches = []

    # 執行每個批次
    for batch_num in range(1, total_batches + 1):
        # 檢查是否已完成
        if is_batch_completed(batch_num):
            print(f"\n批次 {batch_num} 已存在，跳過")
            skipped_batches.append(batch_num)
            continue

        # 執行批次
        success = run_batch(batch_num)

        if success:
            completed_batches.append(batch_num)
        else:
            failed_batches.append(batch_num)

            # 詢問是否繼續
            print(f"\n批次 {batch_num} 失敗")
            response = input("是否繼續下一批次？(y/n): ").strip().lower()
            if response != 'y':
                print("已停止")
                break

    # 總結
    total_elapsed = time.time() - global_start
    print("\n" + "=" * 60)
    print("執行完成")
    print("=" * 60)
    print(f"總耗時: {total_elapsed / 3600:.2f} 小時")
    print(f"完成批次: {len(completed_batches)} / {total_batches}")

    if skipped_batches:
        print(f"跳過批次: {len(skipped_batches)} 個 (已存在)")
        print(f"  {skipped_batches}")

    if completed_batches:
        print(f"新完成批次: {completed_batches}")

    if failed_batches:
        print(f"失敗批次: {failed_batches}")
    else:
        print("所有批次執行成功")

    print("\n下一步：執行 merge_faiss_indices.py 合併所有批次")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程式被使用者中斷")
        sys.exit(1)