from utils.gpt_utils import call_gpt
from config import Config
import pandas as pd

AUDIT_SYSTEM_PROMPT = """
Bạn là chuyên gia kiểm định dữ liệu cho bộ dữ liệu huấn luyện
mô hình cho bản tin thời tiết VTV.

Với mỗi câu (text), cùng content_label, quality_level, thesis_score, duration:

1) WEATHER RELEVANCE:
   - Có thật sự nói về thời tiết hoặc dẫn dắt trong chương trình thời tiết?

2) NATURALNESS & GRAMMAR:
   - Câu có tự nhiên, đúng ngữ pháp, dễ hiểu?

3) INFORMATION CONTENT:
   - Có chứa thông tin hữu ích cho dataset thời tiết?

4) REDUNDANCY:
   - Có phải câu lặp lại nội dung đã xuất hiện nhiều lần?

ĐẦU RA:

FLAG|SCORE|NOTE

FLAG ∈ {OK, WARN, REMOVE}
SCORE: 0-100 (mức độ phù hợp để huấn luyện)
NOTE: ghi chú ngắn, tiếng Việt.
"""

from utils.gpt_utils import client

def audit_one_segment(video_id: str, clip_id: str, text: str, content_label: str, quality_level: str, thesis_score: int, duration: float) -> tuple:
    if not text or not text.strip():
        return "REMOVE", 0, "Câu trống."

    user_prompt = f"""
VIDEO_ID: {video_id}
CLIP_ID: {clip_id}
DURATION: {duration}
CONTENT_LABEL: {content_label}
QUALITY_LEVEL: {quality_level}
THESIS_SCORE: {thesis_score}

TEXT:
{text}

Hãy trả về đúng format: FLAG|SCORE|NOTE
"""
    
    response = client.chat.completions.create(
        model=Config.OPENAI_MODEL,
        messages=[
            {"role": "system", "content": AUDIT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0
    )
    
    raw = response.choices[0].message.content.strip()
    flag = "OK"
    score = 70
    note = ""

    if "|" in raw:
        parts = raw.split("|", 2)
        flag = parts[0].strip().upper()
        if len(parts) > 1:
            digits = "".join(ch for ch in parts[1] if ch.isdigit())
            try:
                score = int(digits)
            except Exception:
                score = 70
        if len(parts) > 2:
            note = parts[2].strip()
    else:
        flag = raw.strip().upper()
        score = 70
    
    if flag not in ["OK", "WARN", "REMOVE"]:
        flag = "WARN"
    score = max(0, min(100, score))
    
    return flag, score, note

DEDUP_SYSTEM_PROMPT = """
Bạn là chuyên gia xử lý dữ liệu thời tiết VTV.
Bạn được đưa 2 câu thoại A và B thuộc cùng timestamp.

Luôn chọn MỘT câu tốt nhất để giữ lại.

TRẢ VỀ DUY NHẤT:

KEEP_A    → nếu A tốt hơn
KEEP_B    → nếu B tốt hơn

Không giải thích, không bình luận.
"""

def ask_agent_dedup(text_a: str, text_b: str) -> str:
    user_prompt = f"""
CÂU A:
{text_a}

CÂU B:
{text_b}

Hãy chọn KEEP_A hoặc KEEP_B.
"""
    response = client.chat.completions.create(
        model=Config.OPENAI_MODEL_MINI,
        messages=[
            {"role": "system", "content": DEDUP_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0
    )
    return response.choices[0].message.content.strip().upper()

def detect_and_resolve_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    # Logic from stage4.ipynb: 
    # Group by [video_id, start, end] -> filter len > 1 -> Group by again
    
    dup_groups = (
        df.groupby(["video_id", "start", "end"])
        .filter(lambda g: len(g) > 1)
        .groupby(["video_id", "start", "end"])
    )
    
    print(f"Số timestamp có ≥ 2 câu: {len(dup_groups)}")
    
    to_drop = []
    
    for (vid, s, e), group in dup_groups:
        idxs = list(group.index)
        
        # 'champion' strategy for group >= 2
        champion_idx = idxs[0]
        for idx in idxs[1:]:
            text_A = df.loc[champion_idx, "text_final"]
            text_B = df.loc[idx, "text_final"]
            
            decision = ask_agent_dedup(text_A, text_B)
            
            print(f"\n====== DUP TIMESTAMP ======")
            print(f"VIDEO: {vid} | {s} → {e}")
            print(f"A: {text_A}")
            print(f"B: {text_B}")
            print(f"Agent: {decision}")
            
            if decision == "KEEP_A":
                to_drop.append(idx)
            elif decision == "KEEP_B":
                to_drop.append(champion_idx)
                champion_idx = idx
            else:
                # fallback: keep A
                to_drop.append(idx)
                
    df_clean = df.drop(to_drop).reset_index(drop=True)
    df_clean['is_duplicate'] = False # For compatibility if needed downstream, but we are dropping them here.
    
    return df_clean
