from utils.gpt_utils import call_gpt
from config import Config

CLASSIFY_SYSTEM_PROMPT = """
Bạn là bộ phân loại nội dung câu thoại trong chương trình truyền hình, đặc biệt là chương trình thời tiết VTV.

Nhiệm vụ: đọc một câu tiếng Việt (text_final) và phân loại vào đúng 1 trong 3 NHÓM:

1) WEATHER_CORE:
   - Câu nói trực tiếp về nội dung THỜI TIẾT: mưa, nắng, bão, lũ, áp thấp, không khí lạnh, thiên tai...
   - Chứa số liệu/thông tin chính: bao nhiêu mm mưa, bao nhiêu độ C, mực nước, cảnh báo...

2) WEATHER_SUPPORT:
   - Lời dẫn, chuyển mạch, chào hỏi, kết thúc nhưng vẫn thuộc chương trình thời tiết.
   - Không có số liệu, nhưng rõ ràng trong ngữ cảnh bản tin thời tiết.

3) NON_WEATHER:
   - Quảng cáo, kêu gọi đăng ký kênh, giới thiệu chương trình khác, câu không liên quan thời tiết.

Chỉ trả về đúng một trong 3 từ:
WEATHER_CORE, WEATHER_SUPPORT, NON_WEATHER.
"""

SCORE_SYSTEM_PROMPT = """
Bạn là bộ chấm điểm mức độ HỮU ÍCH của một câu thoại cho dataset nghiên cứu về bản tin thời tiết VTV.

Hãy cho điểm từ 0 đến 100 dựa trên:
- Nội dung thời tiết càng rõ, cụ thể về hiện tượng, địa điểm, thời gian, mức độ → điểm cao.
- WEATHER_CORE thường cao hơn WEATHER_SUPPORT.
- Chất lượng câu (HIGH/MEDIUM) càng tốt → điểm càng cao.
- Câu chung chung, ít thông tin → điểm thấp.

Chỉ trả về MỘT số nguyên từ 0 đến 100.
"""

from utils.gpt_utils import client

def classify_weather_segment(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return "NON_WEATHER"

    response = client.chat.completions.create(
        model=Config.OPENAI_MODEL_MINI,
        messages=[
            {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
            {"role": "user", "content": f"Văn bản:\n{text}\n\nPhân loại:"}
        ],
        temperature=0.0
    )
    out = response.choices[0].message.content.strip().upper()
    if "WEATHER_CORE" in out:
        return "WEATHER_CORE"
    if "WEATHER_SUPPORT" in out:
        return "WEATHER_SUPPORT"
    return "NON_WEATHER"

def score_segment_for_thesis(text: str, content_label: str, quality_level: str = "MEDIUM", duration: float = 0.0) -> int:
    user = f"""
TEXT_FINAL: {text}
content_label: {content_label}
quality_level: {quality_level}
duration_seconds: {duration:.2f}

Cho điểm 0–100 (chỉ 1 số).
"""
    response = client.chat.completions.create(
        model=Config.OPENAI_MODEL_MINI,
        messages=[
            {"role": "system", "content": SCORE_SYSTEM_PROMPT},
            {"role": "user", "content": user}
        ],
        temperature=0.0
    )
    raw = response.choices[0].message.content.strip()
    digits = "".join(ch for ch in raw if ch.isdigit())
    try:
        score = int(digits)
        return max(0, min(100, score))
    except Exception:
        return 50
