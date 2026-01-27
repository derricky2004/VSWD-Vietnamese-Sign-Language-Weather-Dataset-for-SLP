import json
from pathlib import Path
from openai import OpenAI
from config import Config

client = OpenAI(api_key=Config.OPENAI_API_KEY)

def call_gpt(system_prompt: str, user_message: str, model: str = Config.OPENAI_MODEL_MINI) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

def run_gpt4o_full_transcript(audio_path: Path, out_path: Path) -> dict:
    with open(audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language="vi"
        )
    
    full_text = transcription.text
    result = {"text": full_text}
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    return result

REFINE_PROMPT_SYSTEM = """
Bạn là biên tập viên cao cấp của Bản Tin Thời Tiết VTV.

Nhiệm vụ: biên tập 1 câu ASR thành câu tiếng Việt chuẩn VTV, giữ nguyên NGHĨA 100%.

PHẢI GIỮ NGUYÊN:
- địa danh
- thời gian
- số liệu
- mức độ sự kiện
- logic nguyên nhân - kết quả

ĐƯỢC PHÉP:
- sửa chính tả, dấu câu
- sửa từ sai → đúng theo transcript
- loại bỏ filler
- viết lại mượt hơn

KHÔNG ĐƯỢC:
- thêm sự kiện mới
- thay đổi mức độ thiệt hại
- thêm địa danh/số liệu/thời gian mới
- đảo nghĩa
"""

# CONSIST_PROMPT_SYSTEM = """
# Bạn là bộ kiểm định văn bản.

# Nhiệm vụ: Kiểm tra xem câu có cần sửa lỗi chính tả, dấu câu hay không.

# Nếu câu đư -> OK.
# Nếu cần sửa lỗi chính tả, dấu câu -> REVERT.

# Chỉ trả lời OK hoặc REVERT.
# """


QUALITY_PROMPT_SYSTEM = """
Bạn chấm chất lượng câu bản tin thời tiết.

HIGH:
- rõ, mạch lạc, chính xác
- câu chuẩn, dễ hiểu

MEDIUM:
- còn vài lỗi nhưng ý vẫn rõ

LOW:
- mơ hồ, khó hiểu, lỗi nhiều, không liên quan đến thời tiết.

Chỉ trả lời HIGH / MEDIUM / LOW.
"""

def refine_with_gpt(text_whisper: str, full_transcript: str) -> str:
    user = f"""
Câu ASR:
{text_whisper}

Transcript đầy đủ (tham chiếu):
{full_transcript}

Hãy viết lại thành 1 câu rõ ràng, chuẩn VTV, giữ nguyên nghĩa.
"""
    response = client.chat.completions.create(
        model=Config.OPENAI_MODEL_MINI,
        messages=[
            {"role": "system", "content": REFINE_PROMPT_SYSTEM},
            {"role": "user", "content": user}
        ],
        temperature=0.15
    )
    return response.choices[0].message.content.strip()

def review_ok_revert(original: str, cleaned: str) -> str:
    return "OK" if original.strip() == cleaned.strip() else "REVERT"

def classify_quality(text_raw: str, text_final: str) -> str:
    user = f"[RAW]\n{text_raw}\n\n[FINAL]\n{text_final}"
    response = client.chat.completions.create(
        model=Config.OPENAI_MODEL_MINI,
        messages=[
            {"role": "system", "content": QUALITY_PROMPT_SYSTEM},
            {"role": "user", "content": user}
        ],
        temperature=0.0
    )
    ans = response.choices[0].message.content.strip().upper()
    if "HIGH" in ans:
        return "HIGH"
    if "LOW" in ans:
        return "LOW"
    return "MEDIUM"
