from utils.common import run_cmd
from utils.gpt_utils import call_gpt

FILTER_SYSTEM_PROMPT = """
Bạn là bộ lọc nội dung video cho dự án dữ liệu thời tiết, hiện tượng tự nhiên, thiên tai.
Nhiệm vụ: Dựa vào TIÊU ĐỀ video YouTube, xác định xem video này có liên quan đến:
1. Dự báo thời tiết (Weather forecast).
2. Thiên tai, bão lũ, cảnh báo thời tiết nguy hiểm tại địa phương.
3. Các hiện tượng khí tượng thủy văn, hiện tượng tự nhiên của trái đất.

Trả về "TRUE" nếu video LIÊN QUAN.
Trả về "FALSE" nếu video KHÔNG LIÊN QUAN (ví dụ: ca nhạc, phim ảnh, tin tức chính trị không liên quan, quảng cáo...).

Chỉ trả lời duy nhất 1 từ: TRUE hoặc FALSE.
"""

def get_video_title(url: str) -> str:
    """Lấy tiêu đề video YouTube mà không cần download video."""
    cmd = [
        "yt-dlp",
        "--get-title",
        "--skip-download",
        "--no-playlist",
        url
    ]
    try:
        res = run_cmd(cmd)
        return res.stdout.strip()
    except Exception as e:
        print(f"Error getting title for URL {url}: {e}")
        return ""

def is_weather_related(url: str) -> bool:
    title = get_video_title(url)
    if not title:
        print(f"Không lấy được tiêu đề cho {url}. Bỏ qua an toàn (return False).")
        return False
    
    print(f"Checking Title: {title}")
    response = call_gpt(FILTER_SYSTEM_PROMPT, f"Tiêu đề video: {title}")
    
    decision = "TRUE" in response.strip().upper()
    print(f"  => Decision: {'KEEP' if decision else 'SKIP'} ({response})")
    
    return decision
