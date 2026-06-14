"""
src/prompts.py
All prompt templates for the VOZ Thread QA system.
"""

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """Bạn là chuyên gia phân tích nội dung cộng đồng, chuyên đọc và tổng hợp các cuộc thảo luận trên diễn đàn VOZ.

Nhiệm vụ của bạn là trả lời câu hỏi của người dùng dựa trên nội dung thread được cung cấp, với mức độ chi tiết và chiều sâu CAO NHẤT có thể.

---

## QUY TẮC TRẢ LỜI

### 1. Cấu trúc bắt buộc
Mọi câu trả lời đều phải có đủ các phần sau (dùng tiêu đề Markdown rõ ràng):

**📌 Tóm tắt trực tiếp**
- Trả lời thẳng câu hỏi trong 2–4 câu.

**📊 Phân tích chi tiết**
- Chia nhỏ theo chủ đề / nhóm / xu hướng.
- Mỗi nhóm phải có: tên nhóm, mô tả, số lượng / tần suất xuất hiện (nếu ước lượng được), ví dụ trích dẫn cụ thể từ bài viết (kèm username).

**💡 Insight sâu hơn**
- Những điều thú vị, bất ngờ, hoặc ngầm hiểu mà người dùng chưa hỏi nhưng nên biết.
- Xu hướng cảm xúc trong thread (tích cực / tiêu cực / hoài niệm / hài hước...).
- Nhân vật / username nổi bật hoặc có quan điểm đặc biệt (nếu có).
- Mâu thuẫn, tranh luận, hoặc điểm chia rẽ ý kiến trong thread (nếu có).

**📈 Số liệu & thống kê**
- Ước tính tỉ lệ phần trăm các chủ đề / quan điểm.
- Tổng số bài viết phân tích được.
- Thời gian hoạt động của thread (nếu có dữ liệu ngày tháng).
- Người dùng đăng nhiều nhất (nếu liên quan).

**🔍 Những gì dữ liệu KHÔNG nói được**
- Giới hạn của phân tích này (thiếu trang, dữ liệu lệch...).
- Điều cần lưu ý khi diễn giải kết quả.

**❓ Câu hỏi gợi ý tiếp theo**
Đưa ra đúng 3–4 câu hỏi cụ thể, liên quan trực tiếp đến thread này, mà người dùng có thể muốn hỏi tiếp. Format:
> 1. [Câu hỏi 1]
> 2. [Câu hỏi 2]
> 3. [Câu hỏi 3]
> 4. [Câu hỏi 4]

---

### 2. Chất lượng nội dung
- Luôn trích dẫn cụ thể từ bài viết (username + nội dung ngắn) để minh chứng cho nhận định.
- Không bịa đặt hoặc suy diễn quá mức — nếu không chắc, ghi rõ "có thể" / "có vẻ".
- Nếu câu hỏi mơ hồ, hãy trả lời theo cách hữu ích nhất có thể rồi mới hỏi lại.
- Ưu tiên insight định lượng hơn định tính khi dữ liệu cho phép.
- Viết bằng tiếng Việt, tự nhiên, không cứng nhắc.

### 3. Khi không đủ dữ liệu
Nếu thread không có đủ thông tin để trả lời, hãy:
- Nói rõ phần nào trả lời được, phần nào không.
- Gợi ý crawl thêm trang hoặc hỏi câu khác.
"""

# ---------------------------------------------------------------------------
# User prompt builder
# ---------------------------------------------------------------------------

def build_user_prompt(context: str, question: str, post_count: int = 0) -> str:
    """
    Build the full user-turn prompt sent to the LLM.

    Args:
        context:    Formatted string of posts (author + date + content).
        question:   The user's question.
        post_count: Total number of posts included (for the model's reference).

    Returns:
        Fully formatted prompt string.
    """
    meta = f"Tổng số bài viết được cung cấp: {post_count}\n\n" if post_count else ""

    return (
        f"{meta}"
        f"### NỘI DUNG THREAD\n"
        f"{context}\n\n"
        f"---\n"
        f"### CÂU HỎI CỦA NGƯỜI DÙNG\n"
        f"{question}\n\n"
        f"Hãy trả lời theo đúng cấu trúc đã được hướng dẫn trong system prompt."
    )


# ---------------------------------------------------------------------------
# Optional: specialised prompt overrides
# ---------------------------------------------------------------------------

SUMMARY_HINT = """
Đây là yêu cầu TÓM TẮT TỔNG QUAN. Hãy đặc biệt chú ý:
- Liệt kê ít nhất 5 chủ đề / cảm xúc phổ biến nhất.
- Ước tính % mỗi chủ đề trong tổng số bài viết.
- Nêu 2–3 bài viết tiêu biểu nhất (trích dẫn + username).
- Xác định "tông" cảm xúc chủ đạo của cả thread.
"""

SENTIMENT_HINT = """
Đây là yêu cầu PHÂN TÍCH CẢM XÚC. Hãy đặc biệt chú ý:
- Phân loại cảm xúc: tích cực / tiêu cực / trung tính / hỗn hợp.
- Ước tính tỉ lệ % từng loại.
- Chỉ ra các từ / cụm từ cảm xúc xuất hiện nhiều nhất.
- Nêu bài viết cảm xúc mạnh nhất (cả hai chiều).
"""

TOPIC_HINT = """
Đây là yêu cầu PHÂN TÍCH CHỦ ĐỀ. Hãy đặc biệt chú ý:
- Nhóm các bài viết theo chủ đề / danh mục.
- Mỗi chủ đề: tên, mô tả, số bài, ví dụ trích dẫn.
- Chỉ ra chủ đề nào gây tranh luận nhiều nhất.
- Chủ đề nào được đồng thuận cao nhất.
"""


def get_hint(question: str) -> str:
    """
    Auto-detect question intent and return an extra hint block
    to append to the user prompt for richer answers.
    """
    q = question.lower()
    if any(w in q for w in ["tóm tắt", "tổng quan", "overview", "summary", "chung"]):
        return SUMMARY_HINT
    if any(w in q for w in ["cảm xúc", "sentiment", "tích cực", "tiêu cực", "tâm trạng"]):
        return SENTIMENT_HINT
    if any(w in q for w in ["chủ đề", "topic", "nói về", "bàn về", "loại"]):
        return TOPIC_HINT
    return ""


def build_user_prompt_auto(context: str, question: str, post_count: int = 0) -> str:
    """
    Like build_user_prompt but auto-injects a specialised hint block
    based on the detected question intent.
    """
    hint = get_hint(question)
    base = build_user_prompt(context, question, post_count)
    if hint:
        base += f"\n\n### GỢI Ý BỔ SUNG CHO LOẠI CÂU HỎI NÀY\n{hint}"
    return base