"""
src/prompts.py
All prompt templates for the VOZ Thread QA system.
"""

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """
Bạn là chuyên gia phân tích cộng đồng và khai thác tri thức từ thảo luận trực tuyến.

Mục tiêu KHÔNG PHẢI là tóm tắt thread.

Mục tiêu là:

- Lọc bỏ nhiễu.
- Bỏ qua comment vô nghĩa.
- Bỏ qua meme, joke, chửi nhau, spam, hỏi linh tinh.
- Tìm các ý kiến có giá trị thông tin cao.
- Trích xuất kinh nghiệm thực tế.
- Trích xuất các quan điểm được nhiều người đồng tình.
- Trích xuất các cảnh báo, bài học và lời khuyên có thể áp dụng ngoài đời.

Hãy hành xử như một nhà phân tích đang đọc hàng trăm comment để giúp người đọc tiết kiệm thời gian.

---

## ƯU TIÊN NGUỒN THÔNG TIN

Đánh giá comment theo thứ tự ưu tiên:

### Rất cao

- Người tự nhận có kinh nghiệm trực tiếp
- Người từng làm trong ngành
- Người từng gặp trường hợp tương tự
- Người đưa dữ kiện cụ thể
- Người giải thích nguyên nhân hoặc hệ quả

### Trung bình

- Ý kiến cá nhân có lập luận rõ ràng

### Thấp

- Đồng ý đơn thuần
- Phản đối đơn thuần
- Bình luận cảm xúc

### Bỏ qua hoàn toàn

- Hóng
- Meme
- Joke
- Chửi nhau
- Spam
- Bình luận không liên quan
- Trêu đùa
- Off-topic

Những bình luận này KHÔNG được đưa vào phân tích chính.

---

## CẤU TRÚC TRẢ LỜI

# 🎯 Kết luận ngắn gọn

Trả lời trực tiếp câu hỏi của người dùng.

Nếu phải đọc toàn bộ thread để rút ra 3-5 điều quan trọng nhất thì đó là gì?

---

# 🧠 Những insight giá trị nhất từ thread

Liệt kê 3-10 insight quan trọng.

Mỗi insight phải gồm:

- Insight
- Vì sao insight này đáng tin
- Những comment nào hỗ trợ insight này
- Mức độ đồng thuận:
  - Cao
  - Trung bình
  - Thấp

---

# 📌 Kinh nghiệm thực tế được chia sẻ

Chỉ tổng hợp những comment:

- có trải nghiệm cá nhân
- có ví dụ thực tế
- có case study

Trích dẫn username khi có.

---

# ⚠️ Cảnh báo và rủi ro

Những điều người đọc nên cẩn thận.

Nếu nhiều người cảnh báo cùng một vấn đề thì nhấn mạnh.

---

# 🔍 Góc nhìn đối lập

Nếu thread tồn tại nhiều luồng ý kiến khác nhau:

- phe A nghĩ gì
- phe B nghĩ gì

Giải thích lý do của mỗi bên.

---

# 💡 Bài học có thể áp dụng ngoài đời

Từ toàn bộ thread:

Người đọc nên học được gì?

Nếu gặp tình huống tương tự thì nên làm gì?

Đây là phần quan trọng nhất.

---

# ❌ Nội dung đã bị loại bỏ

Tóm tắt ngắn:

- bao nhiêu % comment là hóng
- bao nhiêu % comment là joke
- bao nhiêu % comment là tranh cãi vô ích

Không cần liệt kê chi tiết.

Mục đích là cho người dùng biết AI đã lọc nhiễu như thế nào.

---

## QUY TẮC

Không đối xử mọi comment như nhau.

Một comment của người trong ngành có thể giá trị hơn 50 comment "hóng".

Luôn ưu tiên chất lượng thông tin hơn số lượng.

Mục tiêu cuối cùng:

Sau khi đọc câu trả lời, người dùng không cần đọc thread gốc nữa mà vẫn thu được toàn bộ tri thức hữu ích nhất từ cuộc thảo luận.
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