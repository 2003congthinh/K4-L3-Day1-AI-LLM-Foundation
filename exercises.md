# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> *Câu trả lời của bạn*

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
Nên đặt temperature ở mức rất thấp, khoảng 0.0 đến 0.2. Do chatbot cần sự chính xác và có thể dự đoán được (deterministic). Nếu đặt temperature cao (ví dụ 0.7 - 1.0), mô hình sẽ có rủi ro bị hallucinate, sai thông tin, giá cả sản phẩm.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
GPT-4o đắt hơn GPT-4o-mini khoảng 16.67 lần ($0.010 / $0.0006) đối với token đầu ra. (Với 10.500.000 token đầu ra mỗi ngày, GPT-4o tốn ~$105/ngày, trong khi mini chỉ tốn ~$6.3/ngày).
- Trường hợp GPT-4o xứng đáng: Các tác vụ đòi hỏi suy luận logic phức tạp, phân tích dữ liệu sâu, viết code, hoặc xử lý các khiếu nại khách hàng dài và rắc rối cần sự tinh tế, thấu cảm trong ngôn ngữ.
- Trường hợp nên dùng Mini: Các tác vụ phân loại ý định (intent routing), tóm tắt văn bản nhanh, trích xuất dữ liệu (như lấy email/số điện thoại từ văn bản), hoặc chatbot FAQ cơ bản nơi tốc độ và chi phí là ưu tiên hàng đầu.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
- Sự khác biệt: Phản hồi của "giáo viên tiểu học" sẽ rất ngắn gọn, sử dụng từ vựng đời thường, giọng điệu vui nhộn và dùng các ví dụ trực quan (ví dụ: ví blockchain như cuốn sổ nhật ký chung của lớp mà không ai được xé trang hay sửa chữ). Ngược lại, phản hồi của "chuyên gia tài chính" sẽ dài hơn, cấu trúc rành mạch và chứa đậm đặc thuật ngữ kỹ thuật như "sổ cái phân tán", "cơ chế đồng thuận", hay "mã hóa phi tập trung".
- Cách system prompt ảnh hưởng: System prompt hoạt động như một "hệ quy chiếu" mạnh mẽ, thiết lập trước vai trò, ngữ cảnh và mức độ chuyên môn, buộc mô hình phải tự động điều chỉnh toàn bộ kho kiến thức khổng lồ của nó để xuất ra văn bản phù hợp chính xác với tệp khán giả mục tiêu đã định.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
- So sánh độ chênh lệch: Nếu lấy 100 từ tiếng Việt, công thức ước lượng thô 100 / 0.75 sẽ cho ra khoảng 133 token. Tuy nhiên, nếu dùng tiktoken đếm thực tế, đoạn văn đó thường tiêu tốn khoảng 200 đến 250 token (chênh lệch thực tế cao hơn từ 50% đến 80% so với công thức ước lượng).
- Tiếng Việt tốn nhiều token hơn do các bộ mã hóa (tokenizer) của OpenAI được huấn luyện chủ yếu trên dữ liệu tiếng Anh, nơi một từ thường khớp trọn vẹn với 1 token. Tiếng Việt là ngôn ngữ đơn lập (phân tách âm tiết bằng khoảng trắng, một từ có thể gồm 2-3 âm tiết) và sử dụng nhiều dấu thanh phức tạp (á, ế, ợ, ũ...). Vì các tổ hợp ký tự có dấu này ít xuất hiện trong tập dữ liệu huấn luyện gốc, tokenizer không nhận diện được chúng như một khối duy nhất, buộc phải "cắt vụn" một chữ ra thành nhiều sub-word (từ phụ) hoặc thậm chí mã hóa theo từng byte, làm đội chi phí token lên rất cao.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
Streaming quan trọng nhất trong các tình huống tương tác trực tiếp với con người (Real-time UI), ví dụ như chatbot hoặc trợ lý ảo, nơi người dùng sẽ cảm thấy sốt ruột nếu phải nhìn biểu tượng loading nhiều giây; streaming giúp giảm độ trễ cảm nhận vì họ có thể đọc ngay những từ đầu tiên. Ngược lại, non-streaming lại phù hợp hơn cho các tác vụ chạy ngầm như tóm tắt hàng nghìn bài báo, trích xuất dữ liệu JSON lưu vào database, hoặc quy trình tự động hóa nơi không có con người ngồi chờ kết quả tức thì và hệ thống chỉ cần toàn bộ dữ liệu cuối cùng để chuyển sang bước logic tiếp theo.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
- So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi thế gì khi API bị quá tải?: Exponential backoff (giãn cách thời gian thử lại theo cấp số nhân: 1s, 2s, 4s, 8s...) giúp giảm dần áp lực lên máy chủ, cho phép hệ thống API có đủ không gian và thời gian để phục hồi khỏi trạng thái quá tải thay vì liên tục bị request.
- Điều gì xảy ra nếu hàng nghìn client cùng retry với delay cố định giống nhau?: Nếu hàng nghìn client cùng bị lỗi và cùng chờ đúng 1 giây để thử lại, chúng sẽ xảy ra hiện tượng máy chủ vừa mới ngóc đầu lên lập tức bị hàng nghìn request đánh sập thêm lần nữa cùng một lúc, dẫn đến vòng lặp lỗi (503/429) kéo dài không hồi kết. Exponential backoff làm phân tán các request này ra nhiều thời điểm khác nhau.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> *Câu trả lời của bạn*

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
- Hạn chế lớn nhất: Trợ lý đang mắc "chứng mất trí nhớ ngắn hạn". Việc cắt cứng lịch sử chỉ giữ lại 3 lượt hội thoại gần nhất (history = history[-6:]) khiến mô hình mất hoàn toàn bối cảnh nếu người dùng hỏi lại một thông tin họ đã cung cấp ở lượt thứ 1.
- Đề xuất cải thiện: Triển khai Summarization Memory kết hợp với lịch sử ngắn hạn.
- Cách triển khai ngắn gọn: Thay vì xóa vĩnh viễn các tin nhắn cũ khi mảng history đạt giới hạn 6 tin nhắn, ta tách 2 tin nhắn cũ nhất ra và gửi cho một model nhỏ (như gpt-4o-mini chạy ngầm) với prompt: "Hãy tóm tắt ngắn gọn thông tin quan trọng trong đoạn hội thoại sau". Chuỗi tóm tắt trả về sẽ được lưu vào một biến summary_memory và được tự động cộng dồn vào system prompt (ví dụ: [System prompt] + [Bối cảnh trước đó: {summary_memory}]). Bằng cách này, bot vẫn nhớ được toàn bộ diễn biến cuộc trò chuyện cốt lõi từ đầu đến cuối mà token không bị phình to.

---

## Danh Sách Kiểm Tra Nộp Bài

- [X] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [X] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [X] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
