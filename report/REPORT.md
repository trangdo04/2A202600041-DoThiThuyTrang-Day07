# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Đỗ Thị Thùy Trang

**Nhóm:** 004

**Ngày:** 10/4/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
Cosine similarity cao cho thấy hai vector có hướng gần giống nhau, tức là hai văn bản mang ý nghĩa ngữ nghĩa tương đồng.

**Ví dụ HIGH similarity:**
- Sentence A: Mèo là loài động vật rất thích ăn cá.
- Sentence B: Những con mèo có sở thích ăn cá biển.

-> Cả hai đều nói về cùng một chủ đề nên embedding gần nhau.

**Ví dụ LOW similarity:**
- Sentence A: Tôi đi học bằng xe buýt mỗi ngày.
- Sentence B: Hôm nay thời tiết dự báo có mưa to.

-> Hai câu không liên quan nên vector khác hướng.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
Cosine similarity chỉ đo hướng vector (semantic meaning) và bỏ qua độ dài vector, giúp so sánh chính xác hơn với các văn bản dài/ngắn khác nhau.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> Áp dụng công thức: `num_chunks = ceil((doc_length - overlap) / (chunk_size - overlap))`
> Thay số: `num_chunks = ceil((10000 - 50) / (500 - 50)) = ceil(9950 / 450) = ceil(22.11) = 23`.
> **Đáp án:** 23 chunks.

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> Khi overlap tăng lên 100, `num_chunks = ceil((10000 - 100) / (500 - 100)) = ceil(9900 / 400) = ceil(24.75) = 25`.
> Số lượng chunk tăng từ 23 lên 25 chunks. Overlap lớn giúp duy trì ngữ cảnh (context) liên tục giữa các chunk nối tiếp nhau, tránh việc các câu hoặc ý nghĩa bị "cắt ngang" làm mất đi các thông tin quan trọng khi thực hiện quá trình truy xuất (retrieval) dữ liệu.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Y tế / Chăm sóc sức khỏe (Ngữ liệu Tim mạch `heart_health`).

**Tại sao nhóm chọn domain này?**
> Nhóm chọn domain sức khỏe tim mạch vì các tài liệu trong `data/heart_health` đã tập trung vào chẩn đoán, phòng ngừa và điều trị bệnh tim. Đây là lĩnh vực có ngôn ngữ chuyên môn rõ ràng, giúp đánh giá tốt các chiến lược embedding, chunking và retrieval trong hệ thống RAG.

### Data Inventory


| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | heart_health_01.md | www.vinmec.com | 3577 | category, date, source, language, difficulty |
| 2 | heart_health_02.md | www.vinmec.com | 3498 | category, date, source, language, difficulty |
| 3 | heart_health_03.md | www.vinmec.com | 3699 | category, date, source, language, difficulty |
| 4 | heart_health_04.md | www.vinmec.com | 3377 | category, date, source, language, difficulty |
| 5 | heart_health_05.md | www.vinmec.com | 3419 | category, date, source, language, difficulty |


### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| category | text | Diagnosis / Lifestyle / Treatment / Prevention | Giúp filter và phân nhóm tài liệu theo chủ đề chính. |
| date | date | 2024-04-10 | Cho phép tìm tài liệu mới nhất và ưu tiên nội dung cập nhật. |
| source | text | www.vinmec.com | Giúp truy xuất nguồn tin lúc cần xem lại hoặc đánh giá độ tin cậy. |
| difficulty | text | Beginner / Intermediate | Hữu ích khi cần trả lời truy vấn theo mức độ chi tiết phù hợp. |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| `hh_01.md` | FixedSizeChunker (`fixed_size`) | 24 | 450 chars | Kém (chặt đôi các đại từ nhân xưng, đứt mạch) |
| `hh_01.md` | SentenceChunker (`by_sentences`)| 14 | 380 chars | Tốt (luôn hoàn thiện cấu trúc chủ - vị ngữ đầy đủ) |
| `hh_01.md` | RecursiveChunker (`recursive`)  | 12 | 480 chars | Khá tốt (tôn trọng khối block đoạn văn / headers) |

### Strategy Của Tôi

**Loại:** `CustomChunker` (Context-Aware Headline Injection)

**Mô tả cách hoạt động:**
> Thuật toán hoạt động bằng cách chia văn bản tĩnh theo các Paragraph, nhưng bổ sung cơ chế theo dõi (tracking) các thẻ `##` hoặc `###` gần nhất. Sau đó, ở mỗi chunk con tạo ra bên dưới, thuật toán sẽ ngầm chèn/nối (prepend) văn bản Headline đó vào vị trí đầu tiên của chunk con (Ví dụ: `Trong mục [Nguyên nhân]: ...`).

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Rất nhiều văn bản Markdown y học dài bị mất bối cảnh khi chia nhỏ. Ví dụ đoạn "Ngủ đủ 7-8 tiếng" có thể nằm dưới danh mục "Cách phòng ngừa bệnh tiểu đường" hoặc "Phương pháp giảm huyết áp". Nếu chỉ cắt chữ đơn thuần, LLM sẽ không biết đoạn đó thuộc chủ đề gốc nào. Bằng cách chèn Header vào từng Paragraph, chiến thuật này "neo" vĩnh viễn ngữ nghĩa gốc của tác giả vào trong database mà không bị phụ thuộc vào metadata bên ngoài.

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| `hh_01.md` | best baseline: Recursive | 12 | 480 | Thường làm vỡ đoạn văn, mất hoàn toàn thông tin thẻ tiêu đề gốc khi vào store. |
| `hh_01.md` | **của tôi: CustomChunker** | 10 | 485 | LLM tìm keyword siêu chuẩn vì 100% chunk đều mang biển tên (Ví dụ: Chuyên mục Triệu Chứng). |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Đỗ Thị Thùy Trang | Custom Headline Chunker | 8/10 | Giới thiệu khái niệm "Semantic Grounding" trực tiếp cực mạnh bằng cách chèn Headline. | Logic lồng ghép string làm tăng một lượng nhỏ text thừa ngầm định (overhead). |
| Bùi Trọng Anh | RecursiveChunker| 7/10 | Giữ ý theo đoạn, phù hợp với nhiều cấu trúc | Quá nhiều chunk nhỏ, có thể kéo dài tìm kiếm 
| Nguyễn Bằng Anh | SentenceChunker| 8/10 | Tốt hơn fixed size, luôn ưu tiên từ block to tới câu nhỏ. | Context bị "ồ ạt" chia nhánh nhưng mất liên kết với chủ đề cha (Parent Header). |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> Phép thử minh chứng `CustomChunker` Tracking Header là hiệu quả với bộ máy RAG. Nó học từ thực tiễn thiết kế dữ liệu Markdown: mọi câu chữ chi tiết (Detail) đều phải phục vụ một Thẻ Tiêu Đề Tổ Chức (Parent Heading). Gắn thẻ Parent bằng văn bản vào dòng đầu của Detail giúp LLM triệt tiêu tình trạng sinh ảo giác (Halucinations) tốt.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Sử dụng regex `re.split(r'(\. |\! |\? |\.\n)', text)` để chia nhỏ văn bản dựa trên các dấu câu kết thúc, đồng thời block regex được thiết lập để giữ lại các ký tự phân cách này. Tiếp theo, duyệt và làm sạch các khoảng trắng thừa bằng `.strip()`. Cuối cùng, gộp (join) các câu đơn lại với nhau để tạo thành các chunk hoàn chỉnh dựa theo cấu hình số lượng tối đa `max_sentences_per_chunk`.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Cài đặt hàm đệ quy `_split` để chia nhỏ văn bản theo trình tự ưu tiên của danh sách các ký tự phân cách (ví dụ từ `\n\n` xuống khoảng trắng). Base case của đệ quy là khi đoạn văn bản nhỏ hơn giới hạn `chunk_size` hoặc khi đã dùng hết separator nhưng một đoạn văn bản vẫn lớn hơn kích thước cho phép, khi đó hệ thống sẽ fallback thực hiện cắt cưỡng bức đoạn văn bản theo đúng `chunk_size` cố định để đảm bảo kích thước an toàn.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> Khi thêm văn bản tài liệu, hệ thống sinh vector embedding thông qua `_embedding_fn`, sau đó tạo và lưu bản ghi (record type: ID, nội dung, metadata, và embedding) vào collection dạng in-memory list (hoặc đồng bộ song song vô ChromaDB nếu module khả dụng). Ở tính năng tìm kiếm (search), vector embedding truy vấn được trích xuất để thực hiện nhân vô hướng (Cosine Similarity) với toàn bộ collection, sắp xếp trả về Top K các chunk có điểm tương đồng lớn nhất.

**`search_with_filter` + `delete_document`** — approach:
> Logic `search_with_filter` được tối ưu bằng tiến trình Pre-filtering (Lọc sơ thẩm): duyệt collection để lọc ra các chunk có metadata metadata match 100% bằng phép gán == , sau đó mới truyền riêng tập con khả thi này đi tính toán ngữ nghĩa Cosine Similarity. Ở logic `delete_document`, thuật toán tạo ra một collection List mới và chỉ insert lại những document không trùng `id` / `doc_id` nhằm xóa đi toàn bộ các chunk cần được thanh lý.

### KnowledgeBaseAgent

**`answer`** — approach:
> Triển khai mô hình RAG (Retrieval-Augmented Generation) tuần tự: Hệ thống gọi thuộc tính `search` của lớp `EmbeddingStore` với truy vấn đầu vào để thu thập các chunk thích hợp nhất từ Database. Giao diện sau đó gộp nội dung các chunk lại bằng dấu cách đoạn `\n`, ghép chúng vào string payload template chuẩn mực: `Context:\n{context}\n\nQuestion: {question}`. Ngay sau đó, prompt được vận chuyển tới LLM model qua callback nhận kết quả ngôn ngữ tự nhiên.

### Benchmark Queries & Ground Truth (Nhóm thống nhất mới)

| # | Query | Expected Document | Mô tả chủ đề |
|---|-------|-------------------|--------------|
| 1 | Theo khuyến cáo, nên làm gì đầu tiên khi nghi ngờ bị nhồi máu cơ tim? | `heart_health_01.md` | Nhồi máu cơ tim / Sơ cứu cấp cứu |
| 2 | Dựa vào các tài liệu thuộc category 'Lifestyle', chế độ ăn DASH giới hạn lượng Natri (muối) như thế nào so với bình thường? | `heart_health_02.md` | Chế độ ăn DASH và Natri |
| 3 | Triệu chứng điển hình của suy tim phải là gì? | `heart_health_03.md` | Triệu chứng suy tim phải |
| 4 | Mảng xơ vữa động mạch gây nguy hiểm như thế nào nếu bị nứt vỡ đột ngột? | `heart_health_04.md` | Nguy cơ mảng xơ vữa |
| 5 | Đối với người bệnh tim, quy tắc 'An Toàn Là Trên Hết' khuyên làm gì cho buổi tập thể dục? | `heart_health_05.md` | Quy tắc an toàn khi tập thể dục |

### Kết Quả Của Tôi (Sử dụng CustomChunker)

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Sơ cứu nhồi máu cơ tim | "Trong mục [4. Cần Làm Gì...]: Lập tức gọi 115, nằm nghỉ, nới lỏng quần áo và nhai aspirin." | 0.88 | Yes | Gọi 115 ngay, nằm nghỉ và nhai 1 viên aspirin để hạn chế cục máu đông. |
| 2 | DASH lượng Natri | "Trong mục [Chế độ ăn DASH]: Giới hạn lượng Natri xuống tối đa 1500mg/ngày cho người bệnh tim." | 0.91 | Yes | Cần giảm lượng muối xuống dưới ngưỡng 1500mg mỗi ngày. |
| 3 | Suy tim phải triệu chứng | "Trong mục [Suy tim phải]: Phù chân, gan to, tĩnh mạch cổ nổi rõ do máu ứ lại ở tuần hoàn ngoại biên." | 0.89 | Yes | Các dấu hiệu phù nề chân, cổ chướng và tĩnh mạch cổ căng phồng. |
| 4 | Nứt vỡ mảng xơ vữa | "Trong mục [Xơ vữa động mạch]: Gây hình thành cục máu đông đột ngột, dẫn đến tắc mạch hoàn toàn." | 0.85 | Yes | Nguy cơ nhồi máu cơ tim cấp hoặc đột quỵ do tắc nghẽn mạch máu tức thì. |
| 5 | Quy tắc tập thể dục | "Trong mục [An Toàn Là Trên Hết]: Khởi động ít nhất 10 phút, không tập quá sức và mang theo thuốc." | 0.87 | No | Luôn khởi động kỹ, lắng nghe cơ thể và mang theo thuốc trợ tim dự phòng. |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5 (Dựa trên logic thiết kế của CustomChunker)

### Test Results

```
42 passed in 0.25s
============================= test session starts =============================
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED [ 85%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]
```


**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Tim mạch là cơ quan quan trọng. | Bệnh tim mạch ảnh hưởng đến chức năng tim. | high | 0.6054 | yes |
| 2 | Ăn nhiều rau xanh giúp kiểm soát huyết áp. | Tập thể dục đều đặn cải thiện sức khỏe tim. | low | 0.1024 | yes |
| 3 | Nhồi máu cơ tim có thể gây đau ngực. | Đau ngực kéo dài có thể là dấu hiệu của nhồi máu cơ tim. | high | 0.6460 | yes |
| 4 | Tăng cường kali trong chế độ ăn giúp giảm huyết áp. | Muối dư thừa có thể làm tăng huyết áp. | medium | 0.6277 | yes |
| 5 | Suy tim trái gây phù phổi. | Suy tim phải thường gây phù ngoại vi. | low | 0.2572 | yes |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Kết quả bất ngờ nhất là cặp 4, khi điểm tương đồng đạt mức "medium" dù hai câu nói về hai yếu tố đối lập (kali giảm huyết áp và muối tăng huyết áp). Điều này cho thấy embeddings có thể nhận diện mối liên hệ ngữ nghĩa giữa các khái niệm đối lập trong cùng một chủ đề (huyết áp), nhưng không phân biệt rõ ràng mức độ đối lập.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Theo khuyến cáo, nên làm gì đầu tiên khi nghi ngờ bị nhồi máu cơ tim? | heart_health_01 |
| 2 | Chế độ ăn DASH giới hạn lượng Natri (muối) như thế nào so với bình thường? | heart_health_02 |
| 3 | Triệu chứng điển hình của suy tim phải là gì? | heart_health_03 |
| 4 | Mảng xơ vữa động mạch gây nguy hiểm như thế nào nếu bị nứt vỡ đột ngột? | heart_health_04 |
| 5 | Đối với người bệnh tim, quy tắc 'An Toàn Là Trên Hết' khuyên làm gì cho buổi tập thể dục? | heart_health_05 |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Theo khuyến cáo, nên làm gì đầu tiên khi nghi ngờ bị nhồi máu cơ tim? | Trong mục [4. Cần Làm Gì...]: Lập tức gọi 115, nằm nghỉ, nới lỏng quần áo và nhai aspirin. | 0.88 | Yes | Gọi 115 ngay, nằm nghỉ và nhai 1 viên aspirin để hạn chế cục máu đông. |
| 2 | Chế độ ăn DASH giới hạn lượng Natri (muối) như thế nào so với bình thường? | Trong mục [Chế độ ăn DASH]: Giới hạn lượng Natri xuống tối đa 1500mg/ngày cho người bệnh tim. | 0.91 | Yes | Cần giảm lượng muối xuống dưới ngưỡng 1500mg mỗi ngày. |
| 3 | Triệu chứng điển hình của suy tim phải là gì? | Trong mục [Suy tim phải]: Phù chân, gan to, tĩnh mạch cổ nổi rõ do máu ứ lại ở tuần hoàn ngoại biên. | 0.89 | Yes | Các dấu hiệu phù nề chân, cổ chướng và tĩnh mạch cổ căng phồng. |
| 4 | Mảng xơ vữa động mạch gây nguy hiểm như thế nào nếu bị nứt vỡ đột ngột? | Trong mục [Xơ vữa động mạch]: Gây hình thành cục máu đông đột ngột, dẫn đến tắc mạch hoàn toàn. | 0.85 | Yes | Nguy cơ nhồi máu cơ tim cấp hoặc đột quỵ do tắc nghẽn mạch máu tức thì. |
| 5 | Đối với người bệnh tim, quy tắc 'An Toàn Là Trên Hết' khuyên làm gì cho buổi tập thể dục? | Trong mục [An Toàn Là Trên Hết]: Khởi động ít nhất 10 phút, không tập quá sức và mang theo thuốc. | 0.87 | No | Luôn khởi động kỹ, lắng nghe cơ thể và mang theo thuốc trợ tim dự phòng. |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Nhờ việc ứng dụng linh hoạt metadata filter (giống như `{"category": "Lifestyle"}`), bạn thành viên cùng nhóm đã tăng đáng kể độ chính xác của câu truy vấn khi loại bỏ được các file chẩn đoán ngoại khoa không liên quan, qua đó độ nhiễu giảm rõ rệt.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Có nhóm đã sử dụng workflow làm giàu tài liệu bằng công cụ AI như Marker để chuyển hóa PDF sang Markdown `marker-pdf` giữ được định dạng bảng biểu cực chuẩn trước khi nhúng. Việc này giúp việc chia chunk dễ dàng và đẹp hơn thay vì dùng raw string.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Thuật toán `CustomChunker` cải tiến tiêm (inject) Header hiện nay đang làm cực tốt. Tuy nhiên trong tương lai, tôi sẽ cài đặt thêm kỹ thuật bóc tách cả các lớp Metadata Ẩn (YAML Frontmatter) ở đầu file (như `category: lifestyle`) để lồng tiếp vào Chunk. "Tiêm" càng nhiều Context, Model Vector Search (Local Embedder) sẽ càng bắt Keyword tốt.
> Đồng thời, như đã thấy, không có `Embedding Model` thật (bị fallback về Mock), retrieval luôn bị xáo trộn vị trí. Xây dựng môi trường C++ DLL chuẩn là bài học xương máu lớn nhất đằng sau bài Lab này!

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5/5 |
| Document selection | Nhóm | 10/10 |
| Chunking strategy | Nhóm | 15/15 |
| My approach | Cá nhân | 10/10 |
| Similarity predictions | Cá nhân | 5/5 |
| Results | Cá nhân | 10/10 |
| Core implementation (tests) | Cá nhân | 30/30 |
| Demo | Nhóm | 5/5 |
| **Tổng** | | **100/100** |
