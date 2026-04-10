from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text.strip():
            return []
        # split on separators, keeping them if possible
        parts = re.split(r'(\. |\! |\? |\.\n)', text)
        sentences = []
        current = ""
        for part in parts:
            if part in [". ", "! ", "? ", ".\n"]:
                current += part
                sentences.append(current.strip())
                current = ""
            else:
                current += part
        if current.strip():
            sentences.append(current.strip())
        
        chunks = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            group = sentences[i:i+self.max_sentences_per_chunk]
            chunks.append(" ".join(group))
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        return self._split(text, list(self.separators))

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        if len(current_text) <= self.chunk_size:
            return [current_text]
        
        if not remaining_separators:
            chunks = []
            for i in range(0, len(current_text), self.chunk_size):
                chunks.append(current_text[i:i+self.chunk_size])
            return chunks

        sep = remaining_separators[0]
        next_seps = remaining_separators[1:]
        
        if sep == "":
            return self._split(current_text, next_seps)

        splits = current_text.split(sep)
        
        results = []
        current_chunk = ""
        for s in splits:
            part = s if not current_chunk else current_chunk + sep + s
            if len(part) <= self.chunk_size:
                current_chunk = part
            else:
                if current_chunk:
                    results.append(current_chunk)
                if len(s) > self.chunk_size:
                    results.extend(self._split(s, next_seps))
                    current_chunk = ""
                else:
                    current_chunk = s
                    
        if current_chunk:
            results.append(current_chunk)
            
        return results


class ParagraphChunker:
    """
    Custom strategy: Tách dữ liệu y tế theo các đoạn văn bản (Paragraph) 
    dựa trên double-newline (\n\n). Điều này giúp các định nghĩa lớn không bị chặt ngang.
    """
    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        
        pieces = text.split('\n\n')
        chunks = []
        for piece in pieces:
            cleaned = piece.strip()
            if cleaned:
                chunks.append(cleaned)
                
        return chunks

class CustomChunker:
    """
    Cải tiến ưu tiên Headline:
    Tách dữ liệu theo Paragraph nhưng tự động lồng ghép (prepend) 
    Headline mục nhỏ (## hoặc ###) gần nhất vào nội dung của từng Chunk.
    Điều này giải quyết triệt để vấn đề LLM đọc chunk mà không biết nó 
    thuộc chủ đề nào.
    """
    def __init__(self, max_chunk_length: int = 500, min_chunk_length: int = 100):
        self.max_chunk_length = max_chunk_length
        self.min_chunk_length = min_chunk_length

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []

        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        chunks = []
        current_header = "Thông tin chung"

        for p in paragraphs:
            lines = p.split('\n')
            first_line = lines[0].strip()

            # Quét xem block này có chứa Markdown Headline không
            if first_line.startswith('##'):
                # Lưu nhãn header
                current_header = first_line.replace('#', '').replace('*', '').strip()
                chunks.append(p)  # Block chứa thư Mục gốc
            else:
                # Gắn Context Nhãn vào đầu văn bản trơn
                enriched_chunk = f"Trong mục [{current_header}]: {p}"
                if len(enriched_chunk) > self.max_chunk_length:
                    # Nếu chunk quá dài, chia nhỏ hơn
                    sub_chunks = self._split_long_chunk(enriched_chunk)
                    chunks.extend(sub_chunks)
                elif len(enriched_chunk) >= self.min_chunk_length:
                    chunks.append(enriched_chunk)

        return chunks

    def _split_long_chunk(self, chunk: str) -> list[str]:
        """Chia nhỏ chunk dài vượt quá max_chunk_length."""
        words = chunk.split()
        sub_chunks = []
        current_chunk = []
        current_length = 0

        for word in words:
            if current_length + len(word) + 1 > self.max_chunk_length:
                sub_chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
                current_length += len(word) + 1

        if current_chunk:
            sub_chunks.append(' '.join(current_chunk))

        return sub_chunks

    def _merge_small_chunks(self, chunks: list[str]) -> list[str]:
        """Gộp các chunk nhỏ hơn min_chunk_length với chunk liền kề."""
        merged_chunks = []
        buffer = ""

        for chunk in chunks:
            if len(buffer) + len(chunk) + 1 < self.min_chunk_length:
                buffer += " " + chunk
            else:
                if buffer:
                    merged_chunks.append(buffer.strip())
                    buffer = ""
                merged_chunks.append(chunk)

        if buffer:
            merged_chunks.append(buffer.strip())

        return merged_chunks

def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    mag_a = math.sqrt(sum(x*x for x in vec_a))
    mag_b = math.sqrt(sum(x*x for x in vec_b))
    if mag_a < 1e-10 or mag_b < 1e-10:  # Thêm kiểm tra để tránh lỗi chia cho 0
        return 0.0
    return _dot(vec_a, vec_b) / (mag_a * mag_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 500) -> dict:
        fixed = FixedSizeChunker(chunk_size=chunk_size, overlap=50)
        sentence = SentenceChunker(max_sentences_per_chunk=3)
        recursive = RecursiveChunker(chunk_size=chunk_size)
        custom = CustomChunker(max_chunk_length=chunk_size)

        c_fixed = fixed.chunk(text)
        c_sentence = sentence.chunk(text)
        c_recursive = recursive.chunk(text)
        c_custom = custom.chunk(text)

        def _stats(chunks):
            count = len(chunks)
            avg = sum(len(c) for c in chunks)/count if count else 0
            return {"count": count, "avg_length": avg, "chunks": chunks}

        results = {
            "fixed_size": _stats(c_fixed),
            "by_sentences": _stats(c_sentence),
            "recursive": _stats(c_recursive),
            "custom_chunker": _stats(c_custom)
        }

        # Log thêm thông tin để đánh giá
        for strategy, stats in results.items():
            print(f"Strategy: {strategy}")
            print(f"  - Chunk count: {stats['count']}")
            print(f"  - Avg length: {stats['avg_length']:.2f}")

        return results
