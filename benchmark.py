from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

from dotenv import load_dotenv

from src import (
    Document,
    EmbeddingStore,
    KnowledgeBaseAgent,
    LocalEmbedder,
    OpenAIEmbedder,
    _mock_embed,
    EMBEDDING_PROVIDER_ENV,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
)
from src.chunking import CustomChunker

DEFAULT_HEART_HEALTH_GLOB = "data/heart_health/*.md"
DEFAULT_CHUNK_SIZE = 900

BENCHMARK_QUERIES = [
    {
        "query": "Theo khuyến cáo, nên làm gì đầu tiên khi nghi ngờ bị nhồi máu cơ tim?",
        "expected_doc": "heart_health_01",
        "description": "Nhồi máu cơ tim / cấp cứu",
        "metadata_filter": {"category": "Diagnosis"},
    },
    {
        "query": "Chế độ ăn DASH giới hạn lượng Natri (muối) như thế nào so với bình thường?",
        "expected_doc": "heart_health_02",
        "description": "Chế độ ăn DASH và natri",
        "metadata_filter": {"category": "Lifestyle"},
    },
    {
        "query": "Triệu chứng điển hình của suy tim phải là gì?",
        "expected_doc": "heart_health_03",
        "description": "Triệu chứng suy tim phải",
        "metadata_filter": {"category": "Treatment"},
    },
    {
        "query": "Mảng xơ vữa động mạch gây nguy hiểm như thế nào nếu bị nứt vỡ đột ngột?",
        "expected_doc": "heart_health_04",
        "description": "Nguy cơ mảng xơ vữa",
        "metadata_filter": {"category": "Prevention"},
    },
    {
        "query": "Đối với người bệnh tim, quy tắc 'An Toàn Là Trên Hết' khuyên làm gì cho buổi tập thể dục?",
        "expected_doc": "heart_health_05",
        "description": "Quy tắc an toàn khi tập thể dục",
        "metadata_filter": {"category": "Lifestyle"},
    },
]


def get_embedder(provider: str):
    provider = provider.strip().lower()
    if provider == "local":
        try:
            return LocalEmbedder(model_name=os.getenv(LOCAL_EMBEDDING_MODEL, LOCAL_EMBEDDING_MODEL))
        except Exception as exc:
            print(f"Warning: local embedder failed, falling back to mock: {exc}")
            return _mock_embed
    if provider == "openai":
        try:
            return OpenAIEmbedder(model_name=os.getenv(OPENAI_EMBEDDING_MODEL, OPENAI_EMBEDDING_MODEL))
        except Exception as exc:
            print(f"Warning: OpenAI embedder failed, falling back to mock: {exc}")
            return _mock_embed
    return _mock_embed


def parse_front_matter(content: str) -> tuple[dict[str, str], str]:
    import re

    front_matter = {}
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, flags=re.DOTALL)
    if not match:
        return front_matter, content

    raw_yaml = match.group(1)
    body = content[match.end():]
    for line in raw_yaml.splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        front_matter[key.strip()] = value.strip()
    return front_matter, body


def load_documents_from_files(file_paths: list[str]) -> list[Document]:
    allowed_extensions = {".md", ".txt"}
    documents: list[Document] = []
    for raw_path in file_paths:
        path = Path(raw_path)
        if path.suffix.lower() not in allowed_extensions:
            print(f"Skipping unsupported file type: {path}")
            continue
        if not path.exists() or not path.is_file():
            print(f"Skipping missing file: {path}")
            continue
        content = path.read_text(encoding="utf-8")
        front_matter, body = parse_front_matter(content)
        metadata = {"source": str(path), "doc_id": path.stem}
        metadata.update(front_matter)
        documents.append(
            Document(
                id=path.stem,
                content=body,
                metadata=metadata,
            )
        )
    return documents


def chunk_documents(documents: list[Document], chunker) -> list[Document]:
    chunked_docs: list[Document] = []
    for doc in documents:
        chunks = chunker.chunk(doc.content)
        for index, chunk in enumerate(chunks, start=1):
            metadata = dict(doc.metadata)
            metadata.update(
                {
                    "source": doc.metadata.get("source"),
                    "doc_id": doc.id,
                    "chunk_index": index,
                }
            )
            chunked_docs.append(
                Document(
                    id=f"{doc.id}-{index}",
                    content=chunk.strip(),
                    metadata=metadata,
                )
            )
    return chunked_docs


def demo_llm(prompt: str) -> str:
    preview = prompt[:400].replace(chr(13), "").replace(chr(10), " ")
    return f"[DEMO LLM] Generated answer from prompt preview: {preview}..."


def print_chunk_stats(chunked_docs: list[Document]) -> None:
    lengths = [len(doc.content) for doc in chunked_docs]
    print(f"Chunked {len(chunked_docs)} chunks")
    print(f"  - min length: {min(lengths) if lengths else 0}")
    print(f"  - max length: {max(lengths) if lengths else 0}")
    print(f"  - avg length: {sum(lengths) / max(1, len(lengths)):.1f}")


def run_benchmark(
    file_paths: list[str],
    top_k: int,
    embedder_provider: str,
    chunk_size: int,
    show_agent: bool,
) -> int:
    load_dotenv(override=False)
    embedder = get_embedder(embedder_provider)
    print(f"Embedding provider: {getattr(embedder, '_backend_name', embedder.__class__.__name__)}")

    documents = load_documents_from_files(file_paths)
    if not documents:
        print("No documents were loaded. Check your input file patterns.")
        return 1

    chunker = CustomChunker()
    chunked_docs = chunk_documents(documents, chunker)
    print_chunk_stats(chunked_docs)

    store = EmbeddingStore(collection_name="heart_health_benchmark", embedding_fn=embedder)
    store.add_documents(chunked_docs)
    print(f"Stored {store.get_collection_size()} chunks in EmbeddingStore")
    print("\n=== Benchmark Queries ===")

    total_expected_hits = 0
    for query_info in BENCHMARK_QUERIES:
        query = query_info["query"]
        expected_doc = query_info["expected_doc"]
        metadata_filter = query_info.get("metadata_filter")
        print(f"\nQuery: {query}")
        if metadata_filter:
            print(f"  Using metadata filter: {metadata_filter}")
            results = store.search_with_filter(query, top_k=top_k, metadata_filter=metadata_filter)
        else:
            results = store.search(query, top_k=top_k)
        expected_hit = any(r["metadata"].get("doc_id") == expected_doc for r in results)
        total_expected_hits += int(expected_hit)
        print(f"Expected source doc: {expected_doc}")
        print(f"Hit expected doc in top-{top_k}: {'YES' if expected_hit else 'NO'}")
        for rank, result in enumerate(results, start=1):
            source = result["metadata"].get("doc_id")
            content_preview = result["content"][:180].replace(chr(10), " ").replace(chr(13), " ")
            print(f"  {rank}. score={result['score']:.4f} source={source} chunk_index={result['metadata'].get('chunk_index')}\n     {content_preview}...")

        if show_agent:
            agent = KnowledgeBaseAgent(store=store, llm_fn=demo_llm)
            print("  Agent answer:")
            print(f"    {agent.answer(query, top_k=top_k)}")

    print(f"\nSummary: {total_expected_hits}/{len(BENCHMARK_QUERIES)} queries hit expected document in top-{top_k}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run benchmark queries against the heart_health retrieval pipeline.")
    parser.add_argument(
        "--files",
        nargs="+",
        default=glob.glob(DEFAULT_HEART_HEALTH_GLOB),
        help="Input Markdown files to load for benchmark.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top results to retrieve per query.",
    )
    parser.add_argument(
        "--embedder",
        choices=["mock", "local", "openai"],
        default=os.getenv(EMBEDDING_PROVIDER_ENV, "local"),
        help="Embedding backend to use.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Maximum chunk size for HeaderAwareChunker.",
    )
    parser.add_argument(
        "--agent",
        action="store_true",
        help="Also run KnowledgeBaseAgent answers for each query.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_benchmark(
        file_paths=args.files,
        top_k=args.top_k,
        embedder_provider=args.embedder,
        chunk_size=args.chunk_size,
        show_agent=args.agent,
    )


if __name__ == "__main__":
    raise SystemExit(main())
