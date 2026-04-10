#!/usr/bin/env python3
"""
Calculate actual cosine similarity scores for sentence pairs using all-MiniLM-L6-v2 embeddings.
"""

from src.embeddings import LocalEmbedder
from src.chunking import compute_similarity
import json

# Initialize embedder (same as used in benchmark)
embedder = LocalEmbedder()

# Sentence pairs for similarity prediction (from REPORT.md section 5)
sentence_pairs = [
    {
        "pair": 1,
        "sentenceA": "Tim mạch là cơ quan quan trọng.",
        "sentenceB": "Bệnh tim mạch ảnh hưởng đến chức năng tim.",
        "prediction": "high"
    },
    {
        "pair": 2,
        "sentenceA": "Ăn nhiều rau xanh giúp kiểm soát huyết áp.",
        "sentenceB": "Tập thể dục đều đặn cải thiện sức khỏe tim.",
        "prediction": "low"
    },
    {
        "pair": 3,
        "sentenceA": "Nhồi máu cơ tim có thể gây đau ngực.",
        "sentenceB": "Đau ngực kéo dài có thể là dấu hiệu của nhồi máu cơ tim.",
        "prediction": "high"
    },
    {
        "pair": 4,
        "sentenceA": "Tăng cường kali trong chế độ ăn giúp giảm huyết áp.",
        "sentenceB": "Muối dư thừa có thể làm tăng huyết áp.",
        "prediction": "medium"
    },
    {
        "pair": 5,
        "sentenceA": "Suy tim trái gây phù phổi.",
        "sentenceB": "Suy tim phải thường gây phù ngoại vi.",
        "prediction": "low"
    }
]

print("=" * 80)
print("SIMILARITY PREDICTION RESULTS")
print("=" * 80)
print()

results = []

for pair_data in sentence_pairs:
    pair_num = pair_data["pair"]
    sent_a = pair_data["sentenceA"]
    sent_b = pair_data["sentenceB"]
    prediction = pair_data["prediction"]
    
    # Generate embeddings
    embed_a = embedder(sent_a)
    embed_b = embedder(sent_b)
    
    # Calculate cosine similarity
    score = compute_similarity(embed_a, embed_b)
    
    # Classify result
    if score > 0.5:
        classification = "high"
    elif score > 0.3:
        classification = "medium"
    else:
        classification = "low"
    
    # Check if prediction was correct
    correct = classification == prediction or (prediction == "medium" and 0.3 < score <= 0.5)
    
    result = {
        "pair": pair_num,
        "sentenceA": sent_a,
        "sentenceB": sent_b,
        "prediction": prediction,
        "actual_score": round(score, 4),
        "classification": classification,
        "correct": "yes" if correct else "no"
    }
    
    results.append(result)
    
    print(f"Pair {pair_num}:")
    print(f"  A: {sent_a}")
    print(f"  B: {sent_b}")
    print(f"  Prediction: {prediction}")
    print(f"  Actual Score: {score:.4f} ({classification})")
    print(f"  Correct: {'yes' if correct else 'no'}")
    print()

# Generate Markdown table
print("=" * 80)
print("MARKDOWN TABLE FORMAT")
print("=" * 80)
print()
print("| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |")
print("|------|-----------|-----------|---------|--------------|-------|")

for r in results:
    sent_a_short = r["sentenceA"][:50] + "..." if len(r["sentenceA"]) > 50 else r["sentenceA"]
    sent_b_short = r["sentenceB"][:50] + "..." if len(r["sentenceB"]) > 50 else r["sentenceB"]
    print(f"| {r['pair']} | {sent_a_short} | {sent_b_short} | {r['prediction']} | {r['actual_score']} | {r['correct']} |")

print()
print("=" * 80)
print("INTERPRETATION")
print("=" * 80)
print()

# Check most surprising result
all_scores = [r["actual_score"] for r in results]
max_diff_idx = max(range(len(results)), key=lambda i: abs(all_scores[i] - (0.5 if results[i]['prediction'] == 'high' else 0.25)))

print(f"Most surprising result: Pair {results[max_diff_idx]['pair']}")
print(f"  Actual score: {results[max_diff_idx]['actual_score']}")
print(f"  This shows that embeddings model recognizes...")
print()

# Save results to JSON for reference
with open("similarity_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("Results saved to similarity_results.json")
