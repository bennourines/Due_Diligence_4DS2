import itertools
import time
import json
from RagAndMetrics import RAGSystem

# Define the search space for hyperparameters
search_space = {
    "top_k": [3,10,15],
    "use_hybrid": [True, False],
    "use_query_expansion": [True, False],
    "use_reranking": [True, False],
    "use_context_compression": [True, False],
}

# Test questions
test_questions = [
    "What are the main challenges in blockchain scalability?",
    "Explain the concept of Proof-of-Stake consensus.",
    "How secure are decentralized finance protocols?",
]

# Optional: reference answers if available for BLEU/METEOR/ROUGE evaluation
reference_answers = {
    "What are the main challenges in blockchain scalability?": "Blockchain scalability faces challenges like transaction throughput limits, network congestion, and high fees. Solutions include layer 2 scaling and sharding.",
    "Explain the concept of Proof-of-Stake consensus.": "Proof-of-Stake (PoS) is a consensus mechanism where validators are chosen based on the amount of cryptocurrency they stake as collateral, reducing energy consumption compared to Proof-of-Work.",
    "How secure are decentralized finance protocols?": "DeFi protocols vary in security depending on smart contract quality, audits, governance models, and user practices. Vulnerabilities can lead to hacks and exploits.",
}

# Initialize RAG system
rag = RAGSystem()

def evaluate_config(config):
    total_bleu = 0
    total_rouge = 0
    total_latency = 0
    total_retrieval_latency = 0
    total_generation_latency = 0
    count = 0

    for question in test_questions:
        reference = reference_answers.get(question, None)

        try:
            # Start timer
            answer, eval_result = rag.query(
                question,
                top_k=config["top_k"],
                use_hybrid=config["use_hybrid"],
                use_query_expansion=config["use_query_expansion"],
                use_reranking=config["use_reranking"],
                use_context_compression=config["use_context_compression"],
                streaming=False
            )

            metrics = rag.evaluate_answer(question, answer, reference_answer=reference)

            bleu = metrics.get("bleu", 0)
            rouge_l = metrics.get("rouge", {}).get("rouge-l", {}).get("f", 0) if "rouge" in metrics else 0

            total_bleu += bleu
            total_rouge += rouge_l
            total_latency += eval_result.total_latency
            total_retrieval_latency += eval_result.retrieval_latency
            total_generation_latency += eval_result.generation_latency
            count += 1

        except Exception as e:
            print(f"Error evaluating question '{question}': {e}")
            continue

    if count == 0:
        return None

    avg_bleu = total_bleu / count
    avg_rouge = total_rouge / count
    avg_latency = total_latency / count
    avg_retrieval_latency = total_retrieval_latency / count
    avg_generation_latency = total_generation_latency / count

    return {
        "config": config,
        "avg_bleu": avg_bleu,
        "avg_rouge": avg_rouge,
        "avg_latency": avg_latency,
        "avg_retrieval_latency": avg_retrieval_latency,
        "avg_generation_latency": avg_generation_latency,
    }

def main():
    configs = list(itertools.product(
        search_space["top_k"],
        search_space["use_hybrid"],
        search_space["use_query_expansion"],
        search_space["use_reranking"],
        search_space["use_context_compression"],
    ))

    all_results = []
    start_time = time.time()

    print(f"Running {len(configs)} configurations...\n")

    for values in configs:
        config = {
            "top_k": values[0],
            "use_hybrid": values[1],
            "use_query_expansion": values[2],
            "use_reranking": values[3],
            "use_context_compression": values[4],
        }
        print(f"Testing config: {config}")
        result = evaluate_config(config)
        if result:
            all_results.append(result)

    # Sort by BLEU + ROUGE score, lowest latency
    best = sorted(all_results, key=lambda x: (-(x["avg_bleu"] + x["avg_rouge"]), x["avg_latency"]))

    print("\nTop 5 Configurations:")
    for b in best[:5]:
        print(json.dumps(b, indent=2))

    # Save all results
    with open("tuning_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    total_time = time.time() - start_time
    print(f"\nTuning completed in {total_time:.2f} seconds.")

if __name__ == "__main__":
    main()
