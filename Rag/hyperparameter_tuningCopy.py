#added llms to the fine tuning
import itertools
import time
import json
from RagAndMetrics import RAGSystem, FAISS_INDEX_PATH, METADATA_PATH, OPENROUTER_API_KEY, USE_CACHE # Import necessary constants
import os # Import os

# Define the search space for hyperparameters
search_space = {
    "embedding_model": [
        "sentence-transformers/all-mpnet-base-v2",
        "sentence-transformers/multi-qa-mpnet-base-dot-v1",
        # Add other embedding models you want to test
    ],
    "llm_model": [
        "meta-llama/llama-4-maverick:free",
        "qwen/qwq-32b:free", # Example: Add another LLM
        # Add other LLM models available via OpenRouter
    ],
    "top_k": [5, 10], # Reduced for faster testing, adjust as needed
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
#rag = RAGSystem()

def evaluate_config(config):
   
    total_bleu = 0
    total_rouge = 0
    total_latency = 0
    total_retrieval_latency = 0
    total_generation_latency = 0
    count = 0
    init_success = False
    rag_instance = None
   
   # --- Initialize RAGSystem for this specific config ---
    try:
        print(f"  Initializing RAGSystem with Embedding: {config['embedding_model']}, LLM: {config['llm_model']}...")
        # Pass models and other necessary params from config or defaults
        rag_instance = RAGSystem(
            model_name=config["embedding_model"],
            llm_model=config["llm_model"],
            index_path=FAISS_INDEX_PATH, # Use imported default or env var
            metadata_path=METADATA_PATH, # Use imported default or env var
            api_key=OPENROUTER_API_KEY, # Use imported default or env var
            use_cache=USE_CACHE # Use imported default or env var
        )
        init_success = True
        print("  RAGSystem Initialized.")
    except Exception as init_err:
        print(f"  ERROR initializing RAGSystem for config {config}: {init_err}")
        # Optionally add traceback here if needed
        return None # Skip this config if initialization fails
    # --- End Initialization ---

    if not init_success or rag_instance is None:
         return None # Should not happen if logic above is correct, but safety check



    for question in test_questions:
        reference = reference_answers.get(question, None)

        try:
            # Start timer
            answer, eval_result = rag_instance.query(
                question,
                top_k=config["top_k"],
                use_hybrid=config["use_hybrid"],
                use_query_expansion=config["use_query_expansion"],
                use_reranking=config["use_reranking"],
                use_context_compression=config["use_context_compression"],
                streaming=False
            )

            metrics = rag_instance.evaluate_answer(question, answer, reference_answer=reference)

            bleu = metrics.get("bleu", 0)
            rouge_l = metrics.get("rouge", {}).get("rouge-l", {}).get("f", 0) if isinstance(metrics.get("rouge"), dict) else 0

            total_bleu += bleu
            total_rouge += rouge_l
            total_latency += eval_result.total_latency
            total_retrieval_latency += eval_result.retrieval_latency
            total_generation_latency += eval_result.generation_latency
            count += 1
            # Add a small delay to potentially avoid rate limits
            time.sleep(0.5)

        except Exception as e:
            print(f"  Error evaluating question '{question}' with config {config}: {e}")
            continue

    if count == 0:
        print("  No questions successfully evaluated for this config.")
        return None

    avg_bleu = total_bleu / count
    avg_rouge = total_rouge / count
    avg_latency = total_latency / count
    avg_retrieval_latency = total_retrieval_latency / count
    avg_generation_latency = total_generation_latency / count

    print(f"  Avg BLEU: {avg_bleu:.4f}, Avg ROUGE-L: {avg_rouge:.4f}, Avg Latency: {avg_latency:.2f}s")

    return {
        "config": config,
        "avg_bleu": avg_bleu,
        "avg_rouge": avg_rouge,
        "avg_latency": avg_latency,
        "avg_retrieval_latency": avg_retrieval_latency,
        "avg_generation_latency": avg_generation_latency,
    }

def main():
    # Create all combinations of hyperparameters
    configs_iter = list(itertools.product(
        search_space["embedding_model"],
        search_space["llm_model"],
        search_space["top_k"],
        search_space["use_hybrid"],
        search_space["use_query_expansion"],
        search_space["use_reranking"],
        search_space["use_context_compression"],
    ))

    all_results = []
    start_time = time.time()

    print(f"--- Starting Hyperparameter Tuning ---")
    print(f"Total configurations to test: {len(configs_iter)}\n")

    for i, values in enumerate(configs_iter):
        config = {
            "embedding_model": values[0],
            "llm_model": values[1],
            "top_k": values[2],
            "use_hybrid": values[3],
            "use_query_expansion": values[4],
            "use_reranking": values[5],
            "use_context_compression": values[6],
        }
        print(f"--- Testing Config {i+1}/{len(configs_iter)} ---")
        print(f"Config: {json.dumps(config)}") # Print config clearly

        result = evaluate_config(config)
        if result:
            all_results.append(result)
        print("-" * 30) # Separator between configs
        # Add a longer delay between full config runs if rate limits are severe
        time.sleep(2)


    # Sort by BLEU + ROUGE score, lowest latency
    # Handle cases where scores might be None or NaN if errors occurred
    best = sorted(
        [r for r in all_results if r.get("avg_bleu") is not None and r.get("avg_rouge") is not None],
        key=lambda x: (-(x.get("avg_bleu", 0) + x.get("avg_rouge", 0)), x.get("avg_latency", float('inf')))
    )
    print("\n--- Top 5 Configurations ---")
    if not best:
        print("No configurations completed successfully.")
    else:
        for b in best[:5]:
            print(json.dumps(b, indent=2))

    # Save all results
    results_filename = "tuning_results_with_models.json"
    print(f"\nSaving all results to {results_filename}...")
    try:
        with open(results_filename, "w") as f:
            json.dump(all_results, f, indent=2)
        print("Results saved successfully.")
    except Exception as save_err:
        print(f"Error saving results: {save_err}")


    total_time = time.time() - start_time
    print(f"\n--- Tuning completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes). ---")

if __name__ == "__main__":
    main()
