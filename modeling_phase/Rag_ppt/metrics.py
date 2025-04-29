from typing import List, Dict, Any


def is_valid_slide_schema(slide: Dict[str, Any], required_fields: List[str]) -> bool:
    return all(field in slide for field in required_fields)


def calculate_schema_compliance_rate(slides: List[Dict[str, Any]], required_fields: List[str]) -> float:
    if not slides:
        return 0.0
    valid = sum(1 for slide in slides if is_valid_slide_schema(slide, required_fields))
    return valid / len(slides)


def hallucination_rate(generated_answers: List[str], ground_truth_answers: List[str]) -> float:
    wrong = sum(1 for gen, truth in zip(generated_answers, ground_truth_answers)
                if gen.strip().lower() != truth.strip().lower())
    return wrong / len(ground_truth_answers) if ground_truth_answers else 1.0


def calculate_precision_at_k(retrieved: List[str], relevant: List[str], k: int = 5) -> float:
    top_k = retrieved[:k]
    return len([doc for doc in top_k if doc in relevant]) / k if k else 0.0


def evaluate_generated_slides(
    generated_slides: List[Dict[str, Any]],
    ground_truth_answers: List[str],
    retrieved_chunks: List[str],
    relevant_chunks: List[str],
    k: int = 5
):
    """Prints selected evaluation metrics cleanly."""
    print("\n📊 Evaluation Metrics:\n" + "-" * 40)

    required_fields = ["title", "bullets", "charts", "images"]

    # Schema compliance
    schema_rate = calculate_schema_compliance_rate(generated_slides, required_fields)
    print(f"Schema Compliance Rate: {schema_rate:.2%}")

    # Hallucination
    halluc_rate = hallucination_rate(
        [slide.get('summary', '') for slide in generated_slides],
        ground_truth_answers
    )
    print(f"Hallucination Rate: {halluc_rate:.2%}")

    # Precision@k
    precision = calculate_precision_at_k(retrieved_chunks, relevant_chunks, k)
    print(f"Precision@{k}: {precision:.2%}")

    print("-" * 40 + "\n")
