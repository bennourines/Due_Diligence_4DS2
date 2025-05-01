import os
import json
from typing import List, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import Dataset
from langsmith import Client
from langsmith.evaluation import RunEvaluator
from langsmith.schemas import Example
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Configuration
EVALUATION_DATA_PATH = "./evaluation_data.json"
RESULTS_PATH = "./evaluation_results.json"
PLOTS_DIR = "./evaluation_plots"
VECTORIZER = TfidfVectorizer(stop_words='english')

# Create plots directory if it doesn't exist
os.makedirs(PLOTS_DIR, exist_ok=True)

def load_evaluation_data(file_path: str) -> List[Dict[str, Any]]:
    """Load evaluation data from JSON file."""
    if not os.path.exists(file_path):
        print(f"Evaluation data file not found at {file_path}")
        return []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_semantic_similarity(text1: str, text2: str) -> float:
    """Calculate semantic similarity between two texts using TF-IDF and cosine similarity."""
    try:
        tfidf_matrix = VECTORIZER.fit_transform([text1, text2])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(similarity)
    except Exception:
        return 0.0

def calculate_rouge_scores(prediction: str, reference: str) -> Dict[str, float]:
    """Calculate ROUGE scores."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(prediction, reference)
    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure
    }

def calculate_bleu_score(prediction: str, reference: str) -> float:
    """Calculate BLEU score."""
    try:
        smoothing = SmoothingFunction().method1
        prediction_tokens = prediction.lower().split()
        reference_tokens = [reference.lower().split()]
        return sentence_bleu(reference_tokens, prediction_tokens, smoothing_function=smoothing)
    except Exception:
        return 0.0

def evaluate_context_metrics(data: Dict[str, Any]) -> Dict[str, float]:
    """Evaluate context-related metrics."""
    context_text = " ".join(data["contexts"])
    return {
        "context_relevancy": calculate_semantic_similarity(data["question"], context_text),
        "context_completeness": calculate_semantic_similarity(data["answer"], context_text),
        "context_precision": calculate_semantic_similarity(data["ground_truth"], context_text)
    }

def evaluate_response_metrics(data: Dict[str, Any]) -> Dict[str, float]:
    """Evaluate response-related metrics."""
    return {
        "response_relevancy": calculate_semantic_similarity(data["question"], data["answer"]),
        "factual_correctness": calculate_semantic_similarity(data["answer"], data["ground_truth"]),
        "semantic_similarity": calculate_semantic_similarity(data["answer"], data["ground_truth"]),
        "bleu_score": calculate_bleu_score(data["answer"], data["ground_truth"]),
        **calculate_rouge_scores(data["answer"], data["ground_truth"])
    }

def evaluate_single_example(data: Dict[str, Any]) -> Dict[str, float]:
    """Evaluate all metrics for a single example."""
    context_metrics = evaluate_context_metrics(data)
    response_metrics = evaluate_response_metrics(data)
    
    return {**context_metrics, **response_metrics}

def evaluate_with_langsmith(evaluation_data: List[Dict[str, Any]]) -> Dict[str, float]:
    """Evaluate using LangSmith metrics."""
    try:
        results = {}
        for item in evaluation_data:
            # Calculate basic metrics without LangSmith
            results["correctness"] = calculate_semantic_similarity(
                item["answer"], item["ground_truth"]
            )
            results["context_relevance"] = calculate_semantic_similarity(
                item["question"], " ".join(item["contexts"])
            )
        return results
    except Exception as e:
        print(f"Warning: Evaluation failed: {str(e)}")
        return {}

def save_results(results: Dict[str, Any], file_path: str):
    """Save evaluation results to JSON file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

def plot_average_metrics(metrics: Dict[str, float]):
    """Create a bar plot of average metrics."""
    plt.figure(figsize=(12, 6))
    metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Score'])
    
    # Create bar plot
    sns.barplot(data=metrics_df, x='Metric', y='Score')
    plt.xticks(rotation=45, ha='right')
    plt.title('Average Evaluation Metrics')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(PLOTS_DIR, 'average_metrics.png'))
    plt.close()

def plot_metrics_heatmap(metrics: Dict[str, float]):
    """Create a heatmap of metrics scores."""
    plt.figure(figsize=(10, 2))
    metrics_df = pd.DataFrame([metrics])
    
    # Create heatmap
    sns.heatmap(metrics_df, annot=True, cmap='RdYlGn', center=0.5,
                fmt='.3f', cbar_kws={'label': 'Score'})
    plt.title('Metrics Heatmap')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(PLOTS_DIR, 'metrics_heatmap.png'))
    plt.close()

def plot_per_question_metrics(results: List[Dict]):
    """Create a plot showing metrics for each question."""
    # Prepare data
    questions = []
    metrics_data = {}
    
    for result in results:
        questions.append(f"Q{len(questions)+1}")
        for metric, value in result['metrics'].items():
            if metric not in metrics_data:
                metrics_data[metric] = []
            metrics_data[metric].append(value)
    
    # Create DataFrame
    df = pd.DataFrame(metrics_data, index=questions)
    
    # Create plot
    plt.figure(figsize=(15, 8))
    df.plot(kind='bar', width=0.8)
    plt.title('Metrics per Question')
    plt.xlabel('Questions')
    plt.ylabel('Score')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(PLOTS_DIR, 'per_question_metrics.png'))
    plt.close()

def plot_metrics_radar(metrics: Dict[str, float]):
    """Create a radar plot of metrics."""
    # Prepare the data
    categories = list(metrics.keys())
    values = list(metrics.values())
    
    # Number of variables
    num_vars = len(categories)
    
    # Compute angle for each axis
    angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
    angles += angles[:1]
    
    # Initialize the plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Plot data
    values += values[:1]
    ax.plot(angles, values)
    ax.fill(angles, values, alpha=0.25)
    
    # Fix axis to go in the right order and start at 12 o'clock
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw axis lines for each angle and label
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    
    # Add title
    plt.title('Metrics Radar Plot')
    
    # Save plot
    plt.savefig(os.path.join(PLOTS_DIR, 'metrics_radar.png'))
    plt.close()

def create_evaluation_report(metrics: Dict[str, float], custom_results: List[Dict]):
    """Create and save all visualization plots."""
    print("\nGenerating evaluation visualizations...")
    
    # Create individual plots
    plot_average_metrics(metrics)
    plot_metrics_heatmap(metrics)
    plot_per_question_metrics(custom_results)
    plot_metrics_radar(metrics)
    
    # Create HTML report
    html_report = f"""
    <html>
    <head>
        <title>RAG System Evaluation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .metric-summary {{ margin-bottom: 20px; }}
            .visualization {{ margin-bottom: 40px; }}
            img {{ max-width: 100%; border: 1px solid #ddd; }}
        </style>
    </head>
    <body>
        <h1>RAG System Evaluation Report</h1>
        
        <div class="metric-summary">
            <h2>Metric Scores</h2>
            <table border="1" cellpadding="5">
                <tr><th>Metric</th><th>Score</th></tr>
                {"".join(f"<tr><td>{k}</td><td>{v:.4f}</td></tr>" for k, v in metrics.items())}
            </table>
        </div>

        <div class="visualization">
            <h2>Average Metrics Bar Plot</h2>
            <img src="average_metrics.png" alt="Average Metrics">
        </div>

        <div class="visualization">
            <h2>Metrics Heatmap</h2>
            <img src="metrics_heatmap.png" alt="Metrics Heatmap">
        </div>

        <div class="visualization">
            <h2>Per-Question Metrics</h2>
            <img src="per_question_metrics.png" alt="Per-Question Metrics">
        </div>

        <div class="visualization">
            <h2>Metrics Radar Plot</h2>
            <img src="metrics_radar.png" alt="Metrics Radar Plot">
        </div>
    </body>
    </html>
    """
    
    with open(os.path.join(PLOTS_DIR, 'evaluation_report.html'), 'w') as f:
        f.write(html_report)
    
    print(f"Evaluation visualizations and report saved in {PLOTS_DIR}")

def main():
    # Load evaluation data
    evaluation_data = load_evaluation_data(EVALUATION_DATA_PATH)
    if not evaluation_data:
        print("No evaluation data found. Please create evaluation_data.json with your test cases.")
        return
    
    # Run custom evaluations
    print("Running custom evaluations...")
    custom_results = []
    for example in evaluation_data:
        metrics = evaluate_single_example(example)
        custom_results.append({
            "question": example["question"],
            "metrics": metrics
        })
    
    # Calculate average metrics
    all_metrics = {}
    for result in custom_results:
        for metric, value in result["metrics"].items():
            if metric not in all_metrics:
                all_metrics[metric] = []
            all_metrics[metric].append(value)
    
    average_metrics = {
        metric: sum(values) / len(values)
        for metric, values in all_metrics.items()
    }
    
    # Try to run additional evaluation
    print("Running additional evaluation metrics...")
    try:
        additional_results = evaluate_with_langsmith(evaluation_data)
    except Exception as e:
        print(f"Additional evaluation failed: {str(e)}")
        additional_results = {}
    
    # Combine results
    combined_results = {
        "custom_metrics": {
            "average": average_metrics,
            "per_question": custom_results
        },
        "additional_metrics": additional_results
    }
    
    # Save results
    save_results(combined_results, RESULTS_PATH)
    print(f"Evaluation completed. Results saved to {RESULTS_PATH}")
    
    # Create visualizations and report
    create_evaluation_report(average_metrics, custom_results)
    
    # Print summary
    print("\nEvaluation Summary:")
    print("-" * 50)
    print("\nAverage Metrics:")
    for metric, value in average_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    if additional_results:
        print("\nAdditional Metrics:")
        for metric, value in additional_results.items():
            print(f"{metric}: {value:.4f}")
    
    print(f"\nDetailed evaluation report and visualizations available in {PLOTS_DIR}/evaluation_report.html")

if __name__ == "__main__":
    main() 