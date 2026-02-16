"""
RAG Evaluation Metrics

Provides RAGAS-based evaluation for retrieval and generation quality.
Falls back to simple metrics when RAGAS is unavailable.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

try:
    from ragas import evaluate
    from ragas.metrics import (
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    )
    from datasets import Dataset

    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False


@dataclass
class EvaluationResult:
    metrics: Dict[str, float] = field(default_factory=dict)
    sample_scores: List[Dict[str, float]] = field(default_factory=list)
    strategy_name: str = ""


class RAGEvaluator:
    """Evaluates RAG pipeline quality using RAGAS or simple fallback metrics."""

    def __init__(self, llm=None, embeddings=None):
        self.llm = llm
        self.embeddings = embeddings

    def evaluate_results(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: Optional[List[str]] = None,
    ) -> EvaluationResult:
        if RAGAS_AVAILABLE:
            return self._evaluate_with_ragas(questions, answers, contexts, ground_truths)
        return self._evaluate_simple(questions, answers, contexts, ground_truths)

    def _evaluate_with_ragas(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: Optional[List[str]] = None,
    ) -> EvaluationResult:
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }
        if ground_truths:
            data["ground_truth"] = ground_truths

        dataset = Dataset.from_dict(data)

        metrics_list = [context_precision, faithfulness, answer_relevancy]
        if ground_truths:
            metrics_list.append(context_recall)

        try:
            eval_kwargs = {"dataset": dataset, "metrics": metrics_list}
            if self.llm:
                eval_kwargs["llm"] = self.llm
            if self.embeddings:
                eval_kwargs["embeddings"] = self.embeddings

            result = evaluate(**eval_kwargs)

            return EvaluationResult(
                metrics=dict(result),
                sample_scores=[],
            )
        except Exception as e:
            print(f"RAGAS evaluation failed, falling back to simple: {e}")
            return self._evaluate_simple(questions, answers, contexts, ground_truths)

    def _evaluate_simple(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: Optional[List[str]] = None,
    ) -> EvaluationResult:
        num_samples = len(questions)
        if num_samples == 0:
            return EvaluationResult()

        context_lengths = [sum(len(c) for c in ctx) for ctx in contexts]
        answer_lengths = [len(a) for a in answers]
        context_coverage = sum(
            1 for ctx in contexts if any(c.strip() for c in ctx)
        ) / num_samples
        answer_coverage = sum(1 for a in answers if a.strip()) / num_samples

        metrics = {
            "avg_context_length": sum(context_lengths) / num_samples,
            "avg_answer_length": sum(answer_lengths) / num_samples,
            "context_coverage": context_coverage,
            "answer_coverage": answer_coverage,
        }

        return EvaluationResult(metrics=metrics)

    def compare_strategies(
        self,
        questions: List[str],
        strategy_results: Dict[str, Dict[str, List]],
        ground_truths: Optional[List[str]] = None,
    ) -> Dict[str, EvaluationResult]:
        results = {}
        for name, data in strategy_results.items():
            result = self.evaluate_results(
                questions=questions,
                answers=data["answers"],
                contexts=data["contexts"],
                ground_truths=ground_truths,
            )
            result.strategy_name = name
            results[name] = result
        return results
