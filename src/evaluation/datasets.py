"""
Evaluation Dataset Management

Provides utilities for creating, loading, and managing evaluation
datasets for RAG quality measurement.
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, field, asdict


@dataclass
class EvaluationSample:
    """A single evaluation sample"""
    question: str
    ground_truth: str = ""
    contexts: List[str] = field(default_factory=list)
    answer: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class EvaluationDataset:
    """Manages evaluation datasets for RAG testing."""

    def __init__(self, name: str = "default"):
        self.name = name
        self.samples: List[EvaluationSample] = []

    def add_sample(self, sample: EvaluationSample):
        self.samples.append(sample)

    def add_samples(self, samples: List[EvaluationSample]):
        self.samples.extend(samples)

    @property
    def questions(self) -> List[str]:
        return [s.question for s in self.samples]

    @property
    def ground_truths(self) -> List[str]:
        return [s.ground_truth for s in self.samples]

    @property
    def contexts(self) -> List[List[str]]:
        return [s.contexts for s in self.samples]

    @property
    def answers(self) -> List[str]:
        return [s.answer for s in self.samples]

    def save(self, path: str):
        data = {
            "name": self.name,
            "samples": [asdict(s) for s in self.samples],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "EvaluationDataset":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        dataset = cls(name=data.get("name", "loaded"))
        for sample_data in data.get("samples", []):
            dataset.add_sample(EvaluationSample(**sample_data))

        return dataset

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        return f"EvaluationDataset(name='{self.name}', samples={len(self.samples)})"
