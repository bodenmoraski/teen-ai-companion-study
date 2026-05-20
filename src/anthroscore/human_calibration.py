"""
Human Validation Calibration for AnthroScore V3.

Loads human annotator data (Stephanie, Boden, Afia, and Answer Key), computes
inter-rater reliability across all 3 annotators, identifies systematic algorithm
biases, and generates calibrated few-shot examples for improved LLM prompting.
"""

import csv
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from itertools import combinations

import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
VALIDATIONS_DIR = PROJECT_ROOT / "Validations"

ANNOTATOR_NAMES = ["stephanie", "boden", "afia"]


@dataclass
class HumanAnnotation:
    item_num: int
    text: str
    subreddit: str
    stephanie_score: Optional[int] = None
    boden_score: Optional[int] = None
    afia_score: Optional[int] = None
    algorithm_score: Optional[int] = None
    algorithm_reasoning: str = ""
    stephanie_reasoning: str = ""
    boden_reasoning: str = ""
    afia_reasoning: str = ""
    human_consensus: Optional[float] = None
    human_scores_available: int = 0

    def get_human_scores(self) -> List[int]:
        """Return all non-None human scores for this item."""
        return [
            s for s in [self.stephanie_score, self.boden_score, self.afia_score]
            if s is not None
        ]


@dataclass
class PairwiseAgreement:
    """Agreement metrics between two annotators."""
    annotator_a: str
    annotator_b: str
    n_paired: int
    exact_agreement: float
    within_1_agreement: float
    cohens_kappa: float
    mean_abs_diff: float


@dataclass
class CalibrationResult:
    """Output of the calibration analysis."""
    n_items: int = 0
    n_annotators: int = 0
    pairwise_agreements: List[PairwiseAgreement] = field(default_factory=list)
    overall_inter_rater_agreement: float = 0.0
    overall_inter_rater_kappa: float = 0.0
    human_algo_agreement: float = 0.0
    human_algo_within_1: float = 0.0
    mean_algo_bias: float = 0.0
    bias_direction: str = ""
    high_disagreement_examples: List[Dict] = field(default_factory=list)
    calibration_examples: List[Dict] = field(default_factory=list)
    score_confusion: Dict = field(default_factory=dict)


def _parse_score(val: str) -> Optional[int]:
    """Parse a 1-5 score value from CSV, handling edge cases."""
    if not val or not val.strip():
        return None
    try:
        score = int(float(val.strip()))
        return score if 1 <= score <= 5 else None
    except (ValueError, TypeError):
        return None


def _parse_int(val: str) -> Optional[int]:
    """Parse any positive integer from CSV (for item numbers, etc.)."""
    if not val or not val.strip():
        return None
    try:
        return int(float(val.strip()))
    except (ValueError, TypeError):
        return None


def _load_annotator_csv(
    csv_path: Path,
    annotations_by_item: Dict[int, HumanAnnotation],
    annotator_name: str,
):
    """Load a single annotator's CSV into the shared annotations dict."""
    logger.info(f"Loading {annotator_name.title()} annotations from {csv_path.name}")
    with open(csv_path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            item_num = _parse_int(row.get("Item #", ""))
            if item_num is None:
                continue
            score = _parse_score(row.get("Your Score (1-5)", ""))
            reasoning = row.get("[OPTIONAL] Your Reasoning (1-2 sentences)", "")

            if item_num not in annotations_by_item:
                annotations_by_item[item_num] = HumanAnnotation(
                    item_num=item_num,
                    text=row.get("Comment Text", ""),
                    subreddit=row.get("Subreddit", ""),
                )

            ann = annotations_by_item[item_num]
            setattr(ann, f"{annotator_name}_score", score)
            setattr(ann, f"{annotator_name}_reasoning", reasoning)


def load_validation_data(validations_dir: Path = VALIDATIONS_DIR) -> List[HumanAnnotation]:
    """
    Load and merge all validation CSVs into unified annotation records.

    Detects annotator files by name pattern (stephanie, boden, afia) and
    the answer key. Supports 2 or 3 human annotators.

    Returns a list of HumanAnnotation objects with scores from all
    human annotators and the algorithm.
    """
    annotations_by_item: Dict[int, HumanAnnotation] = {}

    annotator_paths: Dict[str, Optional[Path]] = {name: None for name in ANNOTATOR_NAMES}
    answer_key_path = None

    for f in validations_dir.iterdir():
        if f.suffix != ".csv":
            continue
        name_lower = f.name.lower()

        if "answer_key" in name_lower:
            answer_key_path = f
            continue

        # Match annotator files — look for the name AND "annotations" in the filename
        # to avoid matching the rubric/instructions file
        if "annotation" in name_lower:
            for annotator_name in ANNOTATOR_NAMES:
                if annotator_name in name_lower:
                    annotator_paths[annotator_name] = f
                    break

    # Load the answer key first (provides algorithm scores + comment text)
    if answer_key_path:
        logger.info(f"Loading answer key from {answer_key_path.name}")
        with open(answer_key_path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                item_num = _parse_int(row.get("Item #", ""))
                if item_num is None:
                    continue
                ann = HumanAnnotation(
                    item_num=item_num,
                    text=row.get("Comment Text (Cleaned)", row.get("Comment Text", "")),
                    subreddit=row.get("Subreddit", ""),
                    algorithm_score=_parse_score(row.get("AnthroScore V3 (Algorithm)", "")),
                    algorithm_reasoning=row.get("Algorithm Reasoning", ""),
                )
                annotations_by_item[item_num] = ann

    # Load each annotator's file
    for annotator_name, path in annotator_paths.items():
        if path is not None:
            _load_annotator_csv(path, annotations_by_item, annotator_name)

    # Compute human consensus (mean of all available human scores)
    for ann in annotations_by_item.values():
        scores = ann.get_human_scores()
        ann.human_scores_available = len(scores)
        if scores:
            ann.human_consensus = np.mean(scores)

    found_annotators = [name for name, path in annotator_paths.items() if path is not None]
    result = sorted(annotations_by_item.values(), key=lambda a: a.item_num)
    logger.info(
        f"Loaded {len(result)} validation items from "
        f"{len(found_annotators)} annotators: {', '.join(found_annotators)}"
    )
    return result


def compute_cohens_kappa(scores_a: List[int], scores_b: List[int]) -> float:
    """Compute Cohen's kappa for two lists of ordinal ratings."""
    if len(scores_a) != len(scores_b) or len(scores_a) == 0:
        return 0.0

    n = len(scores_a)
    categories = sorted(set(scores_a) | set(scores_b))
    k = len(categories)
    cat_to_idx = {c: i for i, c in enumerate(categories)}

    confusion = np.zeros((k, k))
    for a, b in zip(scores_a, scores_b):
        confusion[cat_to_idx[a], cat_to_idx[b]] += 1

    p_o = np.trace(confusion) / n

    row_sums = confusion.sum(axis=1)
    col_sums = confusion.sum(axis=0)
    p_e = (row_sums * col_sums).sum() / (n * n)

    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1.0 - p_e)


def _compute_pairwise_agreements(
    annotations: List[HumanAnnotation],
) -> List[PairwiseAgreement]:
    """Compute pairwise agreement for all annotator pairs."""
    results = []

    for name_a, name_b in combinations(ANNOTATOR_NAMES, 2):
        paired = []
        for ann in annotations:
            score_a = getattr(ann, f"{name_a}_score", None)
            score_b = getattr(ann, f"{name_b}_score", None)
            if score_a is not None and score_b is not None:
                paired.append((score_a, score_b))

        if not paired:
            continue

        scores_a, scores_b = zip(*paired)
        scores_a, scores_b = list(scores_a), list(scores_b)

        exact = sum(1 for a, b in paired if a == b) / len(paired)
        within_1 = sum(1 for a, b in paired if abs(a - b) <= 1) / len(paired)
        kappa = compute_cohens_kappa(scores_a, scores_b)
        mean_diff = np.mean([abs(a - b) for a, b in paired])

        results.append(PairwiseAgreement(
            annotator_a=name_a.title(),
            annotator_b=name_b.title(),
            n_paired=len(paired),
            exact_agreement=exact,
            within_1_agreement=within_1,
            cohens_kappa=kappa,
            mean_abs_diff=mean_diff,
        ))

    return results


def run_calibration(
    annotations: Optional[List[HumanAnnotation]] = None,
    n_examples: int = 10,
) -> CalibrationResult:
    """
    Analyze human vs algorithm agreement and produce calibration data.

    Supports 2 or 3 human annotators. Computes:
    - Pairwise inter-rater reliability for all annotator pairs
    - Overall agreement metrics (averaged across pairs)
    - Systematic bias (does the algorithm over- or under-score?)
    - High-disagreement cases (where humans agree but the algorithm diverges)
    - Calibration examples (human-consensus cases for few-shot prompting)
    """
    if annotations is None:
        annotations = load_validation_data()

    result = CalibrationResult()
    result.n_items = len(annotations)

    # Count how many annotators contributed
    active_annotators = set()
    for ann in annotations:
        for name in ANNOTATOR_NAMES:
            if getattr(ann, f"{name}_score", None) is not None:
                active_annotators.add(name)
    result.n_annotators = len(active_annotators)

    # --- Pairwise inter-rater reliability ---
    result.pairwise_agreements = _compute_pairwise_agreements(annotations)

    if result.pairwise_agreements:
        result.overall_inter_rater_agreement = np.mean(
            [p.exact_agreement for p in result.pairwise_agreements]
        )
        result.overall_inter_rater_kappa = np.mean(
            [p.cohens_kappa for p in result.pairwise_agreements]
        )

    # --- Human consensus vs algorithm ---
    triplets = []
    for a in annotations:
        if a.human_consensus is not None and a.algorithm_score is not None:
            human_rounded = round(a.human_consensus)
            triplets.append((a, human_rounded, a.algorithm_score))

    if triplets:
        exact_matches = sum(1 for _, h, alg in triplets if h == alg)
        within_1_matches = sum(1 for _, h, alg in triplets if abs(h - alg) <= 1)
        result.human_algo_agreement = exact_matches / len(triplets)
        result.human_algo_within_1 = within_1_matches / len(triplets)

        biases = [alg - h for _, h, alg in triplets]
        result.mean_algo_bias = np.mean(biases)
        if result.mean_algo_bias > 0.3:
            result.bias_direction = "algorithm_overscores"
        elif result.mean_algo_bias < -0.3:
            result.bias_direction = "algorithm_underscores"
        else:
            result.bias_direction = "roughly_aligned"

        confusion = defaultdict(lambda: defaultdict(int))
        for _, h, alg in triplets:
            confusion[h][alg] += 1
        result.score_confusion = {k: dict(v) for k, v in confusion.items()}

    # --- High-disagreement examples (algorithm diverges from human consensus by >=2) ---
    disagreements = []
    for ann, h_score, alg_score in triplets:
        diff = abs(alg_score - h_score)
        if diff >= 2:
            disagreements.append({
                "item": ann.item_num,
                "text": ann.text[:200],
                "human_consensus": h_score,
                "algorithm_score": alg_score,
                "difference": alg_score - h_score,
                "stephanie": ann.stephanie_score,
                "boden": ann.boden_score,
                "afia": ann.afia_score,
                "n_human_scores": ann.human_scores_available,
                "algo_reasoning": ann.algorithm_reasoning,
            })
    disagreements.sort(key=lambda d: abs(d["difference"]), reverse=True)
    result.high_disagreement_examples = disagreements

    # --- Calibration examples: cases where humans agree well ---
    calibration_candidates = []
    for ann, h_score, alg_score in triplets:
        human_scores = ann.get_human_scores()
        if len(human_scores) < 2:
            continue
        human_spread = max(human_scores) - min(human_scores)
        if human_spread <= 1:
            # Pick the best reasoning from any annotator
            reasoning = (
                ann.stephanie_reasoning
                or ann.boden_reasoning
                or ann.afia_reasoning
            )
            calibration_candidates.append({
                "text": ann.text,
                "correct_score": h_score,
                "stephanie": ann.stephanie_score,
                "boden": ann.boden_score,
                "afia": ann.afia_score,
                "n_human_scores": ann.human_scores_available,
                "human_spread": human_spread,
                "algorithm_score": alg_score,
                "algo_was_correct": abs(alg_score - h_score) <= 1,
                "reasoning_hint": reasoning,
            })

    # Pick diverse examples across the score range
    by_score = defaultdict(list)
    for ex in calibration_candidates:
        by_score[ex["correct_score"]].append(ex)

    selected = []
    per_score = max(1, n_examples // max(len(by_score), 1))
    for score in sorted(by_score.keys()):
        pool = by_score[score]
        # Prefer: (1) 3-annotator consensus, (2) algorithm was wrong
        pool.sort(key=lambda x: (-x["n_human_scores"], x["algo_was_correct"]))
        selected.extend(pool[:per_score])

    result.calibration_examples = selected[:n_examples]

    return result


def generate_calibrated_prompt(
    base_prompt: str,
    calibration: Optional[CalibrationResult] = None,
    n_examples: int = 6,
) -> str:
    """
    Enhance the base AnthroScore prompt with calibration examples from
    human validation data.

    Injects few-shot examples where human annotators agreed, focusing on
    cases where the algorithm previously erred, teaching the model to
    correct its systematic biases.
    """
    if calibration is None:
        calibration = run_calibration(n_examples=n_examples)

    examples_block = []

    bias_note = ""
    if calibration.bias_direction == "algorithm_overscores":
        bias_note = (
            "\nCALIBRATION NOTE: Previous scoring has tended to OVER-SCORE. "
            "Be conservative — casual use of 'love' for a feature/website is NOT "
            "anthropomorphization (score 1-2). Only score high when emotions are "
            "genuinely attributed TO the AI as a being.\n"
        )
    elif calibration.bias_direction == "algorithm_underscores":
        bias_note = (
            "\nCALIBRATION NOTE: Previous scoring has tended to UNDER-SCORE. "
            "Subtle relationship language and emotional attribution should be "
            "weighted more heavily.\n"
        )

    for ex in calibration.calibration_examples[:n_examples]:
        text_preview = ex["text"][:150].replace('"', "'")
        reasoning = ex.get("reasoning_hint", "") or ""
        reasoning_str = f" ({reasoning})" if reasoning else ""
        n_humans = ex.get("n_human_scores", 2)
        examples_block.append(
            f'  - "{text_preview}..." → Score {ex["correct_score"]} '
            f'[{n_humans}-annotator consensus]{reasoning_str}'
        )

    if not examples_block:
        return base_prompt

    examples_section = "\n".join(examples_block)

    calibration_insert = f"""{bias_note}
HUMAN-VALIDATED CALIBRATION EXAMPLES ({calibration.n_annotators} annotators):
{examples_section}
"""

    if "COMMENT:" in base_prompt:
        parts = base_prompt.split("COMMENT:")
        return parts[0] + calibration_insert + "\nCOMMENT:" + parts[1]

    if "Respond with ONLY" in base_prompt:
        parts = base_prompt.split("Respond with ONLY")
        return parts[0] + calibration_insert + "\nRespond with ONLY" + parts[1]

    return base_prompt + "\n" + calibration_insert


def print_calibration_report(calibration: CalibrationResult) -> str:
    """Generate a human-readable calibration report."""
    lines = [
        "=" * 70,
        "ANTHROSCORE HUMAN CALIBRATION REPORT",
        "=" * 70,
        f"Total validation items: {calibration.n_items}",
        f"Number of human annotators: {calibration.n_annotators}",
        "",
    ]

    # Pairwise agreement table
    if calibration.pairwise_agreements:
        lines.append("--- Pairwise Inter-Rater Reliability ---")
        lines.append(
            f"{'Pair':<25} {'N':>5} {'Exact':>7} {'±1':>7} "
            f"{'Kappa':>7} {'MAD':>7}"
        )
        lines.append("-" * 65)
        for p in calibration.pairwise_agreements:
            lines.append(
                f"{p.annotator_a} vs {p.annotator_b:<12} "
                f"{p.n_paired:>5} {p.exact_agreement:>6.1%} "
                f"{p.within_1_agreement:>6.1%} {p.cohens_kappa:>7.3f} "
                f"{p.mean_abs_diff:>6.2f}"
            )
        lines.append("")
        lines.append(
            f"Overall mean exact agreement: "
            f"{calibration.overall_inter_rater_agreement:.1%}"
        )
        lines.append(
            f"Overall mean Cohen's kappa:  "
            f"{calibration.overall_inter_rater_kappa:.3f}"
        )
        lines.append("")

    lines.extend([
        "--- Algorithm vs Human Consensus ---",
        f"Exact agreement:  {calibration.human_algo_agreement:.1%}",
        f"Within ±1:        {calibration.human_algo_within_1:.1%}",
        f"Mean bias (algo - human): {calibration.mean_algo_bias:+.2f}",
        f"Bias direction: {calibration.bias_direction}",
        "",
    ])

    if calibration.score_confusion:
        lines.append("--- Confusion Matrix (Human Consensus → Algo) ---")
        all_scores = sorted(
            set(list(calibration.score_confusion.keys()))
            | {s for row in calibration.score_confusion.values() for s in row}
        )
        header = "Human\\Algo  " + "  ".join(f"{s:>3}" for s in all_scores)
        lines.append(header)
        for h in all_scores:
            row_data = calibration.score_confusion.get(h, {})
            cells = "  ".join(f"{row_data.get(a, 0):>3}" for a in all_scores)
            lines.append(f"     {h:>3}     {cells}")
        lines.append("")

    if calibration.high_disagreement_examples:
        lines.append(
            f"--- High-Disagreement Cases "
            f"({len(calibration.high_disagreement_examples)}) ---"
        )
        for d in calibration.high_disagreement_examples[:5]:
            human_scores = ", ".join(
                f"{name.title()}={d.get(name)}"
                for name in ANNOTATOR_NAMES
                if d.get(name) is not None
            )
            lines.append(
                f"  Item {d['item']}: Consensus={d['human_consensus']}, "
                f"Algo={d['algorithm_score']} (diff={d['difference']:+d})"
            )
            lines.append(f"    Annotators: {human_scores}")
            lines.append(f"    Text: {d['text'][:100]}...")
        lines.append("")

    lines.append("=" * 70)
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    annotations = load_validation_data()
    calibration = run_calibration(annotations)
    report = print_calibration_report(calibration)
    print(report)
