"""Phase 3.1 — print the per-use-case quality thresholds.

Run:
    python -m scripts.show_quality_thresholds
"""

from app.quality import load_quality_thresholds, threshold_for


def main() -> None:
    thresholds = load_quality_thresholds()
    print("Quality thresholds (configs/quality_thresholds.yaml):\n")
    print(
        f"{'use_case':<16}{'metric':<20}{'op':<4}{'threshold':<10}"
        f"{'judge/ref':<16}notes"
    )
    for use_case in ("extraction", "summarization", "classification"):
        t = thresholds[use_case]
        model = t.judge_model or t.reference_model or "-"
        notes = (t.notes[:48] + "…") if len(t.notes) > 48 else t.notes
        print(
            f"{t.use_case:<16}{t.metric:<20}{t.comparison:<4}"
            f"{t.threshold:<10}{model:<16}{notes}"
        )

    print("\nSmoke threshold_for:")
    for use_case in ("extraction", "summarization", "classification"):
        t = threshold_for(use_case)
        print(
            f"  {use_case}: {t.metric} {t.comparison} {t.threshold}"
            + (f" (scale_max={t.scale_max})" if t.scale_max is not None else "")
        )


if __name__ == "__main__":
    main()
