"""Compatibility entry point for the RSI skill-evolution implementation."""

from research.rsi.evolution.skill_evolution import (
    OUTPUT_SCHEMA, RemoteEvaluator, generate_astra, generation_prompt, main,
    primitive_generation_prompt, training_feedback,
)


if __name__ == '__main__':
    raise SystemExit(main())
