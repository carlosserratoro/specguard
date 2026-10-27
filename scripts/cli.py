# Copyright 2026 Carlos Serra-Toro
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse

from specguard.adapters.llm import DEFAULT_JUDGE_MODEL, DEFAULT_JUDGE_TEMPERATURE, JudgeConfig


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "-k", "--reps", type=int, required=True, help="Number of repetitions per condition."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_JUDGE_MODEL,
        help=f"Judge model. Omit to use the default model {DEFAULT_JUDGE_MODEL}.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_JUDGE_TEMPERATURE,
        help=f"Judge temperature. Omit to use the default temperature {DEFAULT_JUDGE_TEMPERATURE}.",
    )
    parser.add_argument(
        "--effort",
        default=None,
        choices=["none", "low", "medium", "high", "xhigh"],
        help="Judge reasoning effort. Omit for no reasoning.",
    )
    parser.add_argument(
        "--scaffolding",
        action="store_true",
        help="Ask the judge to analyse the threshold before deciding.",
    )
    parser.add_argument(
        "--case",
        default="t01_discounts",
        help="Case ID used for the example files and report names.",
    )
    return parser


def judge_config_from_args(args: argparse.Namespace) -> JudgeConfig:
    return JudgeConfig(
        model=args.model,
        temperature=args.temperature,
        reasoning_effort=args.effort,
        scaffolding=args.scaffolding,
    )
