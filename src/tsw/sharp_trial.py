"""Trial wrapper for Apple's SHARP image-to-Gaussian-splat CLI."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

SHARP_REPO = "git+https://github.com/apple/ml-sharp"


def build_predict_command(
    input_path: Path,
    output_path: Path,
    *,
    checkpoint_path: Path | None = None,
    device: str = "default",
    render: bool = False,
    verbose: bool = False,
) -> list[str]:
    """Build the uvx command that runs SHARP prediction."""
    command = [
        "uvx",
        "--from",
        SHARP_REPO,
        "sharp",
        "predict",
        "--input-path",
        str(input_path),
        "--output-path",
        str(output_path),
        "--device",
        device,
    ]

    if checkpoint_path is not None:
        command.extend(["--checkpoint-path", str(checkpoint_path)])
    if render:
        command.append("--render")
    if verbose:
        command.append("--verbose")

    return command


def predict(
    input_path: Path,
    output_path: Path,
    *,
    checkpoint_path: Path | None = None,
    device: str = "default",
    render: bool = False,
    verbose: bool = False,
    dry_run: bool = False,
) -> int:
    """Run SHARP prediction through uvx."""
    command = build_predict_command(
        input_path,
        output_path,
        checkpoint_path=checkpoint_path,
        device=device,
        render=render,
        verbose=verbose,
    )

    if dry_run:
        print(" ".join(command))
        return 0

    completed = subprocess.run(command, check=False)
    return completed.returncode


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run Apple's SHARP model on an image or image directory.",
    )
    parser.add_argument(
        "-i",
        "--input-path",
        type=Path,
        required=True,
        help="Path to an input image or a directory of images.",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=Path,
        required=True,
        help="Directory where SHARP should write .ply Gaussian splats.",
    )
    parser.add_argument(
        "-c",
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Optional local SHARP .pt checkpoint path.",
    )
    parser.add_argument(
        "--device",
        choices=("default", "cpu", "mps", "cuda"),
        default="default",
        help="Inference device passed through to SHARP.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Ask SHARP to render videos. SHARP currently requires CUDA for rendering.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose SHARP logging.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the uvx command without running it.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    return predict(
        args.input_path,
        args.output_path,
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        render=args.render,
        verbose=args.verbose,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    sys.exit(main())
