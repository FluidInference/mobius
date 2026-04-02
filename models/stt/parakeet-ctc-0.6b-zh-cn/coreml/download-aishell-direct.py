#!/usr/bin/env python3
"""Download AISHELL-1 test set directly from openslr.org.

Downloads and extracts the test partition of AISHELL-1 dataset.

Usage:
    uv run python download-aishell-direct.py --output-dir ./aishell1_test --num-files 100
"""
from __future__ import annotations

import json
import tarfile
import urllib.request
from pathlib import Path

import soundfile as sf
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, DownloadColumn, TransferSpeedColumn

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)
console = Console()

AISHELL_TEST_URL = "https://openslr.elda.org/resources/33/data_aishell.tgz"


@app.command()
def download(
    output_dir: Path = typer.Option(
        Path("aishell1_test"),
        help="Output directory for extracted test files",
    ),
    num_files: int = typer.Option(
        100,
        help="Number of test files to extract",
    ),
) -> None:
    """Download AISHELL-1 test set from openslr.org."""

    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / "data_aishell.tgz"

    # Download if not exists
    if not archive_path.exists():
        console.print(f"[cyan]Downloading AISHELL-1 dataset (~15GB)...[/cyan]")
        console.print("[yellow]This is a one-time download and may take 10-30 minutes[/yellow]\n")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Downloading...", total=None)

            def reporthook(block_num, block_size, total_size):
                if total_size > 0:
                    progress.update(task, total=total_size, completed=block_num * block_size)

            urllib.request.urlretrieve(AISHELL_TEST_URL, archive_path, reporthook)

        console.print(f"[green]✓[/green] Downloaded to {archive_path}")
    else:
        console.print(f"[green]✓[/green] Archive already exists: {archive_path}")

    # Extract test files
    console.print(f"\n[cyan]Extracting {num_files} test samples...[/cyan]")

    test_dir = output_dir / "test"
    test_dir.mkdir(exist_ok=True)

    manifest = []
    extracted_count = 0

    with tarfile.open(archive_path, "r:gz") as tar:
        # Find transcript file
        transcript_member = None
        for member in tar.getmembers():
            if "test/transcript/aishell_transcript_v0.8.txt" in member.name:
                transcript_member = member
                break

        if not transcript_member:
            console.print("[red]Error: Could not find transcript file in archive[/red]")
            return

        # Load transcripts
        console.print("Loading transcripts...")
        transcript_file = tar.extractfile(transcript_member)
        transcripts = {}
        for line in transcript_file.read().decode("utf-8").splitlines():
            if line.strip():
                parts = line.strip().split(None, 1)
                if len(parts) == 2:
                    file_id, text = parts
                    transcripts[file_id] = text

        console.print(f"Loaded {len(transcripts)} transcripts")

        # Extract test audio files
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Extracting audio...", total=num_files)

            for member in tar.getmembers():
                if extracted_count >= num_files:
                    break

                # Only extract test wav files
                if "/test/" in member.name and member.name.endswith(".wav"):
                    file_id = Path(member.name).stem

                    if file_id in transcripts:
                        # Extract to test directory
                        tar.extract(member, output_dir)
                        extracted_path = output_dir / member.name

                        # Copy to flat test directory
                        final_path = test_dir / f"{file_id}.wav"
                        if extracted_path.exists():
                            import shutil
                            shutil.copy(extracted_path, final_path)

                            # Load audio to get duration
                            audio, sr = sf.read(str(final_path))
                            duration = len(audio) / sr

                            manifest.append({
                                "audio_path": str(final_path),
                                "transcript": transcripts[file_id],
                                "sample_id": file_id,
                                "duration": duration,
                            })

                            extracted_count += 1
                            progress.update(task, advance=1)

    console.print(f"[green]✓[/green] Extracted {extracted_count} test samples")

    # Save manifest
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    console.print(f"[green]✓[/green] Saved manifest: {manifest_path}")

    # Print stats
    total_duration = sum(s["duration"] for s in manifest)
    console.print(f"\n[bold green]Dataset Ready![/bold green]")
    console.print(f"  Files: {len(manifest)}")
    console.print(f"  Total duration: {total_duration:.1f}s ({total_duration/60:.1f}min)")
    console.print(f"  Test directory: {test_dir}")


if __name__ == "__main__":
    app()
