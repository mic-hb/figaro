#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path

import pretty_midi

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
  sys.path.insert(0, str(SRC_DIR))

from input_representation import InputRepresentation

def count_bars(midi_path: Path) -> int:
  # Preferred path: use FIGARO's own REMI conversion, then count Bar tokens.
  # This aligns bar counting with what the model actually sees.
  try:
    rep = InputRepresentation(str(midi_path), strict=True)
    events = rep.get_remi_events()
    remi_bars = sum(1 for token in events if token.startswith("Bar_"))
    if remi_bars > 0:
      return remi_bars
  except Exception:
    pass

  # Fallback path: estimate from PrettyMIDI downbeats/tempo.
  pm = pretty_midi.PrettyMIDI(str(midi_path))
  downbeats = pm.get_downbeats()
  if len(downbeats) >= 2:
    return len(downbeats) - 1
  end_time = pm.get_end_time()
  if end_time <= 0:
    return 1
  bpm = pm.estimate_tempo() or 120.0
  beats_per_second = bpm / 60.0
  beats_per_bar = 4.0
  estimated_bars = int(max(1, round((end_time * beats_per_second) / beats_per_bar)))
  return estimated_bars


def count_bars_remi(midi_path: Path) -> int:
  rep = InputRepresentation(str(midi_path), strict=True)
  events = rep.get_remi_events()
  return sum(1 for token in events if token.startswith("Bar_"))


def build_output_subdir(model: str, max_iter: int, max_bars: int, initial_context: int) -> str:
  ctx_tag = "full" if initial_context < 0 else str(initial_context)
  params = f"max_iter={max_iter},max_bars={max_bars},initial_context={ctx_tag}"
  return os.path.join(model, params)


def parse_args():
  parser = argparse.ArgumentParser(
    description="Continue a MIDI file by exactly N bars using FIGARO."
  )
  parser.add_argument("--input_midi", required=True, help="Path to primer MIDI file")
  parser.add_argument("--continue_bars", type=int, required=True, help="How many bars to add")
  parser.add_argument("--model", default="figaro-expert")
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--vae_checkpoint", default=None)
  parser.add_argument("--generator_script", default="src/generate_with_context.py")
  parser.add_argument("--work_dir", default=".")
  parser.add_argument("--output_dir", default="./samples")
  parser.add_argument("--max_iter", type=int, default=4000)
  parser.add_argument("--temp", type=float, default=0.8, help="Sampling temperature")
  parser.add_argument(
    "--attempts_per_round",
    type=int,
    default=4,
    help="Sampling retries per round (best bar extension is kept)"
  )
  parser.add_argument(
    "--max_rounds",
    type=int,
    default=5,
    help="Maximum iterative continuation rounds to reach target bars"
  )
  parser.add_argument(
    "--initial_context",
    type=int,
    default=-1,
    help="Use -1 for full prompt continuation"
  )
  parser.add_argument("--verbose", type=int, default=2)
  parser.add_argument("--keep_temp", action="store_true")
  return parser.parse_args()


def main():
  args = parse_args()
  work_dir = Path(args.work_dir).resolve()
  input_midi = Path(args.input_midi).resolve()
  if not input_midi.exists():
    raise FileNotFoundError(f"Input MIDI not found: {input_midi}")
  if args.continue_bars <= 0:
    raise ValueError("--continue_bars must be > 0")
  if args.max_rounds <= 0:
    raise ValueError("--max_rounds must be > 0")
  if args.attempts_per_round <= 0:
    raise ValueError("--attempts_per_round must be > 0")

  input_bars = count_bars(input_midi)
  target_bars = input_bars + args.continue_bars
  print(f"Detected input bars: {input_bars}")
  print(f"Target total bars: {target_bars} (input + {args.continue_bars})")

  temp_input_dir = work_dir / "data" / "inference" / "_tmp_continue"
  temp_input_dir.mkdir(parents=True, exist_ok=True)
  temp_input = temp_input_dir / input_midi.name
  temp_input.write_bytes(input_midi.read_bytes())

  output_subdir = build_output_subdir(args.model, args.max_iter, target_bars, args.initial_context)
  output_midi = Path(args.output_dir) / output_subdir / input_midi.name
  current_input = input_midi
  current_bars = input_bars

  for round_idx in range(1, args.max_rounds + 1):
    temp_input.write_bytes(current_input.read_bytes())
    cmd = [
      sys.executable,
      args.generator_script,
      "--model", args.model,
      "--checkpoint", args.checkpoint,
      "--lmd_dir", str(temp_input_dir),
      "--output_dir", args.output_dir,
      "--max_n_files", "1",
      "--max_bars", str(target_bars),
      "--max_iter", str(args.max_iter),
      "--temp", str(args.temp),
      "--max_attempts", str(args.attempts_per_round),
      "--initial_context", str(args.initial_context),
      "--verbose", str(args.verbose),
    ]
    if args.vae_checkpoint:
      cmd.extend(["--vae_checkpoint", args.vae_checkpoint])

    print(f"[round {round_idx}] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(work_dir))
    print(f"[round {round_idx}] Expected generated MIDI: {output_midi}")
    if not output_midi.exists():
      raise FileNotFoundError(f"Expected generated MIDI not found: {output_midi}")

    generated_bars = count_bars_remi(output_midi)
    print(f"[round {round_idx}] Generated REMI bars: {generated_bars} / target: {target_bars}")
    if generated_bars >= target_bars:
      print("Done. Target bar count reached.")
      break
    if generated_bars <= current_bars:
      print("Stopped early: model did not increase bars in this round.")
      print("Try a different checkpoint/model or larger --max_iter.")
      break

    current_input = output_midi
    current_bars = generated_bars

  if not args.keep_temp:
    try:
      temp_input.unlink(missing_ok=True)
      temp_input_dir.rmdir()
    except OSError:
      # Keep directory if not empty.
      pass


if __name__ == "__main__":
  main()
