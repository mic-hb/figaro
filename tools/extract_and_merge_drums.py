#!/usr/bin/env python3
import argparse
from pathlib import Path

import pretty_midi


def is_drum_track(inst: pretty_midi.Instrument) -> bool:
  if inst.is_drum:
    return True
  name = (inst.name or "").lower()
  return ("drum" in name) or ("perc" in name)


def parse_args():
  parser = argparse.ArgumentParser(
    description="Extract drum tracks from generated MIDI and merge with original MIDI."
  )
  parser.add_argument("--original", required=True, help="Original MIDI (e.g. piano+bass)")
  parser.add_argument("--generated", required=True, help="Generated MIDI (full arrangement)")
  parser.add_argument("--output", required=True, help="Output merged MIDI path")
  parser.add_argument(
    "--fail_if_no_drums",
    action="store_true",
    help="Exit with non-zero status if generated file has no drum tracks"
  )
  return parser.parse_args()


def main():
  args = parse_args()

  original_path = Path(args.original).resolve()
  generated_path = Path(args.generated).resolve()
  output_path = Path(args.output).resolve()

  if not original_path.exists():
    raise FileNotFoundError(f"Original MIDI not found: {original_path}")
  if not generated_path.exists():
    raise FileNotFoundError(f"Generated MIDI not found: {generated_path}")

  original_pm = pretty_midi.PrettyMIDI(str(original_path))
  generated_pm = pretty_midi.PrettyMIDI(str(generated_path))
  drum_tracks = [inst for inst in generated_pm.instruments if is_drum_track(inst)]

  if not drum_tracks:
    message = "No drum tracks found in generated MIDI."
    if args.fail_if_no_drums:
      raise RuntimeError(message)
    print(message)
    return

  merged = pretty_midi.PrettyMIDI()
  for inst in original_pm.instruments:
    merged.instruments.append(inst)
  for inst in drum_tracks:
    merged.instruments.append(inst)

  output_path.parent.mkdir(parents=True, exist_ok=True)
  merged.write(str(output_path))

  print(f"Merged file written: {output_path}")
  print(f"Original tracks: {len(original_pm.instruments)}")
  print(f"Drum tracks added: {len(drum_tracks)}")
  print(f"Total tracks now: {len(merged.instruments)}")


if __name__ == "__main__":
  main()
