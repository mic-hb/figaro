# FIGARO Getting Started: Generation and Inference (Beginner Guide)

This guide explains, step by step, how to run FIGARO for generation/inference in this repository.

It includes two practical use cases:

1. You have a MIDI file with 1 piano track and 4 bars, and you want continuation for bars 5-8.
2. You have a MIDI file with 2 tracks (piano + bass), and you want to create a drum track conditioned on those tracks.

---

## 1) What FIGARO generation does in this repo

Generation is implemented in `src/generate.py`.

At a high level:

- It loads a trained checkpoint (`--checkpoint` and optional `--vae_checkpoint`).
- It scans all `.mid` files under `--lmd_dir`.
- It builds internal token/description representation.
- It generates output MIDI and writes it into `--output_dir`.

Important behavior in current code:

- The default generation entry uses very short initial context (`initial_context=1`), so out of the box it is not strict long-prompt continuation.
- It can condition on description/latent features depending on model variant.
- It does not provide direct "track inpainting" command like "keep piano+bass untouched and only write drums."

You can still do both target scenarios with either:

- a small and safe continuation tweak (for use case 1),
- and a post-processing merge workflow (for use case 2).

---

## 2) Prerequisites

From this repo's `README.md`, FIGARO expects Python 3.9.

Since you do not want Conda, this guide uses Ubuntu apt + `venv`.

### 2.1 Install Python 3.9

```bash
sudo apt update
sudo apt install -y python3.9 python3.9-venv python3.9-dev
```

Sample output (example):

```text
...
Setting up python3.9 (3.9.25-1+noble1) ...
Setting up python3.9-venv (3.9.25-1+noble1) ...
Setting up python3.9-dev (3.9.25-1+noble1) ...
```

### 2.2 Create virtual environment and install dependencies

```bash
cd /home/michb/dev/01-ISTTS/auto-midi/lib/figaro
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Sample output (example):

```text
Successfully installed numpy-1.26.4 pandas-2.2.3 pretty-midi-0.2.10 ...
```

### 2.3 Verify environment

```bash
python -V
python -c "import torch, sklearn, pandas, numpy, pretty_midi; print('imports ok')"
```

Sample output (example):

```text
Python 3.9.25
imports ok
```

---

## 3) Download checkpoints

If you do not have trained checkpoints yet, download pre-trained ones:

```bash
cd /home/michb/dev/01-ISTTS/auto-midi/lib/figaro
wget -O checkpoints.zip https://polybox.ethz.ch/index.php/s/a0HUHzKuPPefWkW/download
unzip checkpoints.zip
```

Check what is available:

```bash
ls -lh checkpoints
```

Sample output (example):

```text
figaro-expert.ckpt
figaro.ckpt
vq-vae.ckpt
...
```

Notes:

- `figaro-expert` uses only expert description (no VAE checkpoint needed).
- `figaro` and `figaro-learned` typically require `--vae_checkpoint`.

---

## 4) Run a basic generation smoke test

Create test input folder:

```bash
mkdir -p data/inference/basic
cp /path/to/your_test.mid data/inference/basic/
```

Run generation:

```bash
python src/generate.py \
  --model figaro-expert \
  --checkpoint ./checkpoints/figaro-expert.ckpt \
  --lmd_dir ./data/inference/basic \
  --output_dir ./samples \
  --max_n_files 1 \
  --max_bars 8 \
  --max_iter 4000 \
  --verbose 2
```

Sample output (example):

```text
Saving generated files to: ./samples/figaro-expert/max_iter=4000,max_bars=8
Generating sequence (1 initial / 4001 max length / 8 max bars / 1 batch size)
Saving to ./samples/figaro-expert/max_iter=4000,max_bars=8/your_test.mid
```

Output location:

- Generated file: `samples/.../your_test.mid`
- Original reference: `samples/.../ground_truth/your_test.mid`

---

## 5) Use case 1: piano 4 bars -> continue bars 5-8

## 5.1 Why tweak is needed

Default script seeds with only a tiny initial context. For real continuation from your existing bars, you should pass full prompt length as initial context.

## 5.2 Minimal continuation tweak in `src/generate.py`

In `main()`, find this block:

```python
for batch in dl:
  reconstruct_sample(model, batch,
    output_dir=output_dir,
    max_iter=args.max_iter,
    max_bars=max_bars,
    verbose=args.verbose,
  )
```

Change it to:

```python
for batch in dl:
  reconstruct_sample(model, batch,
    initial_context=batch['input_ids'].shape[1],
    output_dir=output_dir,
    max_iter=args.max_iter,
    max_bars=max_bars,
    verbose=args.verbose,
  )
```

What this does:

- Uses your entire tokenized 4-bar prompt as seed.
- Model continues after that prompt instead of starting nearly from scratch.

## 5.3 Prepare your continuation input

Put your MIDI file in:

```text
lib/figaro/data/inference/continuation/piano_4bars.mid
```

## 5.4 Run continuation command

```bash
cd /home/michb/dev/01-ISTTS/auto-midi/lib/figaro
source .venv/bin/activate

python src/generate.py \
  --model figaro-expert \
  --checkpoint ./checkpoints/figaro-expert.ckpt \
  --lmd_dir ./data/inference/continuation \
  --output_dir ./samples \
  --max_n_files 1 \
  --max_bars 8 \
  --max_iter 4000 \
  --verbose 2
```

Sample output (example):

```text
Saving generated files to: ./samples/figaro-expert/max_iter=4000,max_bars=8
Generating sequence (512 initial / 4512 max length / 8 max bars / 1 batch size)
Saving to ./samples/figaro-expert/max_iter=4000,max_bars=8/piano_4bars.mid
```

Note: `512 initial` is just an example token count; yours can differ.

## 5.5 Validate result

Open the generated MIDI in your DAW/editor:

- bars 1-4 should reflect your provided musical context,
- bars 5-8 should be newly generated continuation.

---

## 6) Use case 2: piano+bass -> generate drums conditioned on them

## 6.1 Important limitation

This repo does not expose direct strict track inpainting from CLI ("generate only drums, freeze all other tracks exactly").

Practical workflow:

1. Generate full arrangement from your piano+bass conditioning input.
2. Extract drum track(s) from generated output.
3. Merge extracted drums into original piano+bass MIDI.

## 6.2 Prepare conditioning MIDI

Put your file here:

```text
lib/figaro/data/inference/drum_conditioning/piano_bass.mid
```

## 6.3 Run generation

```bash
python src/generate.py \
  --model figaro-expert \
  --checkpoint ./checkpoints/figaro-expert.ckpt \
  --lmd_dir ./data/inference/drum_conditioning \
  --output_dir ./samples \
  --max_n_files 1 \
  --max_bars 16 \
  --max_iter 6000 \
  --verbose 2
```

Sample output (example):

```text
Saving generated files to: ./samples/figaro-expert/max_iter=6000,max_bars=16
Generating sequence (1 initial / 6001 max length / 16 max bars / 1 batch size)
Saving to ./samples/figaro-expert/max_iter=6000,max_bars=16/piano_bass.mid
```

## 6.4 Extract generated drums and merge with original

Create helper script `tools/extract_and_merge_drums.py`:

```python
import argparse
import pretty_midi

def is_drum_track(inst: pretty_midi.Instrument) -> bool:
    if inst.is_drum:
        return True
    name = (inst.name or "").lower()
    return "drum" in name or "perc" in name

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--original", required=True, help="Original piano+bass midi")
    p.add_argument("--generated", required=True, help="Generated full arrangement midi")
    p.add_argument("--output", required=True, help="Merged output midi")
    args = p.parse_args()

    orig = pretty_midi.PrettyMIDI(args.original)
    gen = pretty_midi.PrettyMIDI(args.generated)

    drum_insts = [i for i in gen.instruments if is_drum_track(i)]
    if not drum_insts:
        print("No drum track found in generated file.")
        return

    merged = pretty_midi.PrettyMIDI()
    for inst in orig.instruments:
        merged.instruments.append(inst)
    for inst in drum_insts:
        merged.instruments.append(inst)

    merged.write(args.output)
    print(f"Merged file written: {args.output}")
    print(f"Original tracks: {len(orig.instruments)}")
    print(f"Drum tracks added: {len(drum_insts)}")
    print(f"Total tracks now: {len(merged.instruments)}")

if __name__ == "__main__":
    main()
```

Run:

```bash
python tools/extract_and_merge_drums.py \
  --original ./data/inference/drum_conditioning/piano_bass.mid \
  --generated ./samples/figaro-expert/max_iter=6000,max_bars=16/piano_bass.mid \
  --output ./samples/merged/piano_bass_plus_drums.mid
```

Sample output (example):

```text
Merged file written: ./samples/merged/piano_bass_plus_drums.mid
Original tracks: 2
Drum tracks added: 1
Total tracks now: 3
```

---

## 7) Recommended command templates

### 7.1 Fast sanity check

```bash
python src/generate.py \
  --model figaro-expert \
  --checkpoint ./checkpoints/figaro-expert.ckpt \
  --lmd_dir ./data/inference/basic \
  --output_dir ./samples \
  --max_n_files 1 \
  --max_bars 8 \
  --max_iter 3000 \
  --verbose 2
```

### 7.2 Longer generation

```bash
python src/generate.py \
  --model figaro-expert \
  --checkpoint ./checkpoints/figaro-expert.ckpt \
  --lmd_dir ./data/inference/basic \
  --output_dir ./samples \
  --max_n_files 1 \
  --max_bars 32 \
  --max_iter 16000 \
  --verbose 2
```

### 7.3 Model with learned description (requires VAE checkpoint)

```bash
python src/generate.py \
  --model figaro \
  --checkpoint ./checkpoints/figaro.ckpt \
  --vae_checkpoint ./checkpoints/vq-vae.ckpt \
  --lmd_dir ./data/inference/basic \
  --output_dir ./samples \
  --max_n_files 1 \
  --max_bars 16 \
  --max_iter 6000 \
  --verbose 2
```

---

## 8) Troubleshooting

### 8.1 `python3.9: command not found`

Install Python 3.9 first (section 2.1).

### 8.2 checkpoint file missing

Verify with:

```bash
ls checkpoints
```

Then correct `--checkpoint` path.

### 8.3 generation is very slow

- CPU fallback is normal if CUDA is unavailable.
- Reduce `--max_iter` and `--max_bars` for quick tests.

### 8.4 no output MIDI generated

Check:

- `--lmd_dir` actually contains `.mid` files,
- files are valid MIDI and readable,
- output path permissions are valid.

### 8.5 output quality not good

Try:

- different model variants (`figaro-expert` vs `figaro`),
- larger generation budget (`max_iter`, `max_bars`),
- better quality input MIDI (clean timing, clear structure).

### 8.6 `ModuleNotFoundError: No module named 'pkg_resources'`

This usually means your active Python environment does not have the right `setuptools` behavior for this older Lightning stack.

Fix inside FIGARO venv:

```bash
cd /home/michb/dev/01-ISTTS/auto-midi/lib/figaro
./.venv/bin/python -m pip install "setuptools<81"
./.venv/bin/python -c "import pkg_resources; import pytorch_lightning; print('ok in venv')"
```

Sample output (example):

```text
Successfully installed setuptools-80.10.2
ok in venv
```

Notes:

- The warning about `pkg_resources` being deprecated is expected here.
- The important thing is that import works and generation runs.

### 8.7 Wrong Python environment is used (global Python instead of `.venv`)

This issue is very common and causes confusing behavior.

Always run generation with one of these two safe patterns:

Pattern A (activate then run):

```bash
cd /home/michb/dev/01-ISTTS/auto-midi/lib/figaro
source .venv/bin/activate
python src/generate.py ...
```

Pattern B (no activation, explicit interpreter):

```bash
cd /home/michb/dev/01-ISTTS/auto-midi/lib/figaro
./.venv/bin/python src/generate.py ...
```

Verification command:

```bash
cd /home/michb/dev/01-ISTTS/auto-midi/lib/figaro
./.venv/bin/python -c "import sys; print(sys.executable)"
```

Sample output (example):

```text
/home/michb/dev/01-ISTTS/auto-midi/lib/figaro/.venv/bin/python
```

### 8.8 Generation says success, but `samples/` is empty

This can happen when your inference set is very small (for example 1 file).

Why:

- The original `generate.py` uses a test split internally.
- For tiny datasets, `n_test = int(0.05 * N)` can become `0`.
- Result: no test files are processed, so no MIDI is written.

Fix:

- Use `src/generate_with_context.py` from this guide (already patched for this case), or
- patch original generation code to fall back to all discovered files when test split is empty.

### 8.9 4-bar continuation is detected as 3 bars

This is a bar counting mismatch problem, usually due to downbeat metadata.

Why:

- DAWs and `pretty_midi.get_downbeats()` can disagree on bar boundaries.
- If you compute bars as `len(downbeats)-1`, some files are undercounted by one.

Fix:

- Use REMI bar tokens for counting (same representation FIGARO uses internally).
- `tools/continue_n_bars.py` in this guide already does that with `InputRepresentation.get_remi_events()`.

### 8.10 Continuation stops at bar 4 or 5

This is the most important caveat for continuation.

Observed behavior:

- The model often emits EOS very early.
- You get either no extension (bar 4) or only one extra bar (bar 5).

Why:

- FIGARO is trained for description-to-sequence conditional generation.
- It is not optimized as a strict "continue exactly N bars" model.
- With a short primer and weak future conditioning, EOS probability can be high.

What helps:

- Excluding terminal EOS from full-context prompt (`initial_context=-1` should use `seq_len-1` tokens).
- Multiple sampling attempts and rounds.
- Constrained decoding (see section 10.3): block EOS until a minimum bar target is reached.

### 8.11 Device mismatch in `figaro` + `vq-vae` mode (`cpu` vs `cuda`)

Symptom:

```text
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0
```

Cause:

- Latent conditioning tensors remain on CPU while model weights are on GPU.

Fix:

- Ensure latent tensors are moved to model device before sampling.
- In this guide's `generate_with_context.py`, this is already handled.

---

## 9) Practical expectations

What works well:

- end-to-end generation from conditioning inputs,
- continuation-style generation with a small context tweak,
- arrangement workflows where post-processing merges selected generated tracks.

What is not exposed directly in current CLI:

- strict constrained track-only infill with hard freezing of all other tracks.

For strict inpainting behavior, further code customization is needed in sampling/conditioning logic.

---

## 10) New helper scripts and safer workflow

During practical testing, several helper scripts and safety features were added. Prefer this workflow over directly calling `src/generate.py` for continuation tasks.

### 10.1 `src/generate_with_context.py` (cloned + enhanced generator)

What it adds:

- `--initial_context` (supports `-1` for full prompt without terminal EOS),
- tiny-dataset fallback (still generates when test split is empty),
- `--temp`, `--max_attempts` sampling retries,
- `--force_min_bars` constrained decoding option.

Example:

```bash
./.venv/bin/python src/generate_with_context.py \
  --model figaro \
  --checkpoint ./checkpoints/figaro.ckpt \
  --vae_checkpoint ./checkpoints/vq-vae.ckpt \
  --lmd_dir ./data/inference/continuation \
  --output_dir ./samples \
  --max_n_files 1 \
  --max_bars 8 \
  --max_iter 1600 \
  --initial_context -1 \
  --temp 1.1 \
  --max_attempts 10 \
  --force_min_bars 8 \
  --verbose 1
```

### 10.2 `tools/continue_n_bars.py` (automatic "continue exactly N bars")

What it does:

- detects current bars from REMI representation,
- computes target bars (`input + N`),
- runs iterative continuation rounds,
- retries multiple stochastic samples per round,
- reports progress with REMI bar counts.

Example:

```bash
./.venv/bin/python tools/continue_n_bars.py \
  --input_midi ./data/inference/continuation/piano-4bars.mid \
  --continue_bars 4 \
  --model figaro \
  --checkpoint ./checkpoints/figaro.ckpt \
  --vae_checkpoint ./checkpoints/vq-vae.ckpt \
  --work_dir . \
  --output_dir ./samples \
  --max_iter 1600 \
  --temp 1.1 \
  --attempts_per_round 10 \
  --max_rounds 6 \
  --initial_context -1 \
  --verbose 1
```

### 10.3 Safe constrained decoding (`--force_min_bars`)

This is the most reliable practical fix for early EOS.

How it works:

- EOS token is blocked during sampling until the sequence reaches `min_bars`.
- After that threshold, EOS is allowed again.
- This does not guarantee perfect musical quality, but it reliably prevents premature stopping at bar 4/5.

### 10.4 `tools/extract_and_merge_drums.py` (drum extraction + merge)

Use this for the "piano+bass -> add drums" workflow:

```bash
./.venv/bin/python tools/extract_and_merge_drums.py \
  --original ./data/inference/drum_conditioning/piano_bass.mid \
  --generated ./samples/figaro/max_iter=1600,max_bars=16,initial_context=full/piano_bass.mid \
  --output ./samples/merged/piano_bass_plus_drums.mid
```

---

## 11) Honest capability summary (important)

Based on paper framing + repository behavior:

- FIGARO **is autoregressive at decoder token level**, but
- it is primarily trained for **description-to-sequence conditional generation**, not strict continuation-to-exact-length.

Therefore:

- Continuation is possible, but may stop early without constrained decoding.
- Strict in-filling (segment/track freezing with guaranteed preservation) is not exposed as a native CLI feature in this repo.
- For production continuation targets, use:
  1) `generate_with_context.py`,
  2) retry rounds,
  3) `--force_min_bars` constraint,
  4) explicit REMI bar-count verification.

