import argparse
import glob
import os
import random

import torch
from torch.utils.data import DataLoader
from transformers.models.bert.modeling_bert import BertAttention

from datasets import MidiDataset, SeqCollator
from input_representation import remi2midi
from models.seq2seq import Seq2SeqModule
from models.vae import VqVaeModule
from utils import medley_iterator


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', type=str, default="figaro-expert")
  parser.add_argument('--checkpoint', type=str, default="../figaro-expert.ckpt")
  parser.add_argument('--vae_checkpoint', type=str, default=None, help="Path to the VQ-VAE model checkpoint (optional)")
  parser.add_argument('--lmd_dir', type=str, default='./lmd_full', help="Path to a folder with .mid files")
  parser.add_argument('--output_dir', type=str, default='./samples', help="Path to the output directory")
  parser.add_argument('--max_n_files', type=int, default=-1)
  parser.add_argument('--max_iter', type=int, default=16_000)
  parser.add_argument('--max_bars', type=int, default=32)
  parser.add_argument('--make_medleys', type=bool, default=False)
  parser.add_argument('--n_medley_pieces', type=int, default=2)
  parser.add_argument('--n_medley_bars', type=int, default=16)
  parser.add_argument('--batch_size', type=int, default=1)
  parser.add_argument('--verbose', type=int, default=2)
  parser.add_argument('--temp', type=float, default=0.8, help="Sampling temperature")
  parser.add_argument('--max_attempts', type=int, default=1, help="Retries per sample to reach target bars")
  parser.add_argument(
    '--initial_context',
    type=int,
    default=1,
    help="How many initial tokens to keep from the prompt. Use -1 to use the full prompt."
  )
  return parser.parse_args()


def load_old_or_new_checkpoint(model_class, checkpoint):
  pl_ckpt = torch.load(checkpoint, map_location="cpu")
  kwargs = pl_ckpt['hyper_parameters']
  if 'flavor' in kwargs:
    del kwargs['flavor']
  if 'vae_run' in kwargs:
    del kwargs['vae_run']
  model = model_class(**kwargs)
  state_dict = pl_ckpt['state_dict']
  state_dict = {k: v for k, v in state_dict.items() if not k.endswith('embeddings.position_ids')}
  try:
    model.load_state_dict(state_dict)
  except RuntimeError:
    config = model.transformer.decoder.bert.config
    for layer in model.transformer.decoder.bert.encoder.layer:
      layer.crossattention = BertAttention(config, position_embedding_type=config.position_embedding_type)
    model.load_state_dict(state_dict)
  model.freeze()
  model.eval()
  return model


def load_model(checkpoint, vae_checkpoint=None, device='auto'):
  if device == 'auto':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  vae_module = None
  if vae_checkpoint:
    vae_module = load_old_or_new_checkpoint(VqVaeModule, vae_checkpoint)
    vae_module.cpu()

  model = load_old_or_new_checkpoint(Seq2SeqModule, checkpoint)
  model.to(device)
  return model, vae_module


@torch.no_grad()
def reconstruct_sample(
  model,
  batch,
  initial_context=1,
  output_dir=None,
  max_iter=-1,
  max_bars=-1,
  temp=0.8,
  max_attempts=1,
  verbose=0,
):
  batch_size, seq_len = batch['input_ids'].shape[:2]
  if initial_context < 0:
    # Use full prompt but exclude terminal EOS token, otherwise decoding
    # often stops immediately (model predicts EOS right away).
    context_tokens = max(1, seq_len - 1)
  else:
    context_tokens = min(initial_context, seq_len)

  batch_ = {key: batch[key][:, :context_tokens] for key in ['input_ids', 'bar_ids', 'position_ids']}
  if model.description_flavor in ['description', 'both']:
    batch_['description'] = batch['description']
    batch_['desc_bar_ids'] = batch['desc_bar_ids']
  if model.description_flavor in ['latent', 'both']:
    # For latent/both flavors, keep latent tensors on model device to avoid
    # cpu/cuda mismatch in cross-attention encoding.
    batch_['latents'] = batch['latents'].to(model.device)

  max_len = seq_len + 1024
  if max_iter > 0:
    max_len = min(max_len, context_tokens + max_iter)
  if verbose:
    print(f"Generating sequence ({context_tokens} initial / {max_len} max length / {max_bars} max bars / {batch_size} batch size)")

  best_sample = None
  best_bar_reached = -1
  attempts = max(1, max_attempts)
  for attempt in range(attempts):
    sample = model.sample(
      batch_,
      max_length=max_len,
      max_bars=max_bars,
      temp=temp,
      verbose=verbose // 2
    )
    sample_max_bar = int(sample['bar_ids'].max().item())
    if sample_max_bar > best_bar_reached:
      best_bar_reached = sample_max_bar
      best_sample = sample
    if max_bars > 0 and sample_max_bar >= max_bars:
      if verbose:
        print(f"Reached target bars on attempt {attempt + 1}/{attempts} (max bar id: {sample_max_bar})")
      break
    if verbose and attempts > 1:
      print(f"Attempt {attempt + 1}/{attempts} ended at max bar id: {sample_max_bar}")

  sample = best_sample

  xs = batch['input_ids'].detach().cpu()
  xs_hat = sample['sequences'].detach().cpu()
  events = [model.vocab.decode(x) for x in xs]
  events_hat = [model.vocab.decode(x) for x in xs_hat]

  pms, pms_hat = [], []
  for rec, rec_hat in zip(events, events_hat):
    try:
      pm = remi2midi(rec)
      pms.append(pm)
    except Exception as err:
      print("ERROR: Could not convert events to midi:", err)
    try:
      pm_hat = remi2midi(rec_hat)
      pms_hat.append(pm_hat)
    except Exception as err:
      print("ERROR: Could not convert events to midi:", err)

  if output_dir:
    os.makedirs(os.path.join(output_dir, 'ground_truth'), exist_ok=True)
    for pm, pm_hat, file in zip(pms, pms_hat, batch['files']):
      if verbose:
        print(f"Saving to {output_dir}/{file}")
      pm.write(os.path.join(output_dir, 'ground_truth', file))
      pm_hat.write(os.path.join(output_dir, file))

  return events


def build_output_dir(args):
  if not args.output_dir:
    raise ValueError("args.output_dir must be specified.")

  params = []
  if args.make_medleys:
    params.append(f"n_pieces={args.n_medley_pieces}")
    params.append(f"n_bars={args.n_medley_bars}")
  if args.max_iter > 0:
    params.append(f"max_iter={args.max_iter}")
  if args.max_bars > 0:
    params.append(f"max_bars={args.max_bars}")
  if args.initial_context >= 0:
    params.append(f"initial_context={args.initial_context}")
  else:
    params.append("initial_context=full")
  return os.path.join(args.output_dir, args.model, ','.join(params))


def main():
  args = parse_args()
  if args.make_medleys:
    max_bars = args.n_medley_pieces * args.n_medley_bars
  else:
    max_bars = args.max_bars

  output_dir = build_output_dir(args)
  print(f"Saving generated files to: {output_dir}")

  model, vae_module = load_model(args.checkpoint, args.vae_checkpoint)

  midi_files = glob.glob(os.path.join(args.lmd_dir, '**/*.mid'), recursive=True)
  if len(midi_files) == 0:
    raise ValueError(f"No MIDI files found under --lmd_dir: {args.lmd_dir}")

  dm = model.get_datamodule(midi_files, vae_module=vae_module)
  dm.setup('test')
  test_files = dm.test_ds.files
  midi_files = test_files if len(test_files) > 0 else midi_files
  random.shuffle(midi_files)

  if args.max_n_files > 0:
    midi_files = midi_files[:args.max_n_files]

  description_options = None
  if args.model in ['figaro-no-inst', 'figaro-no-chord', 'figaro-no-meta']:
    description_options = model.description_options

  dataset = MidiDataset(
    midi_files,
    max_len=-1,
    description_flavor=model.description_flavor,
    description_options=description_options,
    max_bars=model.context_size,
    vae_module=vae_module
  )

  coll = SeqCollator(context_size=-1)
  dl = DataLoader(dataset, batch_size=args.batch_size, collate_fn=coll)

  if args.make_medleys:
    dl = medley_iterator(
      dl,
      n_pieces=args.n_medley_pieces,
      n_bars=args.n_medley_bars,
      description_flavor=model.description_flavor
    )

  with torch.no_grad():
    for batch in dl:
      reconstruct_sample(
        model,
        batch,
        initial_context=args.initial_context,
        output_dir=output_dir,
        max_iter=args.max_iter,
        max_bars=max_bars,
        temp=args.temp,
        max_attempts=args.max_attempts,
        verbose=args.verbose,
      )


if __name__ == '__main__':
  main()
