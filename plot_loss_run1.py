#!/usr/bin/env python3
"""Plot training loss curve from ms-swift logging.jsonl."""
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SRC = 'output/run1/v0-20260624-032554/logging.jsonl'
OUT = 'output/run1/loss_curve.png'

steps, losses = [], []
with open(SRC) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        # keep only per-step metric lines
        if 'loss' not in d or 'global_step/max_steps' not in d:
            continue
        try:
            step = int(str(d['global_step/max_steps']).split('/')[0])
        except Exception:
            continue
        steps.append(step)
        losses.append(float(d['loss']))

steps = np.array(steps)
losses = np.array(losses)

fig, ax = plt.subplots(figsize=(11, 5.5), dpi=150)
ax.plot(steps, losses, color='#1f5fa8', linewidth=0.8, label='loss')
ax.set_xlabel('global step')
ax.set_ylabel('loss')
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right')

fig.tight_layout()
plt.savefig(OUT)
print(f'steps={len(steps)} range=[{steps.min()},{steps.max()}] '
      f'loss: {losses[0]:.3f} -> {losses[-1]:.3f} (min {losses.min():.3f})')
print(f'saved: {OUT}')
