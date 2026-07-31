import torch
import sys
import os

# Pre-init CUDA to avoid bf16 detection issue
torch.cuda.init()

# Force DDP-like environment even for single GPU
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29529'
os.environ['WORLD_SIZE'] = '1'
os.environ['RANK'] = '0'
os.environ['LOCAL_RANK'] = '0'
os.environ['LOCAL_WORLD_SIZE'] = '1'

from swift.llm.train.sft import sft_main
sft_main()
