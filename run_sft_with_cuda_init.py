import torch
# Pre-initialize CUDA context to ensure bf16 detection works in transformers
torch.cuda.init()
print('CUDA initialized:', torch.cuda.get_device_name(0))

from swift.llm.train.sft import sft_main
sft_main()
