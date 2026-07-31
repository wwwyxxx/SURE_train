import transformers.training_args as ta
import inspect
src = inspect.getsource(ta.TrainingArguments.__post_init__)

# patch: print bf16 state before check
orig = ta.TrainingArguments.__post_init__
def patched(self):
    from transformers.utils import is_torch_bf16_gpu_available
    print(f'PATCH: bf16={self.bf16}, fp16={self.fp16}, bf16_full_eval={getattr(self,"bf16_full_eval",None)}, fp16_full_eval={getattr(self,"fp16_full_eval",None)}')
    print(f'PATCH: is_torch_bf16_gpu_available={is_torch_bf16_gpu_available()}')
    print(f'PATCH: device={getattr(self, "device", None)}')
    return orig(self)
ta.TrainingArguments.__post_init__ = patched

from swift.llm.train.sft import sft_main
sft_main()
