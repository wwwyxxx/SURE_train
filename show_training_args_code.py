from transformers import training_args
import inspect
src = inspect.getsource(training_args.TrainingArguments.__post_init__)
lines = src.split('\n')
for i, line in enumerate(lines):
    if 1730 <= i <= 1760:
        print(i, line)
