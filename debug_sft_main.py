import sys
sys.path.insert(0, '/workspace/ms-swift')
from swift.llm.train.sft import SwiftSft
from swift.llm.argument.train_args import TrainArguments
from swift.trainers import TrainerFactory
from swift.utils import parse_args

args_list = sys.argv[1:]
print('argv:', args_list)
args, remaining = parse_args(TrainArguments, args_list)
print('bf16:', args.bf16, 'fp16:', args.fp16)
print('remaining:', remaining)
try:
    ta = TrainerFactory.get_training_args(args)
    print('OK training_args bf16:', ta.bf16)
except Exception as e:
    print('ERROR:', e)
