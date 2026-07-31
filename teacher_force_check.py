import sys
sys.path.insert(0, '/workspace/ms-swift')
import torch
from swift.llm import get_model_tokenizer, get_template, inference, get_dataset, SwiftPipeline
from swift.utils import get_logger

logger = get_logger()

# Load model with custom register
model_info = {}
model, tokenizer = get_model_tokenizer(
    model_type='qwen2_5_omni_custom',
    model_id_or_path='/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train/model/Qwen2.5-Omni-7B',
    model_kwargs={'device_map': 'cuda:0'},
    load_model=True,
)
# import external plugin
import importlib.util
spec = importlib.util.spec_from_file_location('plugin', '/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train/outputs/20260630-172621/custom/qwen2_5_omni_model_register.py')
plugin = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plugin)

# Actually the plugin registers the model globally, so get_model_tokenizer should pick it up
# But we need to call it via swift. Let's use the register path

print('Loaded')
