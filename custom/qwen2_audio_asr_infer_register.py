import os
import sys
from typing import Any, Dict, Optional

# Reuse the model/template registrations from the training register.
sys.path.insert(0, '/workspace/outputs/20260629-183012/custom')
import qwen2_audio_dataset_register  # noqa: F401

from swift.llm import DatasetMeta, ResponsePreprocessor, register_dataset


class TestASRPreprocessor(ResponsePreprocessor):
    """Handles both aishell-style (wav/txt) and librispeech-style (path/target) rows.

    NOTE: ResponsePreprocessor.__init__ seeds self.columns with
    {'prompt': 'query', 'target': 'response', 'text': 'response', ...}, and
    RowPreprocessor.__call__ renames those dataset columns BEFORE preprocess()
    sees each row. So the 'prompt' field in the data files arrives as 'query'
    (silently unused here) and 'target' arrives as 'response'. This matches the
    training register (CombinedASRPreprocessor), which has the same behavior, so
    training and inference use the identical fallback prompt
    'Transcribe the speech to text:'. If you ever want the per-dataset prompt
    strings to take effect, read row.get('query') as well -- but then inference
    would diverge from what this checkpoint was trained with.
    """

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        wav = row.get('wav') or row.get('path') or row.get('audio') or row.get('audio_path')
        text = row.get('txt') or row.get('target') or row.get('text') or row.get('response')
        prompt = row.get('prompt') or 'Transcribe the speech to text:'
        if wav is None or text is None:
            return None
        if not os.path.isabs(wav):
            wav = os.path.abspath(wav)
        return {
            'messages': [
                {'role': 'user', 'content': f'<audio>{prompt}'},
                {'role': 'assistant', 'content': text},
            ],
            'audios': [wav],
        }


register_dataset(
    DatasetMeta(
        dataset_path='data/test/aishell1-test_ASR_infer.jsonl',
        dataset_name='aishell1_test_asr_infer',
        preprocess_func=TestASRPreprocessor(),
    ),
    exist_ok=True,
)

register_dataset(
    DatasetMeta(
        dataset_path='data/test/librispeech_test-clean_ASR.jsonl',
        dataset_name='librispeech_test_clean_asr',
        preprocess_func=TestASRPreprocessor(),
    ),
    exist_ok=True,
)

register_dataset(
    DatasetMeta(
        dataset_path='data/test/librispeech_test-other_ASR.jsonl',
        dataset_name='librispeech_test_other_asr',
        preprocess_func=TestASRPreprocessor(),
    ),
    exist_ok=True,
)
