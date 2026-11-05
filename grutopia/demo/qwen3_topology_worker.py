"""Local JSON-lines worker for Qwen3 semantic-topology scoring.

Run this worker with an environment that has a recent Transformers version:

    $QWEN3_PYTHON grutopia/demo/qwen3_topology_worker.py --device cuda:6

Each input line must be ``{"request_id": ..., "prompt": ...}``.  One JSON
response is emitted per request, which keeps the Isaac Sim process isolated
from the model runtime and its newer PyTorch/Transformers dependencies.
"""

import argparse
import json
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='Qwen/Qwen3-8B')
    parser.add_argument('--device', default='cuda:6')
    parser.add_argument('--max-new-tokens', type=int, default=512)
    parser.add_argument('--local-files-only', action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def load_model(args):
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map={'': args.device},
        local_files_only=args.local_files_only,
    )
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    import torch

    messages = [
        {
            'role': 'system',
            'content': (
                'You rank robot exploration frontiers. Return only one flat JSON '
                'object mapping every candidate id to a numeric score in [0,1]. '
                'Include exactly the candidate ids and no other keys, arrays, '
                'markdown, or explanation.'
            ),
        },
        {'role': 'user', 'content': prompt},
    ]
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors='pt',
    ).to(model.device)
    with torch.inference_mode():
        output = model.generate(
            encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0, encoded.shape[-1] :], skip_special_tokens=True).strip()


def main():
    args = parse_args()
    model, tokenizer = load_model(args)
    print(json.dumps({'event': 'qwen3_ready', 'model': args.model, 'device': args.device}), flush=True)
    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        request_id = None
        try:
            request = json.loads(raw_line)
            request_id = request.get('request_id')
            response = {
                'request_id': request_id,
                'text': generate(
                    model,
                    tokenizer,
                    str(request['prompt']),
                    args.max_new_tokens,
                ),
            }
        except Exception as error:
            response = {
                'request_id': request_id,
                'error': f'{type(error).__name__}: {error}',
            }
        print(json.dumps(response), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
