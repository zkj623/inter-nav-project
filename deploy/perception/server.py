"""Reuse existing visual and topology model implementations."""
import sys


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'qwen3':
        topology_server(sys.argv[2:])
    else:
        from grutopia.demo.serve_semantic_perception import main as serve
        serve()


def topology_server(arguments):
    import argparse
    from flask import Flask, jsonify, request
    from grutopia.demo.qwen3_topology_worker import load_model, generate
    parser = argparse.ArgumentParser(description='Qwen3 topology scorer HTTP server')
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=12187)
    parser.add_argument('--max-new-tokens', type=int, default=512)
    args = parser.parse_args(arguments)
    args.local_files_only = True
    model, tokenizer = load_model(args)
    app = Flask('inter_nav_qwen3')
    app.config['MAX_CONTENT_LENGTH'] = 128*1024

    @app.get('/health')
    def health():
        return jsonify({'ready': True, 'model': args.model})

    @app.post('/generate')
    def inference():
        payload = request.get_json(force=True)
        prompt = payload.get('prompt') if isinstance(payload, dict) else None
        if not isinstance(prompt, str) or not 1 <= len(prompt) <= 64000:
            return jsonify({'error': 'Expected a prompt of 1..64000 characters'}), 400
        text = generate(model, tokenizer, prompt, args.max_new_tokens)
        return jsonify({'text': text, 'model': args.model})

    app.run(host=args.host, port=args.port, threaded=False)
