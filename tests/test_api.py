"""
Quick check that OpenRouter API calls succeed (SSL, auth, model).

Run with:
    OPENROUTER_API_KEY=<your-key> uv run python tests/test_api.py
"""
import json
import os
import ssl
import urllib.request
import certifi


_URL   = 'https://openrouter.ai/api/v1/chat/completions'
_MODEL = 'nvidia/nemotron-3-super-120b-a12b:free'

def check_api(api_key=None):
    key = api_key or os.environ.get('OPENROUTER_API_KEY')
    if not key:
        raise ValueError('Set OPENROUTER_API_KEY env var or pass api_key=')

    payload = json.dumps({
        'model':      _MODEL,
        'messages':   [{'role': 'user', 'content': 'Tell me a joke.'}],
    }).encode('utf-8')

    req = urllib.request.Request(
        _URL,
        data    = payload,
        headers = {
            'Authorization': f'Bearer {key}',
            'Content-Type':  'application/json',
        },
        method = 'POST',
    )

    ssl_ctx = ssl.create_default_context(cafile=certifi.where())

    print(f'Sending request to {_URL} ...')
    with urllib.request.urlopen(req, context=ssl_ctx, timeout=30) as resp:
        data = json.loads(resp.read().decode('utf-8'))

    reply = data['choices'][0]['message']['content']
    print(f'Response: {reply!r}')
    print('API call succeeded.')
    return reply

if __name__ == '__main__':
    check_api()