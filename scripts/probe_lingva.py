import requests, os
from dotenv import load_dotenv
load_dotenv()

L = os.environ.get('LINGVA_URL') or os.environ.get('LINGVANEX_API_HOST') or os.environ.get('LINGVA_URL')
if not L:
    # default used in translate_openai
    L = 'https://api-b2b.lingvanex.com/translate'
host = L.rsplit('/', 1)[0]
print('host', host)

try:
    r = requests.get(host, timeout=5)
    print('GET', r.status_code)
    print(r.text[:400])
except Exception as e:
    print('GET host error', type(e), e)

for path in ['/translate', '/api/translate', '/v1/translate', '/translate/text']:
    url = host + path
    try:
        headers = {'Authorization': f"Bearer {os.environ.get('LINGVA_API_KEY') or os.environ.get('LINGVA_API') or os.environ.get('LINGVA_API_KEY') or os.environ.get('LINGVA_API_KEY') or ''}"}
        r = requests.post(url, json={'text': 'hello', 'from': 'en', 'to': 'mn'}, headers=headers, timeout=5)
        print('POST', url, '->', r.status_code)
        print(r.text[:400])
    except Exception as e:
        print('POST', url, 'error', type(e), e)
