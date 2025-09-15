import requests, os, json
from dotenv import load_dotenv
load_dotenv()

host = os.environ.get('LINGVANEX_API_HOST') or os.environ.get('LINGVA_URL') or os.environ.get('LINGVA_HOST') or 'https://api-b2b.backenster.com'
if not host.startswith('http'):
    host = 'https://' + host
print('base host:', host)

paths = ['','/translate','/api/translate','/v1/translate','/v2/translate','/translate/text','/api/v3/translate','/v3/translate','/api','/']
key = os.environ.get('LINGVA_API_KEY') or os.environ.get('LINGVA_API') or os.environ.get('LINGVA')
print('have key:', bool(key))

for p in paths:
    url = host.rstrip('/') + p
    print('\nTRYING', url)
    # try GET without key
    try:
        r = requests.get(url, timeout=6)
        print('GET', r.status_code, 'len', len(r.text))
        print(r.text[:600])
    except Exception as e:
        print('GET error', type(e), e)

    # try GET with common query param names
    for qname in ['api_key','auth_key','key','access_token','token','apikey']:
        try:
            r = requests.get(url, params={'text':'hello','from':'en','to':'mn', qname: key} if key else {'text':'hello','from':'en','to':'mn'}, timeout=6)
            print('GET param', qname, '->', r.status_code, 'len', len(r.text))
            print(r.text[:300])
        except Exception as e:
            print('GET param error', qname, type(e), e)

    # try POST with different payload styles and header styles
    payloads = [
        {'text':'Hello','from':'en','to':'mn'},
        {'text':'Hello','source':'en','target':'mn'},
        {'q':'Hello','source':'en','target':'mn'},
        {'input':'Hello','from':'en','to':'mn'},
    ]
    headers_list = []
    if key:
        headers_list = [
            {'Authorization': f'Bearer {key}'},
            {'Api-Key': key},
            {'X-API-Key': key},
        ]
    headers_list.append({})

    for hdr in headers_list:
        for payload in payloads:
            try:
                r = requests.post(url, json=payload, headers=hdr, timeout=6)
                print('POST hdr', hdr and list(hdr.keys())[0], 'payload keys', list(payload.keys()), '->', r.status_code)
                s = r.text or ''
                print('body:', (s[:600] + ('...' if len(s) > 600 else '')))
            except Exception as e:
                print('POST error', type(e), e)

print('\nDONE')
