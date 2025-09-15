import json
import os
import sys
# ensure project root is on sys.path so sibling modules can be imported when running from scripts/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from translate_lingva import translate_lines_impl

p = 'output/01.layout.json'
with open(p, 'r', encoding='utf-8') as fh:
    j = json.load(fh)

texts = [b['text'] for b in j['boxes']]
new = translate_lines_impl(texts, force_online=True)
changed = 0
for i, t in enumerate(j['translations']):
    if new[i] != t:
        changed += 1

j['translations'] = new
out = 'output/01.layout.online.json'
with open(out, 'w', encoding='utf-8') as fh:
    json.dump(j, fh, ensure_ascii=False, indent=2)

print('wrote', out, 'changed', changed)
for i, (old, newt) in enumerate(zip(texts, j['translations'])):
    if old != newt:
        print(i, '->', old, '=>', newt)
