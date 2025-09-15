import os
import pathlib
import argparse
from ocr_torii import ocr_image
from translate_lingva import translate_lines


def save_translations(out_dir: str, image_path: str, lines, translations):
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = pathlib.Path(image_path).stem
    out_file = out_dir / (base + '.translations.txt')
    with open(out_file, 'w', encoding='utf-8') as f:
        for src, tgt in zip(lines, translations):
            f.write(src + '\n')
            f.write('-> ' + tgt + '\n\n')
    print(f'Wrote translations to {out_file}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='input/01.png')
    p.add_argument('--out', default='output')
    p.add_argument('--no-network', action='store_true', help='Use dummy OCR and local-only translation; do not call external APIs')
    args = p.parse_args()

    img = args.input
    api_key = os.getenv('TORII_API_KEY')

    if args.no_network:
        print('[no-network] Using dummy OCR lines')
        raw_lines = [{'text': 'Hello world!'}, {'text': 'This is a test.'}, {'text': 'Goodbye!'}]
    else:
        if not api_key:
            print('TORII_API_KEY not set in environment; use --no-network to run without network')
            raise SystemExit(1)

        try:
            data = ocr_image(img, api_key)
        except Exception as e:
            print('OCR failed:', e)
            raise

        # Torii OCR returns a dict with 'lines' (each has 'text')
        if isinstance(data, dict):
            raw_lines = data.get('lines') or []
        else:
            raw_lines = getattr(data, 'lines', [])

    lines = []
    for item in raw_lines:
        t = item.get('text') if isinstance(item, dict) else getattr(item, 'text', None)
        if t:
            s = t.strip()
            if s:
                lines.append(s)

    if not lines:
        print('No text lines found from OCR')
        raise SystemExit(1)

    print('OCR lines:')
    for l in lines:
        print('-', l)

    # Translate using translate_lines (LingvaNex-backed when configured)
    try:
        if args.no_network:
            from translate_lingva import translate_lines_impl
            translations = translate_lines_impl(lines, source_lang='auto', target_lang='mn', local_only=True)
        else:
            translations = translate_lines(lines)
    except Exception as e:
        print('Translation failed:', e)
        raise

    print('\nTranslations:')
    for t in translations:
        print('-', t)

    save_translations(args.out, img, lines, translations)
