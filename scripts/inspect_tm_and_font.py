import sqlite3, os, json, sys
DB='tm.db'
print('cwd:', os.getcwd())
print('tm.db exists:', os.path.exists(DB))
rows=[]
if os.path.exists(DB):
    conn=sqlite3.connect(DB)
    rows=conn.execute('SELECT source,target,model,ts FROM translations ORDER BY ts DESC LIMIT 50').fetchall()
    conn.close()
print('rows count:', len(rows))
print(json.dumps(rows, ensure_ascii=False, indent=2))

# Ensure project root is on sys.path so `from config import ...` works when script is in scripts/
proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from config import DEFAULT_FONT_PATH
print('DEFAULT_FONT_PATH:', DEFAULT_FONT_PATH)
print('font exists:', os.path.exists(DEFAULT_FONT_PATH))
if not os.path.exists(DEFAULT_FONT_PATH):
    # try fonts dir
    fonts_dir = os.path.join(proj_root, 'fonts')
    if os.path.isdir(fonts_dir):
        print('fonts dir contents:')
        for f in os.listdir(fonts_dir)[:50]:
            print(' -', f)
