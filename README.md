## terimnal 1
& C:\lamaenv\Scripts\Activate.ps1
& C:\lamaenv\Scripts\iopaint.exe start --model=lama --device=cpu --host=127.0.0.1 --port=8090

## terimnal 2
& C:/FEDoUP/manga-translation--main/.venv/Scripts/Activate.ps1
cd C:\Users\User\Downloads\torii-openai-image-translator

$env:INPAINT_URL   = "http://127.0.0.1:8090/api/v1/inpaint"
$env:INPAINT_MODEL = "lama"
$env:INPAINT_HD    = "Original"
$env:INPAINT_SIZE  = "4096"
$env:WATERMARK_PATH = "C:\Users\User\Downloads\torii-openai-image-translator\watermark\watermark.png"

python app.py


 



