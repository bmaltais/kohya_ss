@echo off
setx CUDA_VISIBLE_DEVICES "3"
echo [Watcher] Starting caption watcher...
python cap-watcher.py --overwrite
REM --img_dir "../dataset/captioning/mainchar" --out_dir "../dataset/captioning/mainchar"
pause