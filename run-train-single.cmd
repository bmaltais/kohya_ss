@echo off
REM 단순히 CMD에서 인자 전달만
docker exec -it sdxl_train_captioner bash -c "/app/sdxl_train_captioner/sd-scripts/run-train-single.sh %*"
pause
