@echo off
REM 첫 번째 argument를 명령어로 받아서 컨테이너에서 실행
REM 모든 argument를 그대로 넘기려면 %* 사용
docker exec -it sdxl_train_captioner bash -c "cd /app/sdxl_train_captioner/sd-scripts; ./run-train-auto.py 1 2>&1 | tee /app/sdxl_train_captioner/logs/train_$(date +%%Y%%m%%d_%%H%%M%%S).log"

pause