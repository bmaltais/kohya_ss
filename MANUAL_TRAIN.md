# 단일폴더 수동 학습 사용 예시

## 1. 기본 사용 (자동 계산)
```bash
python train_single.py --folder ../dataset/training/01_alice
```

##  2. Epochs만 수동 지정
```bash
python train_single.py --folder ../dataset/training/01_alice --epochs 25
```
 
## 3. 세밀한 조정
```bash
python train_single.py \
  --folder ../dataset/training/01_alice \
  --epochs 30 \
  --repeats 50 \
  --lr 0.00015 \
  --dim 64 \
  --alpha 32
```

## 4. 고해상도 학습
```bash
python train_single.py \
  --folder ../dataset/training/01_alice \
  --resolution 1024,1024 \
  --batch-size 1
```

## 5. 빠른 테스트
```bash
python train_single.py \
  --folder ../dataset/training/01_alice \
  --epochs 5 \
  --repeats 10 \
  --save-every 1
```

## 6. 완전 수동 모드
```bash
python train_single.py \
  --folder ../dataset/training/01_alice \
  --no-auto \
  --epochs 20 \
  --repeats 30 \
  --lr 0.0001 \
  --optimizer AdamW8bit \
  --scheduler cosine
```

## 주요 기능
✨ 자동 + 수동 하이브리드

- 기본값은 자동 계산
- 원하는 파라미터만 오버라이드
- --no-auto 플래그로 완전 수동 제어

## 🎯 주요 파라미터
| 파라미터 | 설명 | 예시 |
|---------|------|------|
| --folder | 학습 폴더 (필수) | ../dataset/training/01_alice |
| --output | 출력 이름 | alice_v2 |
| --epochs | Epoch 수 | 20 |
| --repeats | 반복 횟수 | 30 |
| --lr | Learning rate | 0.0001 |
| --dim | Network dimension | 64 |
| --alpha | Network alpha | 32 |
| --resolution | 해상도 | 1024,1024 |
| --batch-size | 배치 크기 | 2 |
| --optimizer | Optimizer | AdamW8bit, Lion, Prodigy |
| --scheduler | LR Scheduler | cosine, constant |
| --save-every | 저장 주기 | 5 |

## 비교
### train_batch.py (일괄 자동)
```bash
# 여러 폴더 자동 학습
python train_batch.py
→ 01_alice, 02_bob, 03_background 모두 학습
```

### train_single.py (단일 수동)
```bash
# 특정 폴더만 세밀 조정
python train_single.py --folder ../dataset/training/mainchar/01_alice --epochs 30 --lr 0.00015
→ alice만 커스텀 파라미터로 학습
```

## 워크플로우 추천

### 초보자
```bash
# 1. 먼저 일괄 자동으로 테스트
python train_batch.py

# 2. 결과가 좋지 않은 캐릭터만 재학습
python train_single.py --folder ../dataset/training/mainchar/01_alice --epochs 25
```

### 고급 사용자
```bash
# 처음부터 세밀하게 조정
python train_single.py \
  --folder ../dataset/training/mainchar/01_alice \
  --epochs 30 \
  --repeats 50 \
  --lr 0.00012 \
  --dim 64 \
  --alpha 32 \
  --optimizer Prodigy \
  --resolution 1024,1024
```
