# I. 단일폴더 수동 학습 사용 예시

## 1. 기본 사용 (자동 계산)
```cmd
run-train-single --folder ../dataset/training/01_alice
```

##  2. Epochs만 수동 지정
```cmd
run-train-single --folder ../dataset/training/01_alice --epochs 25

또는

run-train-single --folder ../training/mainchar/01_alic3 --epochs 17 --resume alic3-000009.safetensors  
```
 
## 3. 세밀한 조정
```cmd
run-train-single ^
  --folder ../dataset/training/01_alice ^
  --epochs 30 ^
  --repeats 50 ^
  --lr 0.00015 ^
  --dim 64 ^
  --alpha 32
```

## 4. 고해상도 학습
```cmd
run-train-single ^
  --folder ../dataset/training/01_alice ^
  --resolution 1024,1024 ^
  --batch-size 1
```

## 5. 빠른 테스트
```cmd
run-train-single ^
  --folder ../dataset/training/01_alice ^
  --epochs 5 ^
  --repeats 10 ^
  --save-every 1
```

## 6. 완전 수동 모드
```cmd
run-train-single ^
  --folder ../dataset/training/01_alice ^
  --no-auto ^
  --epochs 20 ^
  --repeats 30 ^
  --lr 0.0001 ^
  --optimizer AdamW8bit ^
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
```cmd
# 여러 폴더 자동 학습
python train_batch.py
→ 01_alice, 02_bob, 03_background 모두 학습
```

### train_single.py (단일 수동)
```cmd
# 특정 폴더만 세밀 조정
run-train-single --folder ../dataset/training/mainchar/01_alice --epochs 30 --lr 0.00015
→ alice만 커스텀 파라미터로 학습
```

## 워크플로우 추천

### 초보자
```cmd
# 1. 먼저 일괄 자동으로 테스트
python train_batch.py

# 2. 결과가 좋지 않은 캐릭터만 재학습
run-train-single --folder ../dataset/training/mainchar/01_alice --epochs 25
```

### 고급 사용자
```cmd
# 처음부터 세밀하게 조정
run-train-single ^
  --folder ../dataset/training/mainchar/01_alice ^
  --epochs 30 ^
  --repeats 50 ^
  --lr 0.00012 ^
  --dim 64 ^
  --alpha 32 ^
  --optimizer Prodigy ^
  --resolution 1024,1024
```

# II. 단일폴더 학습재개(resume) 방법

## 1. 기본 Resume
```cmd
run-train-single --folder ../dataset/training/mainchar/01_alice --resume ../output_models/alice-epoch-010.safetensors
```

## 2. Resume + Learning Rate 조정 (Fine-tuning)
```cmd
run-train-single --folder ../dataset/training/mainchar/01_alice ^
    --folder ../dataset/training/01_alice ^
    --resume ../output_models/alice-epoch-010.safetensors ^
    --epochs 20 ^
    --lr 0.00005
```

## 3. Resume + 더 많은 데이터
```cmd
run-train-single --folder ../dataset/training/mainchar/01_alice ^
  --folder ../dataset/training/01_alice_more ^
  --resume ../output_models/alice-epoch-015.safetensors ^
  --epochs 10
```

## 주의사항
✅ Resume 시 동일하게 유지해야 할 것

- --dim (network_dim)
- --alpha (network_alpha)
- 네트워크 구조 관련 설정

## ⚠️ Resume 시 변경 가능한 것

- --epochs (더 학습)
- --lr (learning rate 조정)
- --repeats (데이터 반복)
- --optimizer (optimizer 변경)
- --scheduler (스케줄러 변경)

## ❌ Resume 시 변경하면 안되는 것
```cmd
# 잘못된 예
run-train-single \
  --folder ../dataset/training/01_alice \
  --resume ../output_models/alice-epoch-010.safetensors \
  --dim 64  # ❌ 원래 32였으면 에러!
```

## 실전 예시

### 시나리오 1: 학습이 중단됨
```cmd
# 10 epoch에서 중단
# → 10 epoch부터 이어서 15 epoch까지

run-train-single ^
  --folder ../dataset/training/01_alice ^
  --resume ../output_models/alice-epoch-010.safetensors ^
  --epochs 15
```

### 시나리오 2: Overfitting 방지 (LR 감소)
```cmd
# 학습률 낮춰서 Fine-tuning
run-train-single ^
  --folder ../dataset/training/01_alice ^
  --resume ../output_models/alice-epoch-015.safetensors ^
  --epochs 25 ^
  --lr 0.00005
```

### 시나리오 3: 데이터 추가 후 재학습
```cmd
# 이미지 20장 → 50장으로 증가
run-train-single ^
  --folder ../dataset/training/01_alice_extended ^
  --resume ../output_models/alice-epoch-015.safetensors ^
  --epochs 10 ^
  --repeats 20
```

## 출력 예시
```
======================================================================
🎯 SDXL LoRA Training - Single Mode
======================================================================
📁 Folder:         ../dataset/training/01_alice
💾 Output:         alice.safetensors
📋 Config:         config-24g.json
🖥️  GPU:            0 (24GB VRAM)
⚡ Precision:      bf16
🔄 Resume from:    ../output_models/alice-epoch-010.safetensors
----------------------------------------------------------------------
📊 Training Parameters
----------------------------------------------------------------------
  Images:          25
  Repeats:         48 (auto)
  Epochs:          20 (manual)
  Batch size:      1
  Images/epoch:    1200
  Steps/epoch:     1200
  Total steps:     24000
======================================================================

학습을 시작하시겠습니까? (y/N): y

🔄 Resuming from: ../output_models/alice-epoch-010.safetensors

🚀 Starting training...
```