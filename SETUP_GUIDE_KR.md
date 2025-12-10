# UniGaze 실행 가이드

## 설치 완료!

UniGaze 프로젝트가 성공적으로 설정되었습니다.

### 환경 정보
- **Conda 환경**: `unigaze`
- **Python**: 3.10.19
- **PyTorch**: 2.0.1 + CUDA 11.8
- **timm**: 0.3.2
- **unigaze**: 0.1.3

---

## 실행 방법

### 1. Quick Start (기본 추론)

테스트 파일이 이미 생성되어 있습니다: `test_quick_start.py`

```powershell
# PowerShell에서 실행
conda run -n unigaze python test_quick_start.py
```

또는 Python 코드에서 직접:

```python
import torch
import unigaze

# 모델 로드 (최초 실행 시 Hugging Face에서 자동 다운로드)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = unigaze.load("unigaze_h14_joint", device=device)

# 입력: (B, 3, 224, 224) - 정규화된 이미지 배치
image_batch = torch.ones((10, 3, 224, 224), device=device)

# 출력: {'pred_gaze': (B, 2)} - (pitch, yaw) 각도
pred_gaze = model(image_batch)['pred_gaze']
print(pred_gaze.shape)  # torch.Size([10, 2])
```

### 2. 비디오에서 시선 추정

```powershell
cd unigaze
conda run -n unigaze python predict_gaze_video.py --model_name "unigaze_h14_joint" -i "C:\path\to\your\video.mp4"
```

### 3. 사용 가능한 모델

| Model Name              | Backbone  | Training Data |
|------------------------|-----------|---------------|
| `unigaze_b16_joint`    | UniGaze-B | Joint Datasets|
| `unigaze_l16_joint`    | UniGaze-L | Joint Datasets|
| `unigaze_h14_joint`    | UniGaze-H | Joint Datasets|
| `unigaze_h14_cross_X`  | UniGaze-H | ETH-XGaze     |

---

## 환경 관리

### 환경 활성화 (PowerShell에서 conda activate가 작동하지 않는 경우)

```powershell
# 방법 1: conda run 사용 (권장)
conda run -n unigaze python your_script.py

# 방법 2: conda init 후 재시작
conda init powershell
# PowerShell 재시작 후
conda activate unigaze
```

### 추가 패키지 설치

```powershell
conda run -n unigaze pip install package_name
```

---

## 문제 해결

### CUDA 사용 불가능한 경우

CPU로 실행하려면:
```python
model = unigaze.load("unigaze_h14_joint", device="cpu")
```

### 다른 CUDA 버전 사용

CUDA 12.1 예시:
```powershell
conda run -n unigaze pip uninstall torch torchvision
conda run -n unigaze pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## 참고 자료

- **논문**: https://arxiv.org/pdf/2502.02307
- **프로젝트 페이지**: https://ut-vision.github.io/UniGaze/
- **온라인 데모**: https://huggingface.co/spaces/UniGaze/UniGaze
- **PyPI**: https://pypi.org/project/unigaze/
- **문의**: jqin@iis.u-tokyo.ac.jp
