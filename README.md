# SAMURAI_with_ObjectDetection_Project
SegmentAnything Model v2를 활용한 모델인 SAMURAI. 객체 분할 기법에 추적 모듈을 통해 개선한 방법.
SegmentAnything Model v2를 이용하여 좌표를 찍을 필요 없이 원하는 객체탐지를 바로 한 후에, 트래킹 기능 구현.
---

## 환경 설정

- Python 3.10  
- PyTorch 2.3.1  
- CUDA 11.8
- torchvision>=0.18.1  

필수 패키지 설치:

```bash
git clone https://github.com/yangchris11/samurai.git
pip install -e ".[notebooks]" SAM2 경로로 가서 아래 명령어를 통해 SAM2를 설치()
pip install matplotlib==3.7 tikzplotlib jpeg4py opencv-python lmdb pandas scipy loguru
pip install ultralytics
```

---

##  Config 파일
```bash
# YOLOXPose-M 모델의 config 코드
[gdown 1r3B1xhkyKYcQ9SR7o9hw9zhNJinRiHD-](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/yolox_pose/yoloxpose_m_8xb32-300e_coco-640-84e9a538_20230829.pth)
```

---

## Pretrained Weights 다운로드
```bash
# SAM 2.1 버전의 모델 웨이트
cd checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
```

---

## Result
![Result GIF](https://raw.githubusercontent.com/hyunahn23/SAMURAI_with_ObjectDetection_Project/main/result.gif)
