Cloud Computer (Neural Screen Codec)
Split U-Net Architecture for Real-Time Screen Streaming

이 프로젝트는 딥러닝 모델인 U-Net을 송신부(Sender)와 수신부(Receiver)로 분리하여, 네트워크 대역폭을 절약하면서 고화질로 화면을 송출하는 Split Computing 기반의 뉴럴 코덱(Neural Codec) 시스템입니다.

기존의 H.264/H.265 코덱과 달리, 이미지의 특징(Feature)을 추출하고 압축하여 전송한 뒤 수신측에서 AI가 이를 복원하는 방식을 사용합니다.

🚀 주요 특징 (Key Features)
Split Computing 아키텍처: 모델을 Sender(Mobile/Client)와 Receiver(Server/Edge)로 물리적으로 분리.

Skip Connection 압축: U-Net의 핵심인 Skip Connection 정보를 1/8 수준으로 압축(Quantization & Channel Reduction)하여 전송.

End-to-End 학습: 송신부의 압축 방식과 수신부의 복원 방식을 동시에 학습.

실시간 처리: 경량화된 구조로 30FPS 이상의 실시간 화면 미러링 지향.

Retina 디스플레이 대응: 맥북 등 고해상도(HiDPI) 환경에서의 캡처 및 리사이즈 자동 처리.

📂 파일 구조 (File Structure)
Bash

cloud-computer-new/
├── models/
│   ├── encoder.py       # [Sender] 이미지 압축 및 특징 추출 (SplitSender)
│   └── decoder.py       # [Receiver] 특징 데이터 복원 및 화면 생성 (SplitReceiver)
├── train_codec.py       # AI 모델 학습 코드 (데이터셋 로드 -> 학습 -> 가중치 저장)
├── local_stream.py      # 로컬 실시간 테스트 (캡처 -> 모델 -> 화면 출력)
├── network.py           # 소켓 통신 프로토콜 (패킷 직렬화/역직렬화)
├── client_sender.py     # [실제 배포용] 송신 클라이언트 실행 파일
├── server_receiver.py   # [실제 배포용] 수신 서버 실행 파일
└── README.md
🛠️ 설치 및 환경 설정 (Installation)
Python 3.8 이상 및 PyTorch 환경이 필요합니다.

Bash

# 1. 저장소 클론
git clone https://github.com/Real-Poo/cloud-computer-new.git
cd cloud-computer-new

# 2. 필수 라이브러리 설치
pip install torch torchvision numpy opencv-python pyautogui pillow tqdm
📖 사용 방법 (Usage)
1. 모델 학습 (Training)
화면을 선명하게 복원하기 위해서는 먼저 학습이 필요합니다. 스크린샷 이미지들이 있는 폴더를 준비하세요.

Bash

# 기본 학습 (1280x720 해상도 예시)
python train_codec.py --data_dir ./train_data --width 1280 --height 720 --epochs 50

# 빠른 테스트용 학습 (256x256)
python train_codec.py --data_dir ./train_data --width 256 --height 256 --batch_size 8
학습이 완료되면 sender_final.pth, receiver_final.pth 파일이 생성됩니다.

Data Tip: 코딩 화면, 웹 브라우저, 문서 등 텍스트가 포함된 스크린샷을 학습 데이터로 쓰면 가독성이 좋아집니다.

2. 로컬 실시간 테스트 (Local Mirroring)
네트워크 연결 없이, 내 컴퓨터 화면을 캡처해서 AI가 어떻게 복원하는지 실시간으로 확인합니다. (학습된 .pth 파일이 있어야 선명하게 보입니다.)

Bash

python local_stream.py
기능: 화면 좌상단을 캡처하여 원본과 복원본을 나란히 보여줍니다.

조작: 종료하려면 q 키를 누르세요.

설정: CAPTURE_SIZE 변수를 학습했을 때의 해상도(예: 256)와 맞춰주세요.

3. 네트워크 송수신 (Network Streaming)
실제 두 대의 컴퓨터(또는 터미널 2개)에서 통신합니다.

Step 1: 수신 서버 (Receiver) 실행

Bash

python server_receiver.py
# Server listening on port 9999...
Step 2: 송신 클라이언트 (Sender) 실행

Bash

python client_sender.py
# Connected to server!
🔧 트러블슈팅 (Troubleshooting)
Q. 화면에 회색 노이즈만 나옵니다.

학습된 모델 파일(sender_final.pth, receiver_final.pth)이 실행 파일과 같은 폴더에 있는지 확인하세요.

학습을 진행하지 않았다면 초기화된 랜덤 값 때문에 노이즈가 출력되는 것이 정상입니다.

Q. RuntimeError: ... but got 4 channels instead 에러 발생

맥북(macOS)에서 스크린샷 캡처 시 투명도 채널(Alpha)이 포함되어 4채널(RGBA)이 될 수 있습니다.

local_stream.py에는 이를 자동으로 3채널(RGB)로 변환하는 코드가 포함되어 있습니다. 최신 코드를 사용하세요.

Q. ValueError: ... dimension 0 ... size 512 ... size 1024 에러 발생

Retina 디스플레이는 논리 해상도보다 2배 큰 물리 해상도로 캡처됩니다 (예: 256 설정 시 512로 캡처).

코드 내에서 강제 리사이즈(screenshot.resize) 로직이 이를 보정합니다.

📝 라이선스 (License)
This project is for educational and research purposes.

개발자 노트
Sender: SplitSender (Encoder + Compressors)

Receiver: SplitReceiver (Decompressors + Decoder)

Current Status: 1:1 Codec Mode (No Super-Resolution applied yet)
