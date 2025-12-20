import os
import argparse
import glob
from tqdm import tqdm # 진행바 표시 라이브러리 (pip install tqdm)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

# 우리가 만든 모델 모듈 임포트
from models.encoder import SplitSender
from models.decoder import SplitReceiver

# ==========================================
# 1. 커스텀 데이터셋 클래스
# ==========================================
class ImageFolderDataset(Dataset):
    """
    지정된 폴더에서 이미지 파일(*.jpg, *.png)을 읽어오는 데이터셋
    """
    def __init__(self, folder_path, img_size=256):
        self.files = sorted(glob.glob(os.path.join(folder_path, '*.*')))
        # 이미지 파일만 필터링
        self.files = [f for f in self.files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        if len(self.files) == 0:
            raise RuntimeError(f"폴더 '{folder_path}'에 이미지가 없습니다.")

        print(f"[{folder_path}]에서 {len(self.files)}장의 이미지를 찾았습니다.")

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(), # 0~1 범위로 변환
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            img_path = self.files[idx]
            img = Image.open(img_path).convert('RGB') # 흑백 이미지도 RGB로 강제 변환
            return self.transform(img)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # 에러 발생 시 랜덤 텐서 반환 (학습 중단 방지)
            return torch.zeros(3, 256, 256)

# ==========================================
# 2. 학습 함수
# ==========================================
def train(args):
    # 장치 설정 (Apple Silicon: mps, Nvidia: cuda, CPU)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
        
    print(f"Using device: {device}")

    # 데이터 로더 설정
    dataset = ImageFolderDataset(args.data_dir, img_size=args.img_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # 모델 초기화
    sender = SplitSender(n_channels=3).to(device)
    receiver = SplitReceiver(n_classes=3).to(device)

    # 손실 함수 및 최적화 도구 설정
    criterion = nn.MSELoss()
    
    # [핵심] 두 모델의 파라미터를 하나의 Optimizer에 등록하여 동시에 학습
    # Sender는 "잘 압축하는 법"을 배우고, Receiver는 "잘 복원하는 법"을 배움
    params = list(sender.parameters()) + list(receiver.parameters())
    optimizer = optim.Adam(params, lr=args.lr)

    print("=== 학습 시작 ===")
    
    sender.train()
    receiver.train()

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        
        # tqdm으로 진행바 표시
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for input_imgs in progress_bar:
            input_imgs = input_imgs.to(device)
            
            # --- Forward Pass ---
            # 1. 인코딩 및 압축
            packet_bot, list_skips = sender(input_imgs)
            
            # 2. 디코딩 및 복원
            output_imgs = receiver(packet_bot, list_skips)
            
            # --- 크기 보정 (Safety Guard) ---
            # 모델 구조상 패딩 등으로 인해 출력 크기가 입력과 미세하게 다를 수 있음
            if output_imgs.shape != input_imgs.shape:
                output_imgs = torch.nn.functional.interpolate(
                    output_imgs, 
                    size=(input_imgs.shape[2], input_imgs.shape[3]), 
                    mode='bilinear', 
                    align_corners=False
                )

            # --- Loss 계산 ---
            loss = criterion(output_imgs, input_imgs)
            
            # --- Backward Pass ---
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        # 에폭 종료 후 평균 Loss 출력
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{args.epochs}] Average Loss: {avg_loss:.6f}")

        # 체크포인트 저장 (10 에폭마다)
        if (epoch + 1) % 10 == 0:
            save_path = f"checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'sender_state': sender.state_dict(),
                'receiver_state': receiver.state_dict(),
                'optimizer_state': optimizer.state_dict(),
            }, save_path)
            print(f"Model saved to {save_path}")

    print("=== 학습 완료 ===")
    
    # 최종 모델 저장
    torch.save(sender.state_dict(), "sender_final.pth")
    torch.save(receiver.state_dict(), "receiver_final.pth")
    print("Final models saved as 'sender_final.pth' and 'receiver_final.pth'")

# ==========================================
# 3. 실행 진입점
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Split U-Net")
    
    # 커맨드라인 인자 설정
    parser.add_argument('--data_dir', type=str, default='./train_data', help='이미지 폴더 경로')
    parser.add_argument('--epochs', type=int, default=50, help='학습 반복 횟수')
    parser.add_argument('--batch_size', type=int, default=8, help='배치 크기 (메모리에 따라 조절)')
    parser.add_argument('--lr', type=float, default=0.001, help='학습률')
    parser.add_argument('--img_size', type=int, default=256, help='학습 이미지 크기')

    args = parser.parse_args()
    
    # 데이터 폴더가 없으면 생성 (안내용)
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
        print(f"알림: '{args.data_dir}' 폴더가 생성되었습니다. 여기에 학습할 이미지를 넣어주세요.")
    else:
        train(args)