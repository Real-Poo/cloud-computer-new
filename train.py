import os
import argparse
import glob
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

# 작성해주신 모델 파일 임포트
from models.encoder import SplitSender
from models.decoder import SplitReceiver

# ==========================================
# 1. 데이터셋 (이미지 로드)
# ==========================================
class ScreenDataset(Dataset):
    def __init__(self, folder_path, width, height):
        self.files = sorted(glob.glob(os.path.join(folder_path, '*.*')))
        self.files = [f for f in self.files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # SR이 아니므로, 입력 이미지를 우리가 원하는 전송 해상도로 딱 맞춤
        self.transform = transforms.Compose([
            transforms.Resize((height, width)), # (H, W) 순서
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.files[idx]).convert('RGB')
            return self.transform(img)
        except Exception:
            return torch.zeros(3, 256, 256)

# ==========================================
# 2. U-Net 해상도 제약 해결 (32 배수 맞춤)
# ==========================================
def align_to_32(val):
    """
    U-Net 구조상 가로/세로는 32의 배수여야 에러가 안 납니다.
    입력값을 가장 가까운 32의 배수로 올림합니다.
    """
    return ((val + 31) // 32) * 32

# ==========================================
# 3. 학습 함수 (Reconstruction Only)
# ==========================================
def train_reconstruction(args):
    # 1. 해상도 자동 보정 (모델 에러 방지용)
    target_w = align_to_32(args.width)
    target_h = align_to_32(args.height)
    
    if target_w != args.width or target_h != args.height:
        print(f"[*] 모델 구조에 맞춰 해상도를 조정합니다: ({args.width}x{args.height}) -> ({target_w}x{target_h})")
    
    # 2. 장치 설정
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
        
    print(f"Using device: {device}")

    # 3. 모델 초기화
    sender = SplitSender(n_channels=3).to(device)
    receiver = SplitReceiver(n_classes=3).to(device)

    # SR 모듈이 없는지 확인 (기본 제공된 decoder.py는 SR 모듈이 없음)
    
    # 4. 데이터셋 준비
    dataset = ScreenDataset(args.data_dir, width=target_w, height=target_h)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # 5. 최적화 도구 (MSE Loss: 픽셀 단위 복원력 학습)
    criterion = nn.MSELoss()
    params = list(sender.parameters()) + list(receiver.parameters())
    optimizer = optim.Adam(params, lr=args.lr)

    print(f"=== 학습 시작 (Target: {target_w}x{target_h}) ===")
    
    sender.train()
    receiver.train()

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for input_imgs in progress_bar:
            input_imgs = input_imgs.to(device)
            
            # --- Forward Pass ---
            # 인코더: 이미지를 압축하여 패킷 생성
            packet_bot, list_skips = sender(input_imgs)
            
            # 디코더: 패킷을 받아 원본 해상도로 복원 (Upscaling 아님, Restoration임)
            output_imgs = receiver(packet_bot, list_skips)
            
            # --- Safety Guard ---
            # 출력 크기가 입력과 다르면 강제로 맞춤 (학습 터짐 방지)
            if output_imgs.shape != input_imgs.shape:
                output_imgs = torch.nn.functional.interpolate(
                    output_imgs, 
                    size=(input_imgs.shape[2], input_imgs.shape[3]), 
                    mode='bilinear', align_corners=False
                )

            # --- Loss Calculation ---
            # 입력(원본)과 출력(복원본)이 얼마나 똑같은지 비교
            loss = criterion(output_imgs, input_imgs)
            
            # --- Backward Pass ---
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.5f}"})

        # 체크포인트 저장
        if (epoch + 1) % 10 == 0:
            torch.save(sender.state_dict(), f"sender_epoch{epoch+1}.pth")
            torch.save(receiver.state_dict(), f"receiver_epoch{epoch+1}.pth")

    # 최종 저장
    torch.save(sender.state_dict(), "sender_final.pth")
    torch.save(receiver.state_dict(), "receiver_final.pth")
    print("=== 학습 완료: sender_final.pth, receiver_final.pth 저장됨 ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./train_data', help='학습 이미지 폴더')
    parser.add_argument('--epochs', type=int, default=100)
    
    # 원하는 해상도 (예: 1280 720)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--height', type=int, default=256)
    
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)

    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
        print(f"알림: '{args.data_dir}' 폴더를 만들었습니다. 이미지를 넣어주세요.")
    else:
        train_reconstruction(args)
