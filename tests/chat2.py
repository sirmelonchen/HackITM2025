import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# Definition des Real-ESRGAN Modells (RRDB)
class RRDB(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(RRDB, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 3, padding=1)

    def forward(self, x):
        res = x
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.conv3(out)
        return res + out  # Skip connection

# Definition des Real-ESRGAN Modells
class RealESRGAN(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_rrdb_blocks=23, mid_channels=64):
        super(RealESRGAN, self).__init__()

        # Initial Layer
        self.first_conv = nn.Conv2d(in_channels, mid_channels, 3, padding=1)
        
        # RRDB Blocks
        self.rrdb_blocks = nn.Sequential(
            *[RRDB(mid_channels, mid_channels, mid_channels) for _ in range(num_rrdb_blocks)]
        )

        # Upsampling Layer
        self.upsample = nn.Conv2d(mid_channels, out_channels, 3, padding=1)
        
    def forward(self, x):
        x = self.first_conv(x)
        x = self.rrdb_blocks(x)
        x = self.upsample(x)
        return x

# Laden des vortrainierten Modells
def load_model(model_path, device):
    model = RealESRGAN()  # Modell initialisieren
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)  # Lade die Gewichtedatei
    model.eval()  # Setze das Modell in den Evaluierungsmodus
    return model

# Funktion zum Hochskalieren eines Video-Frames
def upscale_frame(frame, model, device):
    # Konvertiere das Frame von OpenCV (BGR) zu PIL (RGB)
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Vorverarbeitung des Bildes
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),  # Optional: kannst du die Zielgröße ändern
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    img_tensor = transform(pil_img).unsqueeze(0).to(device)

    # Vorhersage durchführen
    with torch.no_grad():
        output = model(img_tensor)

    # Umwandlung in ein Bildformat
    output_img = output.squeeze().cpu().numpy().transpose(1, 2, 0)
    output_img = np.clip(output_img * 255, 0, 255).astype(np.uint8)

    # Rückumwandlung zu BGR für OpenCV
    return cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)

# Video laden und hochskalieren
def upscale_video(input_video_path, output_video_path, model, device):
    # Video laden
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print("Fehler beim Öffnen des Videos!")
        return

    # Ausgabe-Video-Format (z.B. 1080p) und VideoWriter einrichten
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # für mp4 Format
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (1920, 1080))

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Ende des Videos erreicht

        # Hochskalieren des Frames
        upscaled_frame = upscale_frame(frame, model, device)

        # Das hochskalierte Frame zum neuen Video hinzufügen
        out.write(upscaled_frame)

    # Ressourcen freigeben
    cap.release()
    out.release()

# Beispielaufruf: Modell laden und Video hochskalieren
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Wähle GPU, wenn verfügbar, ansonsten CPU
model_path = 'RealESRGAN_x4plus.pth'  # Pfad zu deinem .pth Modell
model = load_model(model_path, device)  # Lade das Modell auf das richtige Gerät

# Video hochskalieren
input_video_path = 'input_video.mp4'  # Pfad zum Eingabe-Video
output_video_path = 'hochskaliertes_video.mp4'  # Pfad zum Ausgabe-Video
upscale_video(input_video_path, output_video_path, model, device)
