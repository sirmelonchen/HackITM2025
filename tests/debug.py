import cv2
import torch
import numpy as np
from tqdm import tqdm
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
from model import RRDBNet  # ESRGAN Super-Resolution Modell

# CUDA prüfen
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Modell laden
model_path = "RealESRGAN_x4plus.pth"
model = RRDBNet(3, 3, 64, 23, gc=32).to(device)
model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
model.eval()

# Video einlesen
video_path = "input_video.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Fehler: Konnte das Eingabevideo nicht öffnen.")
    exit(1)

# Video-Parameter abrufen
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if frame_width == 0 or frame_height == 0:
    print("Fehler: Video konnte nicht geladen werden oder hat ungültige Abmessungen.")
    exit(1)

# Upscaling-Faktor
upscale_factor = 4  # ESRGAN nutzt x4 Upscaling

# Output-Video vorbereiten
output_path = "enhanced_video.mp4"
out = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*'avc1'),  # Falls mp4v nicht funktioniert
    fps,
    (frame_width * upscale_factor, frame_height * upscale_factor)
)

if not out.isOpened():
    print("Fehler: Konnte das Ausgabevideo nicht initialisieren.")
    cap.release()
    exit(1)

# Verarbeitung der Frames
pbar = tqdm(total=frame_count, desc="Processing Frames")
frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Debug: Überprüfen, ob Frames gelesen werden
    print(f"Verarbeite Frame {frame_idx + 1}/{frame_count}")
    
    # Frame von BGR zu RGB konvertieren
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame)

    # Bild zu Tensor konvertieren (von [0,255] auf [-1,1] skalieren)
    input_tensor = ToTensor()(pil_image).unsqueeze(0).to(device) * 2 - 1
    
    # Sicherstellen, dass Tensor-Werte im gültigen Bereich sind
    print(f"Input Tensor min: {input_tensor.min().item()}, max: {input_tensor.max().item()}")

    # Super-Resolution anwenden
    with torch.no_grad():
        output_tensor = model(input_tensor)
    
    # Werte zurück auf [0,1] skalieren
    output_tensor = (output_tensor + 1) / 2
    output_tensor = output_tensor.clamp(0, 1)  # Werte begrenzen
    
    # Tensor in PIL-Bild umwandeln (RGB!)
    output_image = ToPILImage()(output_tensor.squeeze(0).cpu())
    
    # Debug: Erstes hochskaliertes Bild in RGB speichern
    if frame_idx == 0:
        output_image.save("debug_output_frame_rgb_fixed.jpg")
        print("Gespeichertes Debug-Bild: debug_output_frame_rgb_fixed.jpg")
    
    # Bild zurück zu OpenCV (RGB → BGR), damit die Farben korrekt im Video gespeichert werden
    output_frame = cv2.cvtColor(np.array(output_image), cv2.COLOR_RGB2BGR)

    # Frame ins neue Video schreiben
    out.write(output_frame)
    pbar.update(1)
    frame_idx += 1

# Ressourcen freigeben
cap.release()
out.release()
pbar.close()
print(f"Video-Verbesserung abgeschlossen: {output_path}")