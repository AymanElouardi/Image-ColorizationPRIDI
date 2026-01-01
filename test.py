import torch
import matplotlib.pyplot as plt
from models.unet import UNet
from PIL import Image
from torchvision import transforms
import os

def runInference(imagePath):
    """
    Charge une photo et tente de la restaurer avec le U-Net.
    """
    # 1. Gestion du processeur
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Initialisation du modèle
    model = UNet().to(device)
    
    # NOTE: Si tu as déjà un entraînement fini, décommente la ligne ci-dessous :
    # model.load_state_dict(torch.load("outputs/models/best_model.pth", map_all_location=device))
    
    model.eval()

    # 3. Préparation de l'image
    if not os.path.exists(imagePath):
        print(f"Erreur : Le fichier {imagePath} est introuvable.")
        return

    img = Image.open(imagePath).convert("RGB")
    originalSize = img.size # On garde la taille pour l'affichage
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    inputTensor = transform(img).unsqueeze(0).to(device)

    # 4. Prédiction (Inférence)
    print(f"Traitement de l'image {imagePath}...")
    with torch.no_grad():
        output = model(inputTensor)

    # 5. Affichage des résultats
    plt.figure(figsize=(12, 6))
    
    # Image d'entrée
    plt.subplot(1, 2, 1)
    plt.title("Photo Originale (Input)")
    plt.imshow(img)
    plt.axis('off')
    
    # Résultat de l'IA
    plt.subplot(1, 2, 2)
    plt.title("Restauration IA (Output)")
    # On remet l'image dans le bon ordre de dimensions pour matplotlib (H, W, C)
    restoredImg = output.squeeze(0).cpu().permute(1, 2, 0)
    plt.imshow(restoredImg)
    plt.axis('off')
    
    # Sauvegarde du résultat pour ton rapport
    plt.tight_layout()
    plt.savefig("outputs/images/mon_premier_test.png")
    print("Résultat sauvegardé dans outputs/images/mon_premier_test.png")
    plt.show()

if __name__ == "__main__":
    # Correction faite ici : on ferme bien le guillemet "
    nomDeTonImage = "data/imagenet/train/class1/TEST1.jpg"
    
    # On vérifie si le dossier output existe avant de lancer
    if not os.path.exists("outputs/images"):
        os.makedirs("outputs/images", exist_ok=True)
        
    runInference(nomDeTonImage)