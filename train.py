import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torcheval.metrics import StructuralSimilarity

# Imports de tes fichiers locaux
from models.unet import UNet
from utils.vintage_transforms import VintageTransform

def trainModel():
    """
    Main training function for the Old Pictures Colorization project.
    Implements L1 Loss and SSIM metric as requested.
    """
    # 1. Configuration
    dataDir = "data/imagenet"
    batchSize = 16
    epochs = 5
    learningRate = 1e-4
    
    # Choix du processeur (utilise GPU si dispo, sinon CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Training starting on {device} ---")

    # 2. Pipeline de dégradation (Suivant les consignes du sujet)
    vintageTransform = VintageTransform()

    # 3. Préparation des données (ImageNet standard)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    try:
        # On utilise ImageFolder car ImageNet-1K est structuré par dossiers
        trainDataset = datasets.ImageFolder(root=f"{dataDir}/train", transform=transform)
        trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True)
    except Exception as e:
        print(f"Erreur dossier : Assure-toi d'avoir des images dans {dataDir}/train/une_classe/")
        return

    # 4. Initialisation du modèle U-Net, de la Loss et de l'Optimiseur
    model = UNet().to(device)
    criterionL1 = nn.L1Loss() # Consigne : Loss de type L1
    optimizer = optim.Adam(model.parameters(), lr=learningRate)
    
    # Métrique SSIM (Consigne : torcheval.metrics.StructuralSimilarity)
    ssimMetric = StructuralSimilarity(device=device)

    # 5. Boucle d'entraînement
    for epoch in range(epochs):
        model.train()
        runningLoss = 0.0
        
        for batchIdx, (targetImages, _) in enumerate(trainLoader):
            targetImages = targetImages.to(device) # Image propre (Vérité terrain)
            
            # Création de l'image dégradée "On-the-fly"
            # On applique la transformation vintage sur chaque image du batch
            inputImages = torch.stack([vintageTransform(img.cpu()) for img in targetImages]).to(device)

            # Forward pass
            outputImages = model(inputImages)
            loss = criterionL1(outputImages, targetImages)

            # Backward pass (Optimisation)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            runningLoss += loss.item()

            if batchIdx % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Batch {batchIdx} - Loss: {loss.item():.4f}")

        # Calcul SSIM sur le dernier batch de l'époque
        ssimMetric.update(outputImages, targetImages)
        print(f"--- Fin Epoque {epoch+1} | SSIM Moyen: {ssimMetric.compute():.4f} ---")
        
        # Sauvegarde du modèle
        torch.save(model.state_dict(), "outputs/models/best_model.pth")

if __name__ == "__main__":
    trainModel()