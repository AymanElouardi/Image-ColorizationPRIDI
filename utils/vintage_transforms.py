import torch
import numpy as np
import cv2

class VintageTransform:
    """transforme image propre en photo vintage dégradée"""

    def __init__(self, scratchProbability=0.5, noiseLevel=0.05):
        """stocke probas rayures et niveau bruit"""
        self.scratchProbability = scratchProbability
        self.noiseLevel = noiseLevel

    def applySepia(self, imageTensor):
        """applique teinte sépia"""
        imageNp = imageTensor.permute(1, 2, 0).numpy()
        
        # matrice pour couleur sépia
        sepiaMatrix = np.array([
            [0.272, 0.534, 0.131],
            [0.349, 0.686, 0.168],
            [0.393, 0.769, 0.189]
        ])
        
        sepiaImage = cv2.transform(imageNp, sepiaMatrix)
        sepiaImage = np.clip(sepiaImage, 0, 1)
        
        return torch.from_numpy(sepiaImage).float().permute(2, 0, 1)

    def addGaussianNoise(self, imageTensor):
        """ajoute bruit gaussien (grain film)"""
        noise = torch.randn_like(imageTensor) * self.noiseLevel
        return torch.clamp(imageTensor + noise, 0, 1)

    def addScratches(self, imageTensor):
        """dessine rayures blanches aléatoires"""
        if np.random.rand() > self.scratchProbability:
            return imageTensor

        _, height, width = imageTensor.shape
        scratchMask = np.zeros((height, width), dtype=np.float32)

        # crée 1-5 rayures
        for _ in range(np.random.randint(1, 6)):
            x1, y1 = np.random.randint(0, width), np.random.randint(0, height)
            x2, y2 = np.random.randint(0, width), np.random.randint(0, height)
            thickness = np.random.randint(1, 3)
            cv2.line(scratchMask, (x1, y1), (x2, y2), 1.0, thickness)

        scratchTensor = torch.from_numpy(scratchMask).unsqueeze(0).repeat(3, 1, 1)
        return torch.clamp(imageTensor + scratchTensor, 0, 1)

    def __call__(self, imageTensor):
        """pipeline complet: sépia -> bruit -> rayures"""
        x = self.applySepia(imageTensor)
        x = self.addGaussianNoise(x)
        x = self.addScratches(x)
        return x
