# GVAE vs DirVAE sur les lésions pulmonaires — DeepLesion

Projet réalisé dans le cadre du cours **GenAI (ING3 — MLOps)**.

## Objectif

Comparer deux variantes de VAE (Variational Autoencoder) pour la modélisation de lésions pulmonaires issues du dataset médical **DeepLesion** :

- **GVAE** : VAE standard avec prior gaussien N(0, I) (β-VAE, β=5)
- **DirVAE** : VAE avec prior de Dirichlet (Joo et al., 2020), supposé induire un meilleur disentanglement

La contribution originale est d'**isoler uniquement les lésions pulmonaires** (`Coarse_lesion_type == 5`) du dataset DeepLesion et d'évaluer les deux modèles sur trois axes :
1. Qualité de reconstruction (SSIM, MSE, MAE)
2. Structure de l'espace latent (UMAP)
3. Disentanglement visuel (traversées latentes)

---

## Structure du repo

```
GenAI_DeepLesion/
├── main.ipynb          # Notebook principal (tout le code)
└── results/
    ├── metrics.csv             # Métriques quantitatives GVAE vs DirVAE
    ├── loss_curves.png         # Courbes de loss et SSIM pendant l'entraînement
    ├── reconstructions.png     # Comparaison des reconstructions (original / GVAE / DirVAE)
    ├── lung_samples.png        # Exemples de lésions pulmonaires du dataset
    ├── umap_comparison.png     # Projection UMAP des espaces latents
    ├── latent_variance.png     # Variance par dimension latente (activité des dims)
    ├── traversal_gvae.png      # Traversées latentes — GVAE
    ├── traversal_dirvae.png    # Traversées latentes — DirVAE
    └── synthetic_samples.png   # Lésions synthétiques générées depuis le prior
```

Tout le pipeline est contenu dans **un seul notebook** (`main.ipynb`), structuré en 15 sections numérotées.

---

## Dataset

**DeepLesion** (NIH) — dataset de tomodensitométries (CT scans) annotées avec des bounding boxes de lésions.

- Fichier CSV : `DL_info.csv` (fourni par le dataset NIH)
- Images : PNG 16 bits encodées en Hounsfield Units (HU) avec un offset de 32768
- Filtre appliqué : `Coarse_lesion_type == 5` → **lésions pulmonaires uniquement** (2394 dans le CSV, 100 disponibles localement)
- Preprocessing :
  - Crop sur la bounding box (+ padding de 10px)
  - Fenêtrage HU : [-1000, 400]
  - Resize en **64×64** pixels
  - Normalisation en [0, 1]

Split : **70 train / 15 val / 15 test** (seed=42)

---

## Architecture des modèles

Les deux modèles partagent le même encodeur et décodeur convolutifs :

| Composant | Détails |
|-----------|---------|
| Encodeur | 7 couches Conv + GELU + BatchNorm, stride 2 pour le downsampling |
| Décodeur | ConvTranspose + upsampling bilinéaire (évite les checkerboard artifacts) |
| Dimension latente | **256** (BASE=32 × LATENT_SIZE=8) |
| Paramètres | ~18.6M (GVAE) / ~18.2M (DirVAE) |

**GVAE** : prior gaussien N(0,I), reparamétrisation standard (μ + ε·σ).

**DirVAE** : le vecteur latent est échantillonné depuis une distribution de Dirichlet. Les concentrations α sont prédites via softplus (garantissant α > 0). La KL divergence est calculée analytiquement entre Dir(α_pred) et Dir(α_prior) avec α_prior = 0.99.

**Fonction de perte commune** :

```
L = α_ssim · L1 + (1 - α_ssim) · (1 - SSIM) + β_norm · KL · anneal(epoch)
```

Avec α_ssim=0.75, β=5, et un KL annealing exponentiel sur 60 epochs.

---

## Comment reproduire

Le notebook est conçu pour tourner sur **Kaggle** (GPU disponible). Les chemins sont configurés dans la section 1 :

```python
IMAGES_DIR = '/kaggle/input/.../deeplesion'
CSV_PATH   = '/kaggle/input/.../DL_info.csv'
OUTPUT_DIR = '/kaggle/working/results'
```

**Dépendances** (installées dans le notebook) :
```bash
pip install pytorch-msssim umap-learn
```

**Exécuter** : lancer toutes les cellules dans l'ordre. Le notebook entraîne les deux modèles séquentiellement (≈9 minutes chacun sur CPU, moins avec GPU) puis génère automatiquement tous les visuels dans `results/`.

---

## Résultats clés

### Métriques quantitatives (test set)

| Modèle | SSIM (↑) | MSE (↓) | MAE (↓) |
|--------|----------|---------|---------|
| **GVAE** | **0.2944** | **0.0514** | **0.1857** |
| DirVAE | 0.2482 | 0.0936 | 0.2670 |

Le GVAE surpasse le DirVAE sur toutes les métriques de reconstruction.

### Dimensions latentes actives

| Modèle | Dims actives (var > 0.01) |
|--------|--------------------------|
| **GVAE** | **249 / 256** |
| DirVAE | **3 / 256** |

Le DirVAE souffre d'un **collapse sévère de l'espace latent** : seulement 3 dimensions sur 256 sont réellement utilisées. Ce phénomène explique sa moins bonne reconstruction et limite fortement son disentanglement.

### Interprétation

Le prior de Dirichlet impose une contrainte de parcimonie très forte (les vecteurs doivent sommer à 1), ce qui couplé à une dimension latente élevée (256) conduit au collapse. Le GVAE, avec son prior gaussien plus souple, exploite l'espace latent de manière bien plus uniforme et offre de meilleures reconstructions visuelles.

Les traversées latentes (cf. `results/traversal_*.png`) confirment que le GVAE encode des variations continues et interprétables, là où le DirVAE produit des variations plus abruptes.

---

## Visuels

Les figures générées sont sauvegardées dans `results/` :

- **`loss_curves.png`** — évolution des losses et du SSIM sur 60 epochs
- **`reconstructions.png`** — original vs GVAE vs DirVAE côte à côte
- **`umap_comparison.png`** — structure des espaces latents en 2D
- **`latent_variance.png`** — activité des dimensions latentes
- **`traversal_gvae.png`** / **`traversal_dirvae.png`** — traversées latentes dim par dim
- **`synthetic_samples.png`** — lésions synthétiques générées depuis le prior