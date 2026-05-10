# DirVAE vs GVAE — Comparaison équitable sur DeepLesion

Projet réalisé dans le cadre du cours **GenAI (ING3 — MLOps)**.

## Objectif

Comparer deux variantes de VAE (Variational Autoencoder) pour la modélisation de lésions issues du dataset médical **DeepLesion**, avec une architecture **strictement identique** — seul le prior latent change :

- **GVAE** : VAE standard avec prior gaussien N(0, I)
- **DirVAE** : VAE avec prior de Dirichlet, supposé induire un meilleur disentanglement (Keel et al.)

L'évaluation porte sur **tous les types de lésions** (8 classes d'organes) et couvre quatre axes :
1. Qualité de reconstruction (SSIM, L1)
2. Structure de l'espace latent (UMAP)
3. Clustering non supervisé (K-Means, CLASSIX)
4. Disentanglement (MIG, directions latentes, traversées)

---

## Structure du repo

```
GenAI_DeepLesion/
├── main.ipynb          # Notebook principal (tout le code)
```

Tout le pipeline est contenu dans **un seul notebook** (`main.ipynb`), structuré en 14 sections numérotées.

---

## Dataset

**DeepLesion** (NIH) — dataset de tomodensitométries (CT scans) annotées avec des bounding boxes de lésions.

- Fichier CSV : `DL_info.csv`
- Images : PNG 16 bits encodées en Hounsfield Units (HU) avec un offset de 32768
- **Tous les types de lésions** sont utilisés (8 classes : Bone, Abdomen, Mediastinum, Liver, Lung, Kidney, Soft Tissue, Pelvis)
- Preprocessing :
  - Crop sur la bounding box (+ padding de 10px)
  - Fenêtrage HU : [-1000, 500]
  - Resize en **64×64** pixels
  - Normalisation en [0, 1]

Split (splits officiels DeepLesion) :
- **Train** : `Train_Val_Test ∈ {1, 2}`
- **Val** : `Train_Val_Test == 3`
- **Probe set** : toutes les lésions avec `Coarse_lesion_type != -1` (labels organe disponibles)

---

## Architecture des modèles

Les deux modèles partagent **exactement le même squelette convolutif**. Seules les têtes latentes diffèrent.

| Composant | Détails |
|-----------|---------|
| Encodeur | 7 couches Conv + GELU + BatchNorm, stride 2 pour le downsampling, conv 8×8 finale → 1×1 |
| Décodeur | Linear → ConvTranspose + upsampling bilinéaire (évite les checkerboard artifacts) |
| BASE | 64 |
| Dimension latente | **256** |
| Bottleneck | 2048 (= 32 × BASE) |
| Paramètres | ~76.6M (DirVAE) / ~76.7M (GVAE) |

**GVAE** : têtes `mu_fc` + `logvar_fc`, reparamétrisation gaussienne standard, KL forme fermée.

**DirVAE** : tête `alpha_fc`, concentrations α = softplus(logits) + 0.5 (min α=0.5 pour éviter le collapse), reparamétrisation Dirichlet (`rsample`), KL analytique entre Dir(α_pred) et Dir(α_prior=0.6).

**Fonction de perte commune** :

```
L = 0.5 · L1 + 0.5 · (1 - SSIM) + β · KL
```

Avec β=10 et KL annealing linéaire de 0 à β sur les 50 premières epochs (sur 150 au total).

---

## Comment reproduire

Le notebook est conçu pour tourner sur **Kaggle** (GPU Tesla T4). Les chemins sont configurés dans la section 1 :

```python
DEEPLESION_CSV  = '/kaggle/input/.../DL_info.csv'
DEEPLESION_IMGS = '/kaggle/input/.../minideeplesion'
RESULTS_DIR     = '/kaggle/working/results'
```

**Dépendances** (installées dans le notebook) :
```bash
pip install pytorch-msssim umap-learn classixclustering
```

**Exécuter** : lancer toutes les cellules dans l'ordre. Le notebook entraîne les deux modèles séquentiellement (150 epochs chacun) puis génère automatiquement tous les visuels et métriques dans `results/`.

---

## Structure du notebook (14 sections)

1. Setup & dépendances
2. Dataset DeepLesion (chargement, splits, visualisation)
3. Architecture commune (blocs Conv, Encoder, Decoder, DirVAE, GVAE)
4. Boucle d'entraînement générique (loss, train, eval, fit)
5. Entraînement DirVAE
6. Entraînement GVAE
7. Encodage du probe set (extraction des latents)
8. UMAP 2D — visualisation comparative
9. Clustering : K-Means + CLASSIX
10. MIG (Mutual Information Gap)
11. Directions latentes (Lung vs Liver)
12. Latent traversal 5×5 (DirVAE et GVAE)
13. Tableau récap final
14. Discussion finale

---

## Résultats clés

### Tableau récap (150 epochs, base=64, latent=256)

| Métrique | DirVAE | GVAE |
|----------|--------|------|
| Recon val (↓) | 186.46 | **179.91** |
| SSIM val (↑) | **0.6651** | 0.6557 |
| NMI K-Means organe (↑) | 0.2816 | **0.3350** |
| NMI CLASSIX organe (↑) | 0.3694 | **0.4006** |
| MIG organe (↑) | 0.0017 | **0.0169** |

**Verdict : GVAE gagne 4/5 métriques.** Le prior Dirichlet n'apporte pas d'avantage clair à cette échelle. Finding négatif intéressant — potentiellement lié au court training ou à la granularité des labels organe.

### Limites

1. Un seul seed : scores avec haute variance.
2. Pas de vrais labels malignité (proxy `possibly_noisy` à zéro).
3. Subset `minideeplesion` uniquement.
4. Pas de tuning d'hyperparams indépendant par modèle.

### Perspectives

- Étendre à plus d'epochs pour vraie convergence
- Multi-seed (3–5 runs) pour intervalles de confiance
- Ajouter le stage lung-only avec AUC malignité (Keel et al.)
- Tuner `alpha_fill` et `beta` indépendamment pour chaque modèle
