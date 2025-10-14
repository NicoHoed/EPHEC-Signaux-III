from scipy.ndimage import median_filter, binary_opening, binary_closing
from skimage import color, filters


def pretraiter_image(img, taille_filtre=3, seuil=None):
    """
    Pipeline de prétraitement de base pour les images de clavier (ou similaires).

    Étapes :
    1. Convertir l’image RVB en niveaux de gris.
    2. Appliquer un filtrage médian pour réduire le bruit.
    3. Calculer un masque binaire (seuil automatique d’Otsu ou seuil manuel).
    4. Appliquer une ouverture morphologique (supprimer les petites taches)
       puis une fermeture (remplir les petits trous).

    Paramètres :
    ----------
    img : ndarray
        Image RVB d’entrée.
    taille_filtre : int, optionnel
        Taille du filtre médian (par défaut = 3).
    seuil : float ou None, optionnel
        Seuil manuel (entre 0 et 1). Si None, le seuil d’Otsu est utilisé automatiquement.

    Retour :
    -------
    nettoyee : ndarray (booléen)
        Image binaire (True/False) prête pour une analyse ultérieure.
    """
    # Étape 1 : Conversion en niveaux de gris
    gris = color.rgb2gray(img)

    # Étape 2 : Réduction du bruit avec un filtre médian
    filtree = median_filter(gris, size=taille_filtre)

    # Étape 3 : Binarisation avec le seuil d’Otsu (automatique) ou un seuil manuel
    if seuil is None:
        seuil = filters.threshold_otsu(filtree)
    binaire = (
        filtree < seuil
    )  # Les touches sont souvent plus sombres → inverser la logique si nécessaire

    # Étape 4 : Nettoyage morphologique — suppression du bruit, comblement des trous
    nettoyee = binary_closing(binary_opening(binaire))

    return nettoyee
