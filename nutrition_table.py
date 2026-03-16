"""
nutrition_table.py — Table nutritionnelle pour le calcul des calories.

Associe chaque label alimentaire (Food-101 + custom) à ses données
nutritionnelles : kcal/100g, macronutriments et portion standard.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class NutritionInfo:
    """Informations nutritionnelles pour 100g d'un aliment."""
    kcal_per_100g: float
    protein_g: float        # Protéines (g/100g)
    carbs_g: float           # Glucides (g/100g)
    fat_g: float             # Lipides (g/100g)
    fiber_g: float = 0.0     # Fibres (g/100g)
    default_portion_g: float = 100.0  # Portion standard


# ──────────────────────────────────────────────────────────────────────
# Table nutritionnelle — Food-101 (101 classes)
# Sources : USDA FoodData Central, CIQUAL (ANSES)
# ──────────────────────────────────────────────────────────────────────
NUTRITION_DB: dict[str, NutritionInfo] = {
    # --- Viandes & Protéines ---
    "steak":                 NutritionInfo(271, 26.0, 0.0, 18.0, 0.0, 150),
    "grilled_salmon":        NutritionInfo(208, 20.4, 0.0, 13.4, 0.0, 150),
    "fried_rice":            NutritionInfo(163, 4.1, 23.5, 5.8, 0.8, 250),
    "chicken_curry":         NutritionInfo(150, 14.0, 7.0, 7.5, 1.0, 300),
    "chicken_wings":         NutritionInfo(290, 27.4, 0.0, 19.5, 0.0, 150),
    "chicken_quesadilla":    NutritionInfo(216, 13.5, 18.0, 10.0, 1.0, 200),
    "pork_chop":             NutritionInfo(231, 25.7, 0.0, 13.5, 0.0, 150),
    "prime_rib":             NutritionInfo(290, 23.0, 0.0, 22.0, 0.0, 200),
    "filet_mignon":          NutritionInfo(267, 26.0, 0.0, 17.5, 0.0, 170),
    "pulled_pork_sandwich":  NutritionInfo(210, 16.0, 18.0, 8.0, 1.0, 250),
    "hamburger":             NutritionInfo(295, 17.0, 24.0, 14.0, 1.2, 200),
    "hot_dog":               NutritionInfo(290, 10.0, 24.0, 18.0, 0.5, 100),

    # --- Poissons & Fruits de mer ---
    "fish_and_chips":        NutritionInfo(250, 15.5, 22.0, 11.0, 1.5, 300),
    "sashimi":               NutritionInfo(143, 24.0, 0.0, 5.0, 0.0, 120),
    "ceviche":               NutritionInfo(90, 15.0, 4.0, 1.5, 0.5, 150),
    "shrimp_and_grits":      NutritionInfo(170, 12.0, 17.0, 6.0, 0.5, 250),
    "lobster_bisque":        NutritionInfo(122, 7.0, 8.0, 7.0, 0.2, 250),
    "clam_chowder":          NutritionInfo(95, 4.5, 10.0, 4.0, 0.5, 250),
    "mussels":               NutritionInfo(86, 12.0, 3.7, 2.2, 0.0, 200),
    "crab_cakes":            NutritionInfo(196, 14.0, 10.0, 11.0, 0.3, 120),
    "lobster_roll_sandwich": NutritionInfo(190, 15.0, 20.0, 5.0, 1.0, 200),
    "tuna_tartare":          NutritionInfo(130, 23.0, 1.0, 3.5, 0.0, 120),

    # --- Pâtes & Riz ---
    "spaghetti_bolognese":   NutritionInfo(160, 8.0, 18.0, 6.0, 1.5, 350),
    "spaghetti_carbonara":   NutritionInfo(190, 10.0, 20.0, 8.0, 0.8, 350),
    "lasagna":               NutritionInfo(135, 8.5, 12.0, 6.0, 1.0, 350),
    "ravioli":               NutritionInfo(175, 8.0, 22.0, 6.0, 1.2, 250),
    "gnocchi":               NutritionInfo(133, 3.0, 27.0, 1.0, 1.5, 250),
    "macaroni_and_cheese":   NutritionInfo(300, 11.0, 28.0, 16.0, 0.7, 250),
    "pad_thai":              NutritionInfo(170, 8.0, 24.0, 5.0, 1.0, 300),
    "pho":                   NutritionInfo(45, 4.0, 5.0, 1.0, 0.3, 500),
    "ramen":                 NutritionInfo(190, 8.0, 25.0, 7.0, 1.0, 500),
    "risotto":               NutritionInfo(140, 3.5, 20.0, 5.0, 0.5, 300),
    "paella":                NutritionInfo(150, 8.0, 18.0, 5.0, 1.0, 350),
    "bibimbap":              NutritionInfo(130, 7.0, 18.0, 3.5, 2.0, 400),

    # --- Pizzas & Pain ---
    "pizza":                 NutritionInfo(266, 11.4, 33.0, 10.4, 2.0, 200),
    "bruschetta":            NutritionInfo(170, 4.0, 20.0, 8.0, 1.5, 100),
    "garlic_bread":          NutritionInfo(350, 7.0, 40.0, 18.0, 2.0, 80),
    "bread_pudding":         NutritionInfo(230, 5.0, 33.0, 9.0, 0.8, 150),
    "french_toast":          NutritionInfo(230, 7.0, 24.0, 12.0, 0.5, 150),
    "grilled_cheese_sandwich": NutritionInfo(330, 12.0, 28.0, 19.0, 1.0, 150),
    "club_sandwich":         NutritionInfo(210, 15.0, 16.0, 10.0, 1.5, 250),

    # --- Salades & Légumes ---
    "caesar_salad":          NutritionInfo(127, 7.0, 7.0, 8.0, 2.0, 200),
    "greek_salad":           NutritionInfo(80, 3.5, 5.0, 5.5, 1.5, 250),
    "caprese_salad":         NutritionInfo(140, 8.0, 3.0, 11.0, 0.5, 200),
    "edamame":               NutritionInfo(121, 12.0, 9.0, 5.0, 5.0, 100),
    "seaweed_salad":         NutritionInfo(70, 2.0, 9.0, 2.5, 0.5, 100),
    "beet_salad":            NutritionInfo(70, 2.0, 10.0, 2.5, 2.0, 200),

    # --- Soupes ---
    "french_onion_soup":     NutritionInfo(55, 2.5, 6.0, 2.0, 0.5, 300),
    "hot_and_sour_soup":     NutritionInfo(40, 3.0, 4.0, 1.5, 0.5, 300),
    "miso_soup":             NutritionInfo(30, 2.5, 3.0, 1.0, 0.5, 300),
    "wonton_soup":           NutritionInfo(46, 3.0, 5.5, 1.0, 0.3, 350),

    # --- Petit-déjeuner ---
    "pancakes":              NutritionInfo(227, 6.0, 28.0, 10.0, 1.0, 150),
    "waffles":               NutritionInfo(291, 8.0, 33.0, 14.0, 1.0, 120),
    "omelette":              NutritionInfo(154, 11.0, 1.0, 12.0, 0.0, 180),
    "scrambled_eggs":        NutritionInfo(148, 10.0, 2.0, 11.0, 0.0, 150),
    "eggs_benedict":         NutritionInfo(250, 14.0, 18.0, 14.0, 0.5, 250),
    "breakfast_burrito":     NutritionInfo(200, 10.0, 20.0, 9.0, 2.0, 250),
    "deviled_eggs":          NutritionInfo(195, 10.0, 2.0, 16.0, 0.0, 80),
    "huevos_rancheros":      NutritionInfo(140, 8.0, 10.0, 8.0, 2.5, 300),

    # --- Entrées & Amuse-bouches ---
    "spring_rolls":          NutritionInfo(200, 5.0, 25.0, 9.0, 1.5, 100),
    "samosa":                NutritionInfo(260, 5.5, 30.0, 13.0, 2.0, 100),
    "gyoza":                 NutritionInfo(230, 7.0, 25.0, 11.0, 1.0, 100),
    "dumplings":             NutritionInfo(210, 8.0, 24.0, 9.0, 1.0, 120),
    "takoyaki":              NutritionInfo(180, 6.0, 22.0, 7.0, 0.5, 100),
    "falafel":               NutritionInfo(333, 13.0, 32.0, 18.0, 5.0, 100),
    "hummus":                NutritionInfo(166, 8.0, 14.0, 10.0, 6.0, 80),
    "guacamole":             NutritionInfo(160, 2.0, 9.0, 15.0, 7.0, 80),
    "nachos":                NutritionInfo(350, 8.0, 36.0, 19.0, 3.0, 200),
    "onion_rings":           NutritionInfo(332, 4.5, 38.0, 18.0, 2.0, 120),
    "french_fries":          NutritionInfo(312, 3.4, 41.0, 15.0, 3.8, 150),
    "cheese_plate":          NutritionInfo(350, 22.0, 2.0, 28.0, 0.0, 100),
    "escargots":             NutritionInfo(90, 16.0, 2.0, 1.4, 0.0, 100),
    "foie_gras":             NutritionInfo(462, 11.0, 4.0, 44.0, 0.0, 50),
    "baby_back_ribs":        NutritionInfo(280, 20.0, 5.0, 20.0, 0.0, 250),
    "poutine":               NutritionInfo(196, 6.0, 23.0, 9.0, 2.0, 350),

    # --- Plats principaux ---
    "tacos":                 NutritionInfo(226, 12.0, 20.0, 11.0, 2.5, 150),
    "burritos":              NutritionInfo(190, 9.0, 22.0, 7.5, 2.0, 300),
    "gyros":                 NutritionInfo(240, 13.0, 20.0, 12.0, 2.0, 250),
    "peking_duck":           NutritionInfo(225, 18.0, 5.0, 15.0, 0.0, 200),
    "frozen_yogurt":         NutritionInfo(127, 3.0, 22.0, 3.5, 0.0, 150),

    # --- Desserts ---
    "chocolate_cake":        NutritionInfo(367, 5.0, 50.0, 17.0, 2.5, 100),
    "cheesecake":            NutritionInfo(321, 5.5, 26.0, 22.0, 0.3, 120),
    "carrot_cake":           NutritionInfo(340, 4.0, 44.0, 17.0, 1.5, 100),
    "red_velvet_cake":       NutritionInfo(365, 4.0, 46.0, 19.0, 1.0, 100),
    "strawberry_shortcake":  NutritionInfo(250, 3.0, 35.0, 11.0, 1.5, 120),
    "apple_pie":             NutritionInfo(237, 2.0, 34.0, 11.0, 1.5, 150),
    "tiramisu":              NutritionInfo(283, 5.0, 30.0, 16.0, 0.3, 120),
    "creme_brulee":          NutritionInfo(300, 4.0, 28.0, 19.0, 0.0, 100),
    "panna_cotta":           NutritionInfo(230, 3.5, 22.0, 14.0, 0.0, 120),
    "chocolate_mousse":      NutritionInfo(270, 5.0, 24.0, 17.0, 2.0, 100),
    "macarons":              NutritionInfo(400, 6.0, 52.0, 18.0, 2.0, 40),
    "churros":               NutritionInfo(370, 4.0, 42.0, 21.0, 1.5, 80),
    "donuts":                NutritionInfo(400, 5.0, 50.0, 20.0, 1.0, 60),
    "baklava":               NutritionInfo(428, 6.0, 48.0, 25.0, 2.0, 60),
    "cannoli":               NutritionInfo(340, 7.0, 36.0, 19.0, 1.0, 80),
    "cup_cakes":             NutritionInfo(340, 4.0, 48.0, 15.0, 0.5, 70),
    "ice_cream":             NutritionInfo(207, 3.5, 24.0, 11.0, 0.5, 100),

    # --- Sushi ---
    "sushi":                 NutritionInfo(150, 6.0, 20.0, 5.0, 0.3, 200),

    # --- Fruits ---
    "strawberries":          NutritionInfo(32, 0.7, 7.7, 0.3, 2.0, 150),
    "fruit_salad":           NutritionInfo(50, 0.5, 12.0, 0.2, 1.5, 200),

    # --- Divers ---
    "oysters":               NutritionInfo(81, 9.0, 5.0, 2.5, 0.0, 120),
    "fried_calamari":        NutritionInfo(175, 15.0, 7.0, 10.0, 0.3, 120),
    "scallops":              NutritionInfo(111, 21.0, 3.0, 1.0, 0.0, 120),
    "beef_carpaccio":        NutritionInfo(143, 22.0, 0.0, 6.0, 0.0, 100),
    "beef_tartare":          NutritionInfo(170, 21.0, 1.0, 9.0, 0.0, 120),
    "croque_madame":         NutritionInfo(300, 17.0, 20.0, 17.0, 1.0, 250),
    "cup_cakes":             NutritionInfo(340, 4.0, 48.0, 15.0, 0.5, 70),
    "macaroni_and_cheese":   NutritionInfo(300, 11.0, 28.0, 16.0, 0.7, 250),
}


# ──────────────────────────────────────────────────────────────────────
# Valeur par défaut si un aliment n'est pas dans la table
# ──────────────────────────────────────────────────────────────────────
DEFAULT_NUTRITION = NutritionInfo(
    kcal_per_100g=200,
    protein_g=8.0,
    carbs_g=25.0,
    fat_g=8.0,
    fiber_g=2.0,
    default_portion_g=150,
)


def get_nutrition(label: str) -> NutritionInfo:
    """
    Retrouve les données nutritionnelles pour un label donné.
    Normalise le label (minuscules, espaces → underscores).
    Retourne DEFAULT_NUTRITION si le label est inconnu.
    """
    normalized = label.lower().strip().replace(" ", "_").replace("-", "_")
    return NUTRITION_DB.get(normalized, DEFAULT_NUTRITION)


def calculate_calories(label: str, portion_g: float = 100.0) -> dict:
    """
    Calcule les informations nutritionnelles pour une portion donnée.

    Args:
        label: Nom de l'aliment prédit par le modèle.
        portion_g: Poids de la portion en grammes.

    Returns:
        Dictionnaire avec les valeurs nutritionnelles ajustées.
    """
    info = get_nutrition(label)
    ratio = portion_g / 100.0

    return {
        "label": label,
        "portion_g": portion_g,
        "calories": round(info.kcal_per_100g * ratio, 1),
        "protein_g": round(info.protein_g * ratio, 1),
        "carbs_g": round(info.carbs_g * ratio, 1),
        "fat_g": round(info.fat_g * ratio, 1),
        "fiber_g": round(info.fiber_g * ratio, 1),
    }


def list_available_foods() -> list[str]:
    """Retourne la liste triée de tous les aliments disponibles."""
    return sorted(NUTRITION_DB.keys())
