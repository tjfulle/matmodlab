import random
ALICE_BLUE = 'AliceBlue'
DARK_OLIVE_GREEN = 'DarkOliveGreen'
INDIGO = 'Indigo'
MEDIUM_PURPLE = 'MediumPurple'
PURPLE = 'Purple'
ANTIQUE_WHITE = 'AntiqueWhite'
DARK_ORANGE = 'DarkOrange'
IVORY = 'Ivory'
MEDIUM_SEA_GREEN = 'MediumSeaGreen'
RED = 'Red'
AQUA = 'Aqua'
DARK_ORCHID = 'DarkOrchid'
KHAKI = 'Khaki'
MEDIUM_SLATE_BLUE = 'MediumSlateBlue'
ROSY_BROWN = 'RosyBrown'
AQUA_MARINE = 'AquaMarine'
DARK_RED = 'DarkRed'
LAVENDER = 'Lavender'
MEDIUM_SPRING_GREEN = 'MediumSpringGreen'
ROYAL_BLUE = 'RoyalBlue'
AZURE = 'Azure'
DARK_SALMON = 'DarkSalmon'
LAVENDER_BLUSH = 'LavenderBlush'
MEDIUM_TURQUOISE = 'MediumTurquoise'
SADDLE_BROWN = 'SaddleBrown'
BEIGE = 'Beige'
DARK_SEA_GREEN = 'DarkSeaGreen'
LAWN_GREEN = 'LawnGreen'
MEDIUM_VIOLET_RED = 'MediumVioletRed'
SALMON = 'Salmon'
BISQUE = 'Bisque'
DARK_SLATE_BLUE = 'DarkSlateBlue'
LEMON_CHIFFON = 'LemonChiffon'
MIDNIGHT_BLUE = 'MidnightBlue'
SANDY_BROWN = 'SandyBrown'
BLACK = 'Black'
DARK_SLATE_GRAY = 'DarkSlateGray'
LIGHT_BLUE = 'LightBlue'
MINT_CREAM = 'MintCream'
SEA_GREEN = 'SeaGreen'
BLANCHED_ALMOND = 'BlanchedAlmond'
DARK_TURQUOISE = 'DarkTurquoise'
LIGHT_CORAL = 'LightCoral'
MISTY_ROSE = 'MistyRose'
SEA_SHELL = 'SeaShell'
BLUE = 'Blue'
DARK_VIOLET = 'DarkViolet'
LIGHT_CYAN = 'LightCyan'
MOCCASIN = 'Moccasin'
SIENNA = 'Sienna'
BLUE_VIOLET = 'BlueViolet'
DEEP_PINK = 'DeepPink'
LIGHT_GOLDENROD_YELLOW = 'LightGoldenrodYellow'
NAVAJO_WHITE = 'NavajoWhite'
SILVER = 'Silver'
BROWN = 'Brown'
DEEP_SKY_BLUE = 'DeepSkyBlue'
LIGHT_GRAY = 'LightGray'
NAVY = 'Navy'
SKY_BLUE = 'SkyBlue'
BURLY_WOOD = 'BurlyWood'
DIM_GRAY = 'DimGray'
LIGHT_GREEN = 'LightGreen'
OLD_LACE = 'OldLace'
SLATE_BLUE = 'SlateBlue'
CADET_BLUE = 'CadetBlue'
DODGER_BLUE = 'DodgerBlue'
LIGHT_PINK = 'LightPink'
OLIVE = 'Olive'
SLATE_GRAY = 'SlateGray'
CHARTREUSE = 'Chartreuse'
FIRE_BRICK = 'FireBrick'
LIGHT_SALMON = 'LightSalmon'
OLIVE_DRAB = 'OliveDrab'
SNOW = 'Snow'
CHOCOLATE = 'Chocolate'
FLORAL_WHITE = 'FloralWhite'
LIGHT_SEA_GREEN = 'LightSeaGreen'
ORANGE = 'Orange'
SPRING_GREEN = 'SpringGreen'
CORAL = 'Coral'
FOREST_GREEN = 'ForestGreen'
LIGHT_SKY_BLUE = 'LightSkyBlue'
ORANGE_RED = 'OrangeRed'
STEEL_BLUE = 'SteelBlue'
CORN_FLOWER_BLUE = 'CornFlowerBlue'
FUCHSIA = 'Fuchsia'
LIGHT_SLATE_GRAY = 'LightSlateGray'
ORCHID = 'Orchid'
TAN = 'Tan'
CORNSILK = 'Cornsilk'
GAINSBORO = 'Gainsboro'
LIGHT_STEEL_BLUE = 'LightSteelBlue'
PALE_GOLDEN_ROD = 'PaleGoldenRod'
TEAL = 'Teal'
CRIMSON = 'Crimson'
GHOST_WHITE = 'GhostWhite'
LIGHT_YELLOW = 'LightYellow'
PALE_GREEN = 'PaleGreen'
THISTLE = 'Thistle'
CYAN = 'Cyan'
GOLD = 'Gold'
LIME = 'Lime'
PALE_TURQUOISE = 'PaleTurquoise'
TOMATO = 'Tomato'
DARK_BLUE = 'DarkBlue'
GOLDEN_ROD = 'GoldenRod'
LIME_GREEN = 'LimeGreen'
PALE_VIOLET_RED = 'PaleVioletRed'
TURQUOISE = 'Turquoise'
DARK_CYAN = 'DarkCyan'
GRAY = 'Gray'
LINEN = 'Linen'
PAPAYA_WHIP = 'PapayaWhip'
VIOLET = 'Violet'
DARK_GOLDEN_ROD = 'DarkGoldenRod'
GREEN = 'Green'
MAGENTA = 'Magenta'
PEACH_PUFF = 'PeachPuff'
WHEAT = 'Wheat'
DARK_GRAY = 'DarkGray'
GREEN_YELLOW = 'GreenYellow'
MAROON = 'Maroon'
PERU = 'Peru'
WHITE = 'White'
DARK_GREEN = 'DarkGreen'
HONEY_DEW = 'HoneyDew'
MEDIUM_AQUA_MARINE = 'MediumAquaMarine'
PINK = 'Pink'
WHITE_SMOKE = 'WhiteSmoke'
DARK_KHAKI = 'DarkKhaki'
HOT_PINK = 'HotPink'
MEDIUM_BLUE = 'MediumBlue'
PLUM = 'Plum'
YELLOW = 'Yellow'
DARK_MAGENTA = 'DarkMagenta'
INDIAN_RED = 'IndianRed'
MEDIUM_ORCHID = 'MediumOrchid'
POWDER_BLUE = 'PowderBlue'
YELLOW_GREEN = 'YellowGreen'
COLORS = [ALICE_BLUE, DARK_OLIVE_GREEN, INDIGO, MEDIUM_PURPLE, PURPLE,
          MEDIUM_SEA_GREEN, RED, AQUA, DARK_ORCHID, KHAKI, MEDIUM_SLATE_BLUE,
          AQUA_MARINE, DARK_RED, LAVENDER, MEDIUM_SPRING_GREEN, ROYAL_BLUE,
          DARK_SALMON, LAVENDER_BLUSH, MEDIUM_TURQUOISE, SADDLE_BROWN, BEIGE,
          LAWN_GREEN, MEDIUM_VIOLET_RED, SALMON, BISQUE, DARK_SLATE_BLUE,
          MIDNIGHT_BLUE, SANDY_BROWN, BLACK, DARK_SLATE_GRAY, LIGHT_BLUE,
          SEA_GREEN, BLANCHED_ALMOND, DARK_TURQUOISE, LIGHT_CORAL,
          SEA_SHELL, BLUE, DARK_VIOLET, LIGHT_CYAN, MOCCASIN, SIENNA,
          DEEP_PINK, LIGHT_GOLDENROD_YELLOW, SILVER, BROWN, DEEP_SKY_BLUE,
          NAVY, SKY_BLUE, BURLY_WOOD, DIM_GRAY, LIGHT_GREEN, OLD_LACE,
          CADET_BLUE, DODGER_BLUE, LIGHT_PINK, OLIVE, SLATE_GRAY, CHARTREUSE,
          LIGHT_SALMON, OLIVE_DRAB, SNOW, CHOCOLATE, LIGHT_SEA_GREEN, ORANGE,
          CORAL, FOREST_GREEN, LIGHT_SKY_BLUE, ORANGE_RED, STEEL_BLUE,
          FUCHSIA, LIGHT_SLATE_GRAY, ORCHID, TAN, CORNSILK, GAINSBORO,
          PALE_GOLDEN_ROD, TEAL, CRIMSON, LIGHT_YELLOW, PALE_GREEN, THISTLE,
          GOLD, LIME, PALE_TURQUOISE, TOMATO, DARK_BLUE, GOLDEN_ROD,
          PALE_VIOLET_RED, TURQUOISE, DARK_CYAN, GRAY, LINEN, PAPAYA_WHIP,
          DARK_GOLDEN_ROD, GREEN, MAGENTA, PEACH_PUFF, WHEAT, DARK_GRAY,
          MAROON, PERU, DARK_GREEN, HONEY_DEW, MEDIUM_AQUA_MARINE, PINK,
          HOT_PINK, MEDIUM_BLUE, PLUM, YELLOW, DARK_MAGENTA, INDIAN_RED]


def random_color(lower=False):
    c = COLORS[random.randint(0, len(COLORS)-1)]
    if lower:
        return c.lower()
    return c


def next_color(i=0, _i=[0]):
    if i:
        _i[0] = i
    c = COLORS[_i[0] % (len(COLORS) - 1)]
    _i[0] += 1
    return c
