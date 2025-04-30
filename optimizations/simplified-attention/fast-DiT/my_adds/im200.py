from torchvision.datasets import ImageFolder
import json

class ImageFolder200(ImageFolder):
    """
    Custom ImageFolder class that loads images from a directory.
    """
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)

    def find_classes(self, directory):
        """
        Get class indices and class names from json source.
        """
        class_to_idx = json.loads('{ "bulletproof_vest": 35, "basset": 19, "daisy": 53, "llama": 100, "sunscreen": 159, "lorikeet": 101, \
            "carousel": 38, "trifle": 179, "lakeside": 93, "walking_stick": 184, "drumstick": 58, "jellyfish": 89, "lighter": 99, "welsh_springer_spaniel": 190,\
            "bikini": 26, "typewriter_keyboard": 180, "model_t": 107, "traffic_light": 176, "moving_van": 112, "grasshopper": 72, "grand_piano": 70, \
            "diamondback": 54, "dishwasher": 55, "bee": 22, "wooden_spoon": 195, "tape_player": 161, "goblet": 69, "baseball": 18, "rugby_ball": 135, \
            "manhole_cover": 104, "leonberg": 96, "sarong": 137, "yawl": 197, "tiger": 169, "projector": 129, "artichoke": 10, "eft": 62, "sewing_machine": 142,\
            "dock": 56, "quill": 131, "cello": 40, "pretzel": 127, "whippet": 191, "bobsled": 28, "reel": 133, "water_tower": 188, "puck": 130, \
            "hand_blower": 78, "christmas_stocking": 45, "spindle": 152, "pomegranate": 125, "bathing_cap": 20, "harmonica": 79, "rifle": 134, \
            "recreational_vehicle": 132, "yellow_lady_s_slipper": 198, "cuirass": 52, "cucumber": 51, "washbasin": 186, "snowmobile": 148, "bookshop": 29,\
            "television": 163, "african_elephant": 4, "triceratops": 178, "mountain_tent": 110, "strawberry": 157, "beer_bottle": 23, "brabancon_griffon": 32, \
            "border_terrier": 30, "fur_coat": 68, "siamese_cat": 145, "motor_scooter": 109, "brain_coral": 33, "pencil_sharpener": 118, "tibetan_terrier": 168,\
            "wallet": 185,  "colobus": 49,   "trench_coat": 177,   "cicada": 46,   "appenzeller": 7,   "dugong": 59,   "chihuahua": 44,   "teddy": 162,   "backpack": 11,   "paper_towel": 116,   "howler_monkey": 85,   "theater_curtain": 166,   "guinea_pig": 75,   "sliding_door": 146,   "sea_cucumber": 140,   "saltshaker": 136,   "jeep": 88,   "stingray": 155,   "hourglass": 83,   "toilet_seat": 172,   "yurt": 199,   "mosquito_net": 108,   "house_finch": 84,   "beacon": 21,   "jacamar": 86,   "leaf_beetle": 94,   "sea_anemone": 139,   "lhasa": 98,   "soft_coated_wheaten_terrier": 149,   "mousetrap": 111,   "mailbox": 103,   "worm_fence": 196,   "cougar": 50,   "bighorn": 25,   "plow": 123,   "petri_dish": 120,   "shetland_sheepdog": 143,   "staffordshire_bullterrier": 153,   "leopard": 97,   "green_mamba": 73,   "chain_mail": 41,   "thatch": 165,   "throne": 167,   "cleaver": 47,   "toaster": 171,   "bagel": 13,   "pickup": 121,   "prayer_rug": 126,   "castle": 39,   "sea_snake": 141,   "electric_locomotive": 63,   "polaroid_camera": 124,   "wine_bottle": 194,   "bulbul": 34,   "bouvier_des_flandres": 31,   "vending_machine": 182,   "soup_bowl": 151,   "hamster": 77,   "windsor_tie": 193,   "chest": 42,   "norfolk_terrier": 114,   "chiffonier": 43,   "tench": 164,   "sombrero": 150,   "tow_truck": 174,   "jersey": 90,   "violin": 183,   "stone_wall": 156,   "leafhopper": 95,   "ballplayer": 15,   "water_snake": 187,   "granny_smith": 71,   "badger": 12,   "apron": 8,   "ambulance": 5,   "ox": 115,   "fig": 66,   "envelope": 64,   "missile": 106,   "cabbage_butterfly": 36,   "grocery_store": 74,   "perfume": 119,   "banjo": 17,   "parachute": 117,   "projectile": 128,   "holster": 81,   "earthstar": 61,   "band_aid": 16,   "fiddler_crab": 65,   "african_crocodile": 3,   "window_screen": 192,   "megalith": 105,   "torch": 173,   "zucchini": 200,   "planetarium": 122,   "king_crab": 92,   "toy_terrier": 175,   "snow_leopard": 147,   "afghan_hound": 1,   "hay": 80,   "hammerhead": 76,   "standard_schnauzer": 154,   "umbrella": 181,   "weasel": 189,   "dowitcher": 57,   "cannon": 37,   "bernese_mountain_dog": 24,   "lycaenid": 102,   "home_theater": 82,   "american_staffordshire_terrier": 6,   "scorpion": 138,   "ear": 60,   "studio_couch": 158,   "jaguar": 87,   "swab": 160,   "black_footed_ferret": 27,   "fireboat": 67,   "balance_beam": 14,   "arctic_fox": 9,   "coho": 48,   "tiger_cat": 170,   "shoji": 144,   "mushroom": 113,   "african_chameleon": 2,   "kimono": 91     }')

        classes = [k for k in class_to_idx.keys()]
        return classes, class_to_idx