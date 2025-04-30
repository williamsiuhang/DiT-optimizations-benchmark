import torch
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import argparse
import json

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX # type: ignore
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC # type: ignore # type: ignore
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

def create_npz_from_sample_folder(sample_dir, folders, out_file, num=50000, ):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    k = num // len(folders) + 1
    perm = torch.randperm(200)
    idx = perm[:k]
    for folder in tqdm(folders, desc="Building .npz"):
        folder_path = os.path.join(sample_dir, folder)
        for j in idx:
            sample_pil = Image.open(f"{folder_path}/{j:03d}.jpg")
            sample_pil = center_crop_arr(sample_pil, args.image_size)
            sample_np = np.asarray(sample_pil).astype(np.uint8)
            if sample_np.ndim == 2:
                sample_np = np.tile(sample_np[:, :, np.newaxis], (1, 1, 3))
            assert sample_np.shape == (args.image_size, args.image_size, 3), f"Image shape mismatch: {sample_np.shape}, folder {folder}, idx {j}"
            samples.append(sample_np)
    samples = np.stack(samples)
    npz_path = f"{out_file}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path

def main(args):
    """
    Run sampling.
    """
    seed = args.global_seed
    torch.manual_seed(seed)

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
    
    create_npz_from_sample_folder(args.sample_dir, classes, args.out_file, args.num)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample from a DiT model.")
    parser.add_argument("--global_seed", type=int, default=42, help="Global seed for random number generation.")
    parser.add_argument("--tf32", action="store_true", help="Use TF32 for faster computation.")
    parser.add_argument("--sample_dir", type=str, required=True, help="Directory containing the sample images.")
    parser.add_argument("--out_file", type=str, required=True, help="Output file name for the .npz file.")
    parser.add_argument("--num", type=int, default=50000, help="Number of samples to include in the .npz file.")
    parser.add_argument("--image_size", type=int, default=256, help="Size of the images to be saved in the .npz file.")
    args = parser.parse_args()
    
    main(args)