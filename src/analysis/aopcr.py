import os
import sys
import argparse
import matplotlib as mpl

sys.path.append("..")
from analyzer import AOPCRCalculatorPatchDecomp, AOPCRCalculatorTFT
from utils import suppress_output_except_tqdm, make_folder, save_args, load_epf_dataset


# Command-line parameters
parser = argparse.ArgumentParser()
parser.add_argument(
    "--path", type=str, required=True, help="path to the result folder to analyze"
)
parser.add_argument(
    "--method",
    type=str,
    choices=["PatchDecomp", "TFT-point", "TFT-patch", "random"],
    default="PatchDecomp",
    help="how to remove important inputs",
)
parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
parser.add_argument(
    "--ks",
    nargs="*",
    type=float,
    default=[5.0, 7.5, 10.0, 12.5, 15.0],
    help="list of removal ratio",
)
parser.add_argument(
    "--replace",
    type=str,
    choices=[
        "zero",
        "max-minus",
        "swap",
        "local-mean",
        "global-mean",
        "local-noise",
        "global-noise",
    ],
    default="global-mean",
    help="how to replace the removed inputs",
)
parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument(
    "--testsize", type=int, default=10464, help="length of data to test"
)
parser.add_argument(
    "--unique_ids",
    nargs="*",
    type=str,
    default=["BE"],
    help="unique ID",
)
args = parser.parse_args()

# CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
# Font
mpl.rcParams["font.family"] = "Times New Roman"

# Output folder
output_path = make_folder(args.path + f"/analysis/AOPCR/{args.method}/")

# Save command-line parameters
save_args(args.__dict__, output_path)


# Load datasets
df, static_df = load_epf_dataset(args.unique_ids)

# Initialize calculator
with suppress_output_except_tqdm() as _:
    if "TFT" in args.method:
        calculator = AOPCRCalculatorTFT(
            method=args.method, dirname=args.path, seed=args.seed
        )
    else:
        calculator = AOPCRCalculatorPatchDecomp(
            method=args.method, dirname=args.path, seed=args.seed
        )
    calculator.prepare(df, static_df)

# Calculate AOPCR
calculator.calculate_aopcr(
    test_size=args.testsize, ks=args.ks, replace=args.replace, path=output_path
)

print("\nThe results were saved in", output_path)
