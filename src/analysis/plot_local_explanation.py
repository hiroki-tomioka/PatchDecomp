import os
import sys
import argparse
import matplotlib as mpl

sys.path.append("..")
from analyzer import LocalExplanationDrawer
from utils import suppress_output_except_tqdm, make_folder, save_args, load_epf_dataset


# Command-line parameters
parser = argparse.ArgumentParser()
parser.add_argument(
    "--path", type=str, required=True, help="path to the result folder to analyze"
)
parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
parser.add_argument("--seed", type=int, default=0, help="random seed")
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
output_path = make_folder(args.path + f"/analysis/local_explanation/")

# Save command-line parameters
save_args(args.__dict__, output_path)


# Load datasets
df, static_df = load_epf_dataset(args.unique_ids)

# Initialize drawer
with suppress_output_except_tqdm() as _:
    drawer = LocalExplanationDrawer(
        method="PatchDecomp", dirname=args.path, seed=args.seed
    )
    drawer.prepare(df, static_df)

# Plot local explanations
drawer.simulate(test_size=10464, path=output_path)

print("\nThe results were saved in", output_path)
