import argparse
import torch
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Prune a model")
parser.add_argument("model_prune", type=str, help="Path to model to prune")
parser.add_argument("prune_output", type=str, help="Path to pruned ckpt output")
parser.add_argument("--half", action="store_true", help="Save weights in half precision.")
args = parser.parse_args()

print("Loading model...")
model_prune = torch.load(args.model_prune)
theta_prune = model_prune["state_dict"]
theta = {}

print("Pruning model...")
for key in tqdm(theta_prune.keys(), desc="Pruning keys"):
    if "model" in key:
        theta.update({key: theta_prune[key]})

del theta_prune

if args.half:
    print("Halving model...")
    state_dict = {k: v.half() for k, v in tqdm(theta.items(), desc="Halving weights")}
else:
    state_dict = theta

del theta

print("Saving pruned model...")

torch.save({"state_dict": state_dict}, args.prune_output)

del state_dict

print("Done pruning!")