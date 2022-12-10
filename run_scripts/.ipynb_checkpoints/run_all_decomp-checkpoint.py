import subprocess
import os
import csv

seeds = list(range(10, 15))
models = [
    # "mlp",
    # "neumiss",
    "supmiwae",
    "supnotmiwae",
    # "gbt",
]

if not os.path.exists("../results/decomp.csv"):
    header = ["seed", "model", "score"]
    with open("../results/decomp.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)

for model in models:
    for seed in seeds:  
        print()
        print()
        print(f"RUNNING SEED={seed} \t MODEL={model}")
        print()
        print()
        subprocess.run(["python", "run_decomp.py", f"--seed={seed}", f"--model={model}"])