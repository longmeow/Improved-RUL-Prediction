import os
import yaml
import itertools


def main():
    params = {
        "l_win": [120],
        "batch_size": [64],
        "num_workers": [2],
        "n_head": [23],
        "dff": [256],
        "num_layers": [3],
        "lr": [0.0017423754808603394],
        "weight_decay": [0.0012942104039810626],
        "n_epochs": [1],
        "dropout": [0.14232544872612077],
        "kernel_size": [3, 4, 5, 6, 7, 8],
    }

    keys, values = zip(*params.items())
    combs = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"GENERATING {len(combs)} NEW CONFIGS ...")

    for comb in combs:
        filename = "{}stacks_{}nhead_{}lwin_{}lr_{}dff_{}batch_{}epcs_{}dropout_{}kernelsize".format(
            comb["num_layers"],
            comb["n_head"],
            comb["l_win"],
            comb["lr"],
            comb['dff'],
            comb["batch_size"],
            comb["n_epochs"],
            comb['dropout'],
            comb["kernel_size"]
        ).replace(".", "_")
        config_path = os.path.join("configs/", "{}.yml".format(filename))
        config = {
            "experiment": filename,
            ##FD001
            # "dataset": 1,
            # "d_model": 16, 
            ##FD002 
            # "dataset": 2,
            # "d_model": 23, 
            #FD003
            # "dataset": 3,
            # "d_model": 18, 
            # ##FD004 
            "dataset": 4,
            "d_model": 23, 
            #   1 denote Transformer
	        # "model": 1, 
            # 2 denote hybrid model 
	        "model": 2 
        }
        config.update(comb)
        
        #check if config legit
        for val in params["n_head"]:
            assert(config['d_model'] % val == 0)
            
        print(filename)
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False)

    print("DONE.")


if __name__ == "__main__":
    main()
