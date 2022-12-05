import os
import yaml
import itertools


def main():
    params = {
        "l_win": [115, 117, 120, 123, 125],
        "batch_size": [64, 128],
        "num_workers": [2],
        "n_head": [23],
        "dff": [128, 256],
        "num_layers": [2, 3, 4],
        "lr": [0.001],
        "weight_decay": [0.0005],
        "n_epochs": [2],
        "dropout": [0.1],
        "kernel_size": [3, 5, 7],
    }

    keys, values = zip(*params.items())
    combs = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"GENERATING {len(combs)} NEW CONFIGS ...")

    for comb in combs:
        filename = "{}stacks_{}nhead_{}lwin_{}lr_{}dff_{}batch_{}epcs_{}dropout".format(
            comb["num_layers"],
            comb["n_head"],
            comb["l_win"],
            comb["lr"],
            comb['dff'],
            comb["batch_size"],
            comb["n_epochs"],
            comb['dropout'],
        ).replace(".", "_")
        config_path = os.path.join("configs/", "{}.yml".format(filename))
        config = {
            "experiment": filename,
            # "d_model": 16, ##FD001 
            # "d_model": 23, ##FD002
            # "d_model": 18, ##FD003
            "d_model": 23, ##FD004 
	    "model": 1, #denote Transformer
	    #"model": 2, #denote hybrid model
        }
        config.update(comb)
        print(filename)
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False)

    print("DONE.")


if __name__ == "__main__":
    main()
