import subprocess
import itertools

import random

def generate_comb(r, n_comb=5, prop_train=0.7, seed=None):
    if seed is not None:
        random.seed(seed)

    perm = []
    total = len(r)
    num_train = int(round(prop_train * total))

    for _ in range(n_comb):
        r_copy = r[:]
        random.shuffle(r_copy)
        train = r_copy[:num_train]
        test = r_copy[num_train:]
        train_list, test_list = "", ""
        for i, t in enumerate(train):
            train_list += str(t)
            if i < len(train) - 1:
                train_list += ','
        for i, t in enumerate(test):
            test_list += str(t)
            if i < len(test) - 1:
                test_list += ','
        perm.append((train_list, test_list))

    return perm

def main():
    ids_r = [3,6,7,11,12,13,14,15,18,19]
    train_rosettes_options, test_rosettes_options = [], []
    comb = generate_comb(ids_r, n_comb=5, prop_train=0.7, seed=42)
    for (train, test) in comb:
        train_rosettes_options.append(train)
        test_rosettes_options.append(test)

    test_size_options = [0.05, 0.3]

    optimize_options = [False, False]
    plot_flag = False  #  plots each time
    shap_flag = False  # SHAP plots

    combinations = list(itertools.product(
        train_rosettes_options,
        test_rosettes_options,
        test_size_options,
        optimize_options
    ))

    for (train_ros, test_ros, t_size, opt) in combinations:
        cmd = ["python",
               "../src/random_forest/main.py",
               f"--train_rosettes={train_ros}",
               f"--test_rosettes={test_ros}",
               f"--test_size={t_size}",]
        if opt:
            pass
        if plot_flag:
            cmd.append("--plot")
        if shap_flag:
            cmd.append("--shap")

        print("Running:", " ".join(cmd))

        subprocess.run(cmd, check=True)

    print("runs completed")

if __name__ == "__main__":
    main()