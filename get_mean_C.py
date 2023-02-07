from argparse import ArgumentParser
from paper import *
import os


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="testing")
    parser.add_argument("--hmn_path", type=str, default="data/links.npy")
    parser.add_argument("--num_rep", type=int, default=1000)
    parser.add_argument("--I_start", type=int, default=5000)
    parser.add_argument("--kappa", type=float, default=0.3)
    args = parser.parse_args()

    os.makedirs(f"results/{args.exp_name}", exist_ok=True)
    with open(f"results/{args.exp_name}/metadata.csv", "w") as f:
        f.write("hmn_path;num_rep;I_start;kappa\n")
        f.write(f"{args.hmn_path};{args.num_rep};{args.I_start};{args.kappa}")

    links = np.load(args.hmn_path)

    initials = np.random.randint(0, 2, size=(args.num_rep, len(links)), dtype=np.uint8)
    a_vals = np.ones(args.num_rep, dtype=np.uint64) * args.I_start
    b_vals = np.ones(args.num_rep, dtype=np.uint64) * 11 * args.I_start
    intervals = np.stack([a_vals, b_vals], axis=1)
    delta = 1
    C = get_mean_C(links, initials, args.num_rep, intervals, delta, args.kappa)

    np.savetxt(f"results/{args.exp_name}/C.txt", C)
