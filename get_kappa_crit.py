from argparse import ArgumentParser
from paper import *
from animation import DynamicsAnimation
from tqdm import tqdm
from lmfit import minimize, Parameters
import os


def power_law(x, gamma, a):
    return a * x ** -gamma


def residual(params, x, data, uncertainty):
    gamma = params["gamma"]
    a = params["a"]
    model = power_law(x, gamma, a)
    return (data - model) / uncertainty


def determine_kappa_auto(num_rep, T, num_trials, kappa_min, kappa_max, M0, levels, alpha, exp_name):
    links = get_HMN(M0, levels, alpha)
    for k, kappa in enumerate(np.linspace(kappa_min, kappa_max, num_trials)):
        print(f"Trying out kappa = {kappa}")
        # Running experiment
        initials = np.random.randint(0, 2, (num_rep, len(links)), dtype=np.uint8)
        density = np.zeros(T+1)
        for i in tqdm(range(num_rep)):
            # print(".", end="")
            data = run_dynamics(links, initials[i], T, kappa)
            density += np.mean(data, axis=1)
        density /= num_rep
        np.savetxt(f"results/{exp_name}/density_{k}.txt")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="testing")
    parser.add_argument("--M0", type=int, default=4)
    parser.add_argument("--levels", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=2.5)
    parser.add_argument("--num_rep", type=int, default=100)
    parser.add_argument("--T", type=int, default=1024 * 100)
    parser.add_argument("--num_trials", type=int, default=10)
    parser.add_argument("--kappa_min", type=float, default=0.15)
    parser.add_argument("--kappa_max", type=float, default=0.16)
    args = parser.parse_args()

    os.makedirs(f"results/{args.exp_name}", exist_ok=True)
    with open(f"results/{args.exp_name}/metadata.csv", "w") as f:
        f.write("M0;levels;alpha;num_rep;T;num_trials;kappa_min;kappa_max\n")
        f.write(f"{args.M0};{args.levels};{args.alpha};{args.num_rep};{args.T};{args.num_trials};{args.kappa_min};{args.kappa_max}")

    determine_kappa_auto(args.num_rep, args.T, args.num_trials, args.kappa_min, args.kappa_max, args.M0, args.levels, args.alpha, args.exp_name)
