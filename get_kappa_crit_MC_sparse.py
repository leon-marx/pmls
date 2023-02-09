from paper import *


def determine_kappa_manual_MC(num_rep, T, kappa, M0, levels, alpha, HMN_per_rep=False,):
    links = get_HMN(M0, levels, alpha)
    density = np.zeros(T+1)
    print(f"Trying out kappa = {kappa}")

    # Running experiment
    for i in tqdm(range(num_rep)):
        if HMN_per_rep:
            links = get_HMN(M0, levels, alpha)
        density += run_dynamics_MC_sparse(links, T, kappa)
    density /= num_rep

    # Plotting results
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18, 6))
    ax1.set_title(f"kappa = {kappa}")
    ax1.plot(density, label="data")
    ax2.set_title(f"kappa = {kappa} (log-log")
    ax2.loglog(density, label="data")
    plt.show()


if __name__ == "__main__":
    M0 = 4
    levels = 8
    alpha = 2.5

    num_rep = 100
    T = 1000000

    kappa_min = 0.5224609375
    kappa_max = 0.5234375

    # kappa_min: 0.5224609375
    # kappa: 0.52294921875
    # kappa_max: 0.5234375

    # kappa_min: 0.522705078125
    # kappa: 0.52276611328125
    # kappa_max: 0.5228271484375
    kappa_min = 0.5
    kappa_max = 0.7
    while True:
        kappa = 0.5 * (kappa_min + kappa_max)
        determine_kappa_manual_MC(num_rep, T, kappa, M0, levels, alpha)
        flag = input("Next step: go down [d] or up [u] or repeat same [enter] or quit [q]? ")
        if "d" in flag:
            kappa_max = kappa
        elif "u" in flag:
            kappa_min = kappa
        elif "q" in flag:
            print("Quitting instantly...")
            break
        elif "tp" in flag:
            print("Increasing time")
            T *= 10
        elif "tm" in flag:
            print("Decreasing time")
            T /= 10
        else:
            print("Repeating same...")
    print(f"kappa_min: {kappa_min}")
    print(f"kappa: {kappa}")
    print(f"kappa_max: {kappa_max}")
