import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as la


# parameters
Nmax = 60
lmax = 2  # lmax=2, Nmax=10 already converged for BdB vs t, not for larger epsilon
mmax = 2
epsilon = 0.05
alpha = 1.0
Delta = epsilon**2 / (3 * alpha)
gamma = 1e-5
time_unit = epsilon**2 / (3 * alpha)  # tau = time_unit * t

liouv_version = 2   # 1: gamma D[B], 2: gamma (D[a1] + D[a2] + D[a3])
use_H2 = False      # effective hamiltonian H2 instead of the full one
init_state = 2      # 1: vacuum, 2: vacuum evolved up to tau=1 at Delta = eps^2/(3 alpha)

# plots
plot_time_evolution = False
plot_diagnostic = False
plot_glueball = False
plot_glueball_multi_gamma = False
plot_glueball_multi_Nmax = False
plot_L2 = True
plot_L2_comparison = True
plot_steady_state = False
plot_steady_state_effective = False
plot_steady_state_L1 = False
plot_steady_state_multi_Nmax = False
show_effective = True

# offsets only in the plots so exact and effective curves don't lie on top of each other
shift_multi_gamma = 0.003
shift_steady_state = 0.08

gammas_glueball_plot = [0.0015, 0.003]
gammas_ss_sweep = np.linspace(1e-5, 3e-3, 100)
Nmax_values = [30, 60, 100]
epsilon_values = np.linspace(0.0, 0.4, 25)

xlabel_tau = r"$\frac{\epsilon^2 t}{3U}$"


# basis |N,l,m> with N-l even
def build_basis(Nmax, lmax, mmax):
    basis = []
    for l in range(lmax + 1):
        for m in range(-min(l, mmax), min(l, mmax) + 1):
            for N in range(l, Nmax + 1, 2):
                basis.append((N, l, m))
    index = {}
    for i in range(len(basis)):
        index[basis[i]] = i
    return np.array(basis), index


# <N+1,l+1,m+q| a_q^dag |N,l,m>
def betaP_q(N, l, m, q):
    if q == 1:
        return -np.sqrt((N + l + 3) * (l + m + 1) * (l + m + 2) / (2 * (2*l + 1) * (2*l + 3)))
    return np.sqrt((N + l + 3) * (l - m + 1) * (l - m + 2) / (2 * (2*l + 1) * (2*l + 3)))


# <N+1,l-1,m+q| a_q^dag |N,l,m>
def betaM_q(N, l, m, q):
    if l == 0:
        return 0.0
    if q == 1:
        return np.sqrt((N - l + 2) * (l - m - 1) * (l - m) / (2 * (2*l - 1) * (2*l + 1)))
    return -np.sqrt((N - l + 2) * (l + m - 1) * (l + m) / (2 * (2*l - 1) * (2*l + 1)))


# <N+1,l+1,m| a_0^dag |N,l,m>
def betaP_0(N, l, m):
    return -np.sqrt((N + l + 3) * (l + m + 1) * (l - m + 1) / ((2*l + 1) * (2*l + 3)))


# <N+1,l-1,m| a_0^dag |N,l,m>
def betaM_0(N, l, m):
    if l == 0:
        return 0.0
    return -np.sqrt((N - l + 2) * (l + m) * (l - m) / ((2*l - 1) * (2*l + 1)))


# operators (built as dense matrices, then made sparse)
def build_aq(q, basis, index):
    D = len(basis)
    adag = np.zeros((D, D), dtype=complex)
    for i, (N, l, m) in enumerate(basis):
        if (N + 1, l + 1, m + q) in index:
            adag[index[(N + 1, l + 1, m + q)], i] = betaP_q(N, l, m, q)
        if (N + 1, l - 1, m + q) in index:
            adag[index[(N + 1, l - 1, m + q)], i] = betaM_q(N, l, m, q)
    return sp.csr_matrix(adag.conj().T)


def build_a3(basis, index):
    D = len(basis)
    a3 = np.zeros((D, D), dtype=complex)
    for i, (N, l, m) in enumerate(basis):
        if (N - 1, l - 1, m) in index:
            a3[index[(N - 1, l - 1, m)], i] = betaP_0(N - 1, l - 1, m)
        if (N - 1, l + 1, m) in index:
            a3[index[(N - 1, l + 1, m)], i] = betaM_0(N - 1, l + 1, m)
    return sp.csr_matrix(a3)


# B|N,l,m> = sqrt((N(N+1) - l(l+1))/6) |N-2,l,m>
def build_B(basis, index):
    D = len(basis)
    B = np.zeros((D, D), dtype=complex)
    for i, (N, l, m) in enumerate(basis):
        val = (N * (N + 1) - l * (l + 1)) / 6
        if (N - 2, l, m) in index and val > 0:
            B[index[(N - 2, l, m)], i] = np.sqrt(val)
    return sp.csr_matrix(B)


def build_operators(Nmax, lmax, mmax):
    basis, index = build_basis(Nmax, lmax, mmax)
    a_plus = build_aq(1, basis, index)
    a_minus = build_aq(-1, basis, index)
    ops = {
        "basis": basis,
        "index": index,
        "a1": (a_minus - a_plus) / np.sqrt(2),
        "a2": 1j * (a_minus + a_plus) / np.sqrt(2),
        "a3": build_a3(basis, index),
        "B": build_B(basis, index),
    }
    return ops


# diagonal operators
def N_op(basis):
    return np.diag(basis[:, 0])


def l_op(basis):
    return np.diag(basis[:, 1])


def K_op(basis):
    # glueball number k = (N-l)/2
    return np.diag((basis[:, 0] - basis[:, 1]) // 2)


def L2_op(basis):
    return np.diag(basis[:, 1] * (basis[:, 1] + 1))


def BdB_op(basis):
    return np.diag((basis[:, 0] * (basis[:, 0] + 1) - basis[:, 1] * (basis[:, 1] + 1)) / 6)


# hamiltonians
def H_full(eps, Delta, alpha, ops):
    # H = Delta N + alpha l(l+1) + eps (a3 + a3^dag)
    N = ops["basis"][:, 0]
    l = ops["basis"][:, 1]
    a3 = ops["a3"]
    H0 = sp.diags(Delta * N + alpha * l * (l + 1), format="csr")
    return (H0 + eps * (a3 + a3.getH())).tocsr()


def H_2(eps, Delta, alpha, ops):
    # H2 = (Delta - eps^2/(3 alpha)) N - eps^2/(sqrt(6) alpha) (B + B^dag)
    N = ops["basis"][:, 0]
    B = ops["B"]
    delta_eff = Delta - eps**2 / (3 * alpha)
    C = eps**2 / (np.sqrt(6) * alpha)
    return (sp.diags(delta_eff * N, format="csr") - C * (B + B.getH())).tocsr()


def get_H(eps, Delta, alpha, ops):
    if use_H2:
        return H_2(eps, Delta, alpha, ops)
    return H_full(eps, Delta, alpha, ops)


# liouvillian, rho is stacked column by column
def dissipator(c):
    I = sp.identity(c.shape[0], format="csr")
    cdc = c.getH() @ c
    return sp.kron(c.conj(), c) - 0.5 * sp.kron(I, cdc) - 0.5 * sp.kron(cdc.T, I)


def liouvillian(H, a1, a2, a3, B, gamma, version=liouv_version):
    I = sp.identity(H.shape[0], format="csr")
    L = -1j * (sp.kron(I, H) - sp.kron(H.T, I))
    if version == 1:
        L = L + gamma * dissipator(B)
    else:
        L = L + gamma * (dissipator(a1) + dissipator(a2) + dissipator(a3))
    return L.tocsr()


def liouvillian_ops(H, ops, gamma, version=liouv_version):
    return liouvillian(H, ops["a1"], ops["a2"], ops["a3"], ops["B"], gamma, version)


# initial states and time evolution
def vacuum(ops):
    D = len(ops["basis"])
    i0 = ops["index"][(0, 0, 0)]
    rho = np.zeros((D, D), dtype=complex)
    rho[i0, i0] = 1
    return rho.flatten(order="F")


def prepared_state(eps, alpha, ops, gamma):
    # vacuum evolved at Delta = eps^2/(3 alpha) until tau = 1
    H = H_full(eps, eps**2 / (3 * alpha), alpha, ops)
    L = liouvillian_ops(H, ops, gamma)
    t_prep = 3 * alpha / eps**2
    return spla.expm_multiply(L * t_prep, vacuum(ops))


def initial_state(eps, alpha, ops, gamma):
    if init_state == 1:
        return vacuum(ops)
    return prepared_state(eps, alpha, ops, gamma)


def evolve(L, rho0, t):
    # t has to be equally spaced (linspace)
    return spla.expm_multiply(L, rho0, start=t[0], stop=t[-1], num=len(t), endpoint=True)


def trajectory(tau, eps, Delta, alpha, ops, rho0, gamma):
    H = get_H(eps, Delta, alpha, ops)
    L = liouvillian_ops(H, ops, gamma)
    return evolve(L, rho0, tau / time_unit)


def expval(rho_list, O):
    # <O> = tr(rho O) for every rho in the list
    if sp.issparse(O):
        O = O.toarray()
    D = O.shape[0]
    values = []
    for rho_vec in rho_list:
        rho = rho_vec.reshape((D, D), order="F")
        values.append(np.trace(rho @ O))
    return np.array(values)


def steady_state(L, D):
    # solve L rho = 0, first row replaced by tr(rho) = 1
    A = L.tolil()
    A[0, :] = 0
    for i in range(D):
        A[0, i * (D + 1)] = 1
    b = np.zeros(D * D, dtype=complex)
    b[0] = 1
    rho = spla.spsolve(A.tocsr(), b)
    if not np.all(np.isfinite(rho)):
        raise np.linalg.LinAlgError("steady state has nan/inf")
    return rho


# first order schrieffer-wolff, done numerically in the truncated basis
def sw_transform(S, O):
    if sp.issparse(O):
        O = O.toarray()
    return sp.csr_matrix(O + S @ O - O @ S)


def sw_effective(eps, Delta, alpha, ops, rho0):
    basis = ops["basis"]
    D = len(basis)
    N = basis[:, 0]
    l = basis[:, 1]
    E = Delta * N + alpha * l * (l + 1)
    V = (eps * (ops["a3"] + ops["a3"].getH())).toarray()

    S = np.zeros((D, D), dtype=complex)
    for i in range(D):
        for j in range(D):
            if i != j and abs(E[i] - E[j]) > 1e-12:
                S[i, j] = V[i, j] / (E[i] - E[j])
    S = 0.5 * (S - S.conj().T)

    H_eff = np.diag(E) + 0.5 * (S @ V - V @ S)
    H_eff = 0.5 * (H_eff + H_eff.conj().T)

    rho = rho0.reshape((D, D), order="F")
    rho = la.expm(S) @ rho @ la.expm(-S)
    rho = 0.5 * (rho + rho.conj().T)
    rho = rho / np.trace(rho)

    sw = {
        "H": sp.csr_matrix(H_eff),
        "a1": sw_transform(S, ops["a1"]),
        "a2": sw_transform(S, ops["a2"]),
        "a3": sw_transform(S, ops["a3"]),
        "B": sw_transform(S, ops["B"]),
        "K": sw_transform(S, K_op(basis)),
        "rho0": rho.flatten(order="F"),
    }
    return sw


def glueball_effective(tau, eps, Delta, alpha, ops, rho0, gamma):
    sw = sw_effective(eps, Delta, alpha, ops, rho0)
    L = liouvillian(sw["H"], sw["a1"], sw["a2"], sw["a3"], sw["B"], gamma)
    rho_t = evolve(L, sw["rho0"], tau / time_unit)
    return expval(rho_t, sw["K"])


# steady states
def steady_state_vs_gamma(gammas, eps, Delta, alpha, ops, rho0, tau, effective=False, version=liouv_version):
    D = len(ops["basis"])
    if effective:
        sw = sw_effective(eps, Delta, alpha, ops, rho0)

    results = []
    for g in gammas:
        print(f"steady state: gamma={g:.3g}, effective={effective}, L{version}")
        if effective:
            L = liouvillian(sw["H"], sw["a1"], sw["a2"], sw["a3"], sw["B"], g, version)
            K = sw["K"]
            start = sw["rho0"]
        else:
            L = liouvillian_ops(get_H(eps, Delta, alpha, ops), ops, g, version)
            K = K_op(ops["basis"])
            start = rho0

        try:
            rho = steady_state(L, D)
        except np.linalg.LinAlgError:
            print("  solver failed, using final time of the evolution instead")
            rho = evolve(L, start, tau / time_unit)[-1]
        results.append(expval([rho], K)[0].real)
    return np.array(results)


def steady_state_BdB_vs_eps(eps_values, Nmax_values, Delta, alpha, gamma, tau):
    results = {}
    for n in Nmax_values:
        print(f"building operators for Nmax={n}")
        ops = build_operators(n, lmax, mmax)
        D = len(ops["basis"])
        BdB = BdB_op(ops["basis"])
        rho_vac = vacuum(ops)

        values = []
        for eps in eps_values:
            print(f"  eps={eps:.3f}")
            if eps == 0:
                values.append(expval([rho_vac], BdB)[0].real)
                continue
            L = liouvillian_ops(get_H(eps, Delta, alpha, ops), ops, gamma)
            try:
                rho = steady_state(L, D)
            except np.linalg.LinAlgError:
                print("  solver failed, using final time of the evolution instead")
                rho = spla.expm_multiply(L * (tau[-1] / time_unit), rho_vac)
            values.append(expval([rho], BdB)[0].real)
        results[n] = np.array(values)
    return results


# plotting
def save(filename):
    os.makedirs("BA_finalgraphs", exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join("BA_finalgraphs", filename))
    print("saved", filename)
    plt.show()


def style(ax, fontsize=12, nx=4, ny=3, width=1.4):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
    ax.tick_params(direction="out", length=6, width=width, pad=8, labelsize=fontsize, top=False, right=False)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=nx))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=ny))
    ax.grid(False)


def gamma_label(g):
    e = int(np.floor(np.log10(g)))
    m = g / 10**e
    if np.isclose(m, 1):
        return rf"10^{{{e}}}"
    return rf"{m:.2g}\cdot 10^{{{e}}}"


def plot_BdB(tau, curves):
    plt.figure(figsize=(8, 5))
    for D_val, values in curves.items():
        plt.plot(tau, values.real, label=rf"$\epsilon={epsilon}, \Delta={D_val:.3g}, \alpha={alpha}, \gamma={gamma}$")
    plt.plot(tau, 1.5 * np.sinh(tau)**2, color="orange", linestyle="--", label=r"$\frac{3}{2}\sinh^2(\tau)$")
    plt.xlabel(r"dimensionless time $\tau = \frac{\epsilon^2}{\sqrt{6}\alpha} t = C\cdot t$")
    plt.ylabel(r"$\langle B^\dagger B \rangle$")
    plt.legend()
    save(f"plot_expectation_BdB_multi_eps{epsilon}_alpha{alpha}.pdf")


def plot_diagnostic_N_l_K(tau, N_vals, l_vals, K_vals):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(tau, N_vals.real, color="tab:blue", linewidth=2, label=r"$\langle N \rangle$")
    ax.plot(tau, l_vals.real, color="tab:orange", linewidth=2, label=r"$\langle l \rangle$")
    ax.plot(tau, K_vals.real, color="tab:green", linewidth=2,
            label=r"$\langle K \rangle = \frac{1}{2}(\langle N \rangle - \langle l \rangle)$")
    ax.set_xlabel(r"dimensionless time $\tau = \frac{\epsilon^2}{3\alpha} t$")
    ax.set_ylabel("expectation values")
    ax.set_title(rf"diagnostic: $\epsilon={epsilon}, \Delta={Delta:.3g}, \alpha={alpha}, \gamma={gamma}$")
    ax.legend(fontsize=12)
    save(f"plot_diagnostic_N_l_K_eps{epsilon}_Delta{Delta:.3g}_alpha{alpha}.pdf")


def plot_glueball(tau, exact, effective=None):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(tau, exact.real, color="tab:blue", label="exact")
    if effective is not None:
        ax.plot(tau, effective.real, color="tab:green", label="effective")
    ax.legend(loc="lower right", fontsize=12)
    ax.set_xlabel(xlabel_tau, fontsize=12)
    ax.set_ylabel(r"$\langle \mathcal{N} \rangle$", fontsize=12)
    style(ax)
    save(f"plot_glueball_population_multi_eps{epsilon}_alpha{alpha}_Bgamma{gamma}.pdf")


def plot_glueball_multi_gamma_curves(tau, exact, effective):
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = [("tab:blue", "tab:green"), ("tab:orange", "skyblue")]
    for i, g in enumerate(sorted(exact.keys())):
        c_exact, c_eff = colors[i % 2]
        label = gamma_label(g)
        ax.plot(tau, exact[g].real, color=c_exact, linewidth=2, label=rf"exact, $\gamma_{{1,2,3}}={label}$")
        if effective.get(g) is not None:
            ax.plot(tau, effective[g].real + shift_multi_gamma, color=c_eff, linestyle="--", linewidth=1.8,
                    alpha=0.8, label=rf"eff., $\gamma_{{1,2,3}}={label}$")
    ax.legend(loc="lower right", fontsize=12)
    ax.set_xlabel(xlabel_tau, fontsize=12)
    ax.set_ylabel(r"$\langle \mathcal{N} \rangle$", fontsize=12)
    style(ax)
    save(f"plot_glueball_population_multi_gamma_eps{epsilon}_alpha{alpha}.pdf")


def plot_glueball_multi_Nmax_curves(tau, curves):
    fig, ax = plt.subplots(figsize=(6, 5))
    for n, values in curves.items():
        ax.plot(tau, values.real, linewidth=2, label=rf"$N_{{\mathrm{{max}}}}={n}$")
    ax.plot(tau, 1.5 * np.sinh(tau)**2, color="orange", linestyle="--", label="analytic")
    ax.legend(loc="upper left", fontsize=14)
    ax.set_xlabel(xlabel_tau, fontsize=17)
    ax.set_ylabel(r"$\langle \mathcal{N} \rangle$", fontsize=17)
    style(ax, fontsize=17, width=1.2)
    save(f"plot_glueball_population_multi_Nmax_eps{epsilon}_alpha{alpha}.pdf")


def plot_L2_curves(tau, curves):
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, values in curves.items():
        ax.plot(tau, values.real, linewidth=2, label=label)
    ax.set_xlabel(xlabel_tau, fontsize=12)
    ax.set_ylabel(r"$\langle L^2 \rangle$", fontsize=12)
    ax.legend()
    style(ax, nx=3, ny=4, width=1.2)
    save(f"plot_expectation_L2_eps{epsilon}_Delta{Delta:.3g}_alpha{alpha}_comparison.pdf")


def plot_steady_state(gammas, exact, effective=None, exact_L1=None, effective_L1=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    g = 1e3 * np.asarray(gammas)
    ax.plot(g, exact, color="tab:blue", linewidth=2, marker="o", label="exact")
    if effective is not None:
        ax.plot(g, effective + shift_steady_state, color="tab:green", linewidth=1.8, marker="o",
                alpha=0.9, label="effective")
    if exact_L1 is not None:
        ax.plot(g, exact_L1, color="tab:orange", linewidth=2, marker="o", label=r"ex., $B$ diss.")
    if effective_L1 is not None:
        ax.plot(g, effective_L1 + shift_steady_state, color="skyblue", linestyle="--", linewidth=1.8,
                marker="o", alpha=0.9, label=r"eff., $B$ diss.")
    ax.legend(fontsize=17)
    ax.set_xlabel(r"$\gamma_{1,2,3}\,(10^{-3})$", fontsize=17)
    ax.set_ylabel(r"$\langle \mathcal{N} \rangle_{\mathrm{ss}}$", fontsize=17)
    ax.set_xlim(left=0)
    style(ax, fontsize=17, nx=5)
    save(f"plot_steady_state_glueballN_vs_gamma_alpha{alpha}.pdf")


def plot_BdB_vs_eps(eps_values, results):
    plt.figure(figsize=(8, 5))
    for n, values in results.items():
        plt.plot(eps_values, values, "o-",
                 label=rf"$N_{{\mathrm{{max}}}}={n}, \Delta={Delta:.3g}, \alpha={alpha}, \gamma={gamma}$")
    plt.xlabel(r"$\epsilon$")
    plt.ylabel(r"Steady state $\langle B^\dagger B \rangle$")
    plt.legend()
    save(f"plot_steady_state_BdB_multi_Nmax_Delta{Delta:.3g}_alpha{alpha}.pdf")


# main
if __name__ == "__main__":
    tau = np.linspace(0, 2, 200)

    ops = build_operators(Nmax, lmax, mmax)
    basis = ops["basis"]
    print(f"Nmax={Nmax}, lmax={lmax}, mmax={mmax}, D={len(basis)}")

    rho_vac = vacuum(ops)
    rho_prep = prepared_state(epsilon, alpha, ops, gamma)
    if init_state == 1:
        rho0 = rho_vac
    else:
        rho0 = rho_prep

    if plot_time_evolution:
        rho_t = trajectory(tau, epsilon, Delta, alpha, ops, rho0, gamma)
        plot_BdB(tau, {Delta: expval(rho_t, BdB_op(basis))})

    if plot_diagnostic:
        rho_t = trajectory(tau, epsilon, Delta, alpha, ops, rho0, gamma)
        N_vals = expval(rho_t, N_op(basis))
        l_vals = expval(rho_t, l_op(basis))
        K_vals = 0.5 * (N_vals - l_vals)
        plot_diagnostic_N_l_K(tau, N_vals, l_vals, K_vals)

    if plot_glueball:
        rho_t = trajectory(tau, epsilon, Delta, alpha, ops, rho0, gamma)
        exact = expval(rho_t, K_op(basis))
        effective = None
        if show_effective:
            effective = glueball_effective(tau, epsilon, Delta, alpha, ops, rho0, gamma)
        plot_glueball(tau, exact, effective)

    if plot_glueball_multi_gamma:
        exact = {}
        effective = {}
        for g in gammas_glueball_plot:
            rho_t = trajectory(tau, epsilon, Delta, alpha, ops, rho0, g)
            exact[g] = expval(rho_t, K_op(basis))
            effective[g] = None
            if show_effective:
                effective[g] = glueball_effective(tau, epsilon, Delta, alpha, ops, rho0, g)
        plot_glueball_multi_gamma_curves(tau, exact, effective)

    if plot_glueball_multi_Nmax:
        curves = {}
        for n in Nmax_values:
            print(f"building operators for Nmax={n}")
            ops_n = build_operators(n, lmax, mmax)
            start = initial_state(epsilon, alpha, ops_n, gamma)
            rho_t = trajectory(tau, epsilon, Delta, alpha, ops_n, start, gamma)
            curves[n] = expval(rho_t, K_op(ops_n["basis"]))
        plot_glueball_multi_Nmax_curves(tau, curves)

    if plot_L2:
        L2 = L2_op(basis)
        if plot_L2_comparison:
            curves = {
                r"$\rho(t=0)$ = vac": expval(trajectory(tau, epsilon, Delta, alpha, ops, rho_vac, gamma), L2),
                r"$\rho(t=0)$ = prep gb condensate": expval(trajectory(tau, epsilon, Delta, alpha, ops, rho_prep, gamma), L2),
            }
        else:
            curves = {r"$L^2$": expval(trajectory(tau, epsilon, Delta, alpha, ops, rho0, gamma), L2)}
        plot_L2_curves(tau, curves)

    if plot_steady_state:
        exact = steady_state_vs_gamma(gammas_ss_sweep, epsilon, Delta, alpha, ops, rho0, tau, False, 2)
        effective = None
        exact_L1 = None
        effective_L1 = None
        if plot_steady_state_effective:
            effective = steady_state_vs_gamma(gammas_ss_sweep, epsilon, Delta, alpha, ops, rho0, tau, True, 2)
        if plot_steady_state_L1:
            exact_L1 = steady_state_vs_gamma(gammas_ss_sweep, epsilon, Delta, alpha, ops, rho0, tau, False, 1)
            if plot_steady_state_effective:
                effective_L1 = steady_state_vs_gamma(gammas_ss_sweep, epsilon, Delta, alpha, ops, rho0, tau, True, 1)
        plot_steady_state(gammas_ss_sweep, exact, effective, exact_L1, effective_L1)

    if plot_steady_state_multi_Nmax:
        results = steady_state_BdB_vs_eps(epsilon_values, Nmax_values, Delta, alpha, gamma, tau)
        plot_BdB_vs_eps(epsilon_values, results)