import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap


# parameters
Kmax = 20  # number of glueball states |K>
epsilon = 0.05
alpha = 1.0
Delta = 0.0
gamma = 0.1
C = epsilon**2 / (alpha * np.sqrt(6))
delta_eff = Delta - epsilon**2 / (3 * alpha)  # zero for Delta = eps^2/(3 alpha)

n_times = 300
tau_max = 600.0
selected_tau = float(os.environ.get("SELECTED_TAU", "600.0"))  # this time gets compared with t=0

# plots
plot_cloud = False
plot_time_evolution = False
plot_wigner = True

# wigner grid
x_min, x_max = -5.0, 5.0
p_min, p_max = -5.0, 5.0
n_grid = int(os.environ.get("WIGNER_GRID_POINTS", "41"))

colors_wigner = LinearSegmentedColormap.from_list("wigner", ["#f0f0f0", "#aaffaa", "#0000ff"])


def save(filename):
    os.makedirs("BA_finalgraphs", exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join("BA_finalgraphs", filename))
    print("saved", filename)
    plt.show()


# operators, everything dense since Kmax small
def build_operators(Kmax):
    N = np.diag(2 * np.arange(Kmax))  # N|K = 2K|K
    B = np.zeros((Kmax, Kmax))
    for k in range(1, Kmax):
        B[k - 1, k] = np.sqrt(2 * k * (2 * k + 1) / 6)
    return N, B


def hamiltonian(N, B):
    return delta_eff * N - C * (B + B.T)


def liouvillian(H, B, gamma):
    # rho stacked column after column
    I = np.eye(len(H))
    BdB = B.conj().T @ B
    L = -1j * (np.kron(I, H) - np.kron(H.T, I))
    L = L + gamma * (np.kron(B.conj(), B) - 0.5 * np.kron(I, BdB) - 0.5 * np.kron(BdB.T, I))
    return L


def to_matrix(rho_vec):
    D = int(np.sqrt(len(rho_vec)))
    return rho_vec.reshape((D, D), order="F")


def expval(rho_vec, O):
    return np.trace(to_matrix(rho_vec) @ O)


# quadratures
def quadratures(B):
    Bd = B.conj().T
    X = (B + Bd) / np.sqrt(2)
    P = -1j * (B - Bd) / np.sqrt(2)
    return X, P


def moments(rho_vec, B):
    X, P = quadratures(B)
    ev_X = expval(rho_vec, X)
    ev_P = expval(rho_vec, P)
    return {
        "B": expval(rho_vec, B),
        "Bd": expval(rho_vec, B.conj().T),
        "X": ev_X,
        "P": ev_P,
        "var_X": np.real(expval(rho_vec, X @ X) - ev_X**2),
        "var_P": np.real(expval(rho_vec, P @ P) - ev_P**2),
        "cov_XP": np.real(0.5 * expval(rho_vec, X @ P + P @ X) - ev_X * ev_P),
    }


def print_moments(label, m):
    print(f"\nquadratures at {label}")
    for key in m:
        print(f"  {key:7s} = {np.real_if_close(m[key])}")


def covariance_matrix(m):
    cov = np.array([[m["var_X"], m["cov_XP"]], [m["cov_XP"], m["var_P"]]], dtype=float)
    # remove small negative eigenvalues 
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 0, None)
    return vecs @ np.diag(vals) @ vecs.T


def ellipse(m, n_points=500):
    # one sigma ellipse around the mean
    angles = np.linspace(0, 2 * np.pi, n_points)
    vals, vecs = np.linalg.eigh(covariance_matrix(m))
    points = vecs @ np.diag(np.sqrt(np.clip(vals, 0, None))) @ np.array([np.cos(angles), np.sin(angles)])
    return points[0] + np.real(m["X"]), points[1] + np.real(m["P"])


# wigner function W = tr(D^dag rho D Pi)/pi
def wigner(rho, a, x_values, p_values):
    ad = a.conj().T
    parity = la.expm(1j * np.pi * ad @ a)
    W = np.zeros((len(p_values), len(x_values)))
    for i in range(len(p_values)):
        for j in range(len(x_values)):
            beta = (x_values[j] + 1j * p_values[i]) / np.sqrt(2)
            D = la.expm(beta * ad - np.conj(beta) * a)
            W[i, j] = np.real(np.trace(D.conj().T @ rho @ D @ parity)) / np.pi
    return W


# plots
def plot_cloud_at(m, title, n_points=1500):
    # gaussian cloud from first and second moments
    rng = np.random.default_rng(7)
    mean = [np.real(m["X"]), np.real(m["P"])]
    cloud = rng.multivariate_normal(mean, covariance_matrix(m), size=n_points)

    plt.figure(figsize=(6.5, 6.0))
    plt.scatter(cloud[:, 0], cloud[:, 1], s=9, alpha=0.25, label="moment cloud")
    plt.scatter([mean[0]], [mean[1]], color="crimson", s=70, zorder=5, label=r"mean $(\langle X\rangle,\langle P\rangle)$")
    plt.axhline(0, color="0.82", linewidth=1)
    plt.axvline(0, color="0.82", linewidth=1)
    plt.xlabel(r"$X = (B+B^\dagger)/\sqrt{2}$")
    plt.ylabel(r"$P = -i(B-B^\dagger)/\sqrt{2}$")
    plt.title(title)
    plt.axis("equal")
    plt.grid(alpha=0.25)
    plt.legend()
    save(f"plot_quadrature_cloud_{title.replace(' ', '_')}.pdf")


def plot_wigner(rho, a, time_label, filename):
    x_values = np.linspace(x_min, x_max, n_grid)
    p_values = np.linspace(p_min, p_max, n_grid)
    W = wigner(rho, a, x_values, p_values)
    Xg, Pg = np.meshgrid(x_values, p_values)

    vlim = np.max(np.abs(W))
    if vlim == 0:
        vlim = 1.0
    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim)
    levels = np.linspace(-vlim, vlim, 41)
    levels3d = np.linspace(-vlim, vlim, 21)

    fig = plt.figure(figsize=(13.0, 5.8))

    # heatmap
    ax1 = fig.add_subplot(1, 2, 1)
    image = ax1.imshow(W, extent=[x_min, x_max, p_min, p_max], origin="lower", cmap=colors_wigner, norm=norm)
    ax1.contour(x_values, p_values, W, levels=levels, colors="black", linewidths=0.4, alpha=0.55)
    ax1.axhline(0, color="0.25", linewidth=0.8, alpha=0.7)
    ax1.axvline(0, color="0.25", linewidth=0.8, alpha=0.7)
    ax1.set_xlabel(r"$X_B = (B+B^\dagger)/\sqrt{2}$")
    ax1.set_ylabel(r"$P_B = -i(B-B^\dagger)/\sqrt{2}$")
    ax1.set_title(rf"Wigner function for B at {time_label}")
    fig.colorbar(image, ax=ax1, ticks=levels[::5], label=r"$W(X,P)$")

    # surface
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.plot_surface(Xg, Pg, W, rstride=1, cstride=1, facecolors=colors_wigner(norm(W)),
                     linewidth=0, antialiased=False, shade=False)
    step = max(1, n_grid // 12)
    ax2.plot_wireframe(Xg, Pg, W, rstride=step, cstride=step, color="black", linewidth=0.4, alpha=0.35)
    ax2.set_xlabel(r"$X$")
    ax2.set_ylabel(r"$P$")
    ax2.set_zlabel(r"$W(X,P)$")
    ax2.set_title("Wigner surface")
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=colors_wigner), ax=ax2, shrink=0.6, aspect=12,
                 boundaries=levels3d, ticks=levels3d[::3])
    save(filename)


def plot_quadrature_evolution(tau, series, n_ellipses=6):
    X = np.real([m["X"] for m in series])
    P = np.real([m["P"] for m in series])

    plt.figure(figsize=(7.0, 6.0))
    points = plt.scatter(X, P, c=tau, s=18, cmap="viridis", label=r"trajectory $(\langle X\rangle,\langle P\rangle)$")
    plt.plot(X, P, color="0.35", linewidth=1, alpha=0.65)
    plt.scatter([X[0]], [P[0]], color="black", s=60, label=r"$t=0$")
    plt.scatter([X[-1]], [P[-1]], color="crimson", s=60, label="final")

    for i in np.linspace(0, len(tau) - 1, n_ellipses, dtype=int):
        ex, ep = ellipse(series[i])
        plt.plot(ex, ep, color="black", linewidth=0.9, alpha=0.45)

    plt.axhline(0, color="0.82", linewidth=1)
    plt.axvline(0, color="0.82", linewidth=1)
    plt.xlabel(r"$\langle X\rangle$")
    plt.ylabel(r"$\langle P\rangle$")
    plt.title(r"Quadrature time evolution in the $X$-$P$ plane")
    plt.axis("equal")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.colorbar(points, label=r"dimensionless time $\tau = Ct$")
    save("plot_quadrature_time_evolution.pdf")


# main
if __name__ == "__main__":
    N, B = build_operators(Kmax)
    H = hamiltonian(N, B)
    L = liouvillian(H, B, gamma)

    # start in |0><0|
    rho0 = np.zeros((Kmax, Kmax), dtype=complex)
    rho0[0, 0] = 1
    rho0_vec = rho0.flatten(order="F")
    print("tr(rho) =", np.trace(rho0))
    print("tr(rho^2) =", np.trace(rho0 @ rho0))

    tau = np.linspace(0, tau_max, n_times)
    t = tau / C
    # one propagator for a single time step, then just multiply (t is equally spaced)
    U = la.expm(L * (t[1] - t[0]))
    rho_t = [rho0_vec]
    for i in range(n_times - 1):
        rho_t.append(U @ rho_t[-1])
    series = [moments(r, B) for r in rho_t]
    print_moments("t = 0", series[0])

    # time step closest to selected_tau (but not t=0)
    i_sel = int(np.argmin(np.abs(tau - selected_tau)))
    if i_sel == 0:
        i_sel = 1
    tau_sel = tau[i_sel]

    if plot_cloud:
        plot_cloud_at(series[0], r"Initial quadrature cloud at $t=0$")
        plot_cloud_at(series[i_sel], rf"Quadrature cloud at $\tau = {tau_sel:.1f}$")

    if plot_wigner:
        print(f"wigner plots at t = 0 and tau = {tau_sel:.3g}")
        plot_wigner(rho0, B, r"$t=0$", "plot_wigner_mode_t0.pdf")
        plot_wigner(to_matrix(rho_t[i_sel]), B, rf"$\tau={tau_sel:.3g}$", f"plot_wigner_mode_tau{tau_sel:.3g}.pdf")

    if plot_time_evolution:
        print_moments("final time", series[-1])
        print("\nsmallest variances in the time window:")
        print("  min Var(X) =", min(m["var_X"] for m in series))
        print("  min Var(P) =", min(m["var_P"] for m in series))
        plot_quadrature_evolution(tau, series)