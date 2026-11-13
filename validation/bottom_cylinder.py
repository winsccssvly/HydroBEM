"""A fixed bottom-mounted vertical cylinder (radius a = 1 m, depth h = 2a) in head seas: the surge
exciting force against the closed form of MacCamy & Fuchs (1954) and the mean surge drift force
against the pressure integration of the same analytic solution (Havelock 1940 in deep water),
both from the scattering series derived below; HydroBEM's pressure-integration and
momentum-conservation drifts in the low order on four meshes and in the higher order
(docs/validation/cases/bottom_mounted_cylinder.md).

The total first-order potential of a wave of amplitude A (time factor e^{iωt}) about the cylinder,
k tanh kh = ν = ω²/g, Z(z) = cosh k(z + h)/cosh kh, ε_0 = 1, ε_m = 2:

    φ = (gA/ω) Z(z) Σ_m ε_m (−i)^m [J_m(kr) − J_m'(ka) H_m(kr)/H_m'(ka)] cos mθ   (H = H⁽²⁾),

on r = a, by the Wronskian, φ = (gA/ω) Z(z) S(θ), S = Σ ε_m (−i)^m (−2i)/(π ka H_m'(ka)) cos mθ.
The surge force −∫ p n_x dS with p = −iωρφ gives |X₁| = 4ρgA tanh(kh)/(k²|H₁'(ka)|) (MacCamy &
Fuchs 1954). The mean force of the fixed body (n out of the body) is

    F̄_x = (ρ/4) ∫∫ |∇φ|² n_x dS − (ρg/4) ∮ |ζ|² n_x dl,  ζ = −(iω/g) φ(z = 0),

with |∇φ|² = (gA/ω)² [Z² |S'|²/a² + Z_z² |S|²] on the wall and the depth integrals in closed form,
∫Z² dz = (h/2 + sinh 2kh/(4k))/cosh² kh, ∫Z_z² dz = k²(sinh 2kh/(4k) − h/2)/cosh² kh. In deep
water this is Havelock's (1940) eq. (36); its closed sum, eq. (39), checks the series here.

    python validation/bottom_cylinder.py [--quick]
"""

from __future__ import annotations

import math
import sys

import matplotlib.ticker
import mesh_figure
import numpy as np
from common import REF, RHO, C, G, Timer, e, pc, plt, record, save, size, solve, table, threads
from scipy import special as sp

from bodies import surface
from meshes import cylinder_panels
from problems import analytic as analytic_shape
from problems import model

H_A = 2.0


def wavenumber(nu, h):
    k = max(nu, math.sqrt(nu / h))
    for _ in range(100):
        t = math.tanh(k * h)
        k -= (k * t - nu) / (t + k * h * (1 - t * t))
    return k


def h2p(m, x):
    """H_m^(2)'(x)."""
    return sp.jvp(m, x) - 1j * sp.yvp(m, x)


def analytic(ka, h, a=1.0, mmax=None):
    """|X1|/(rho g A a^2) and the mean drift F_x/(rho g A^2 a) of the bottom-mounted cylinder."""
    k = ka / a
    nu = k * math.tanh(k * h)
    mmax = mmax or int(ka + 40)
    th = np.linspace(0.0, 2 * np.pi, 4097)[:-1]
    S = np.zeros_like(th, dtype=complex)
    dS = np.zeros_like(th, dtype=complex)
    for m in range(mmax + 1):
        e = 1.0 if m == 0 else 2.0
        c = e * (-1j) ** m * (-2j) / (math.pi * ka * h2p(m, ka))
        S += c * np.cos(m * th)
        dS += -m * c * np.sin(m * th)
    x1 = 4.0 * math.tanh(k * h) / (k * k * abs(h2p(1, ka))) / a**2
    ch2 = math.cosh(k * h) ** 2
    ic = (h / 2 + math.sinh(2 * k * h) / (4 * k)) / ch2
    is_ = k * k * (math.sinh(2 * k * h) / (4 * k) - h / 2) / ch2
    mean = lambda f: f.mean() * 2 * np.pi  # noqa: E731  (the trapezoidal rule of a periodic function)
    cos = np.cos(th)
    wall = (a / 4) / nu * mean((ic * abs(dS) ** 2 / a**2 + is_ * abs(S) ** 2) * cos)  # (gA/w)^2 = g A^2 / nu
    line = -(a / 4) * mean(abs(S) ** 2 * cos)
    return x1, (wall + line) / a


def havelock39(ka):
    """Havelock (1940), eq. (39): the deep-water drift of a vertical cylinder of infinite draft over
    rho g A^2 a."""
    n = np.arange(0, int(ka + 25))
    hp = np.abs([h2p(m, ka) for m in range(len(n) + 1)]) ** 2
    return 4.0 / (math.pi**2 * ka**3) * np.sum((1 - n * (n + 1) / ka**2) ** 2 / (hp[:-1] * hp[1:]))


def body_of(n):
    """Low order (n > 0): n panels a quarter circle and n down the side (cosine-spaced), the graded
    interior surface, which removes the irregular frequencies (the first at ka = j_{0,1} = 2.405,
    J_0(ka) = 0). Higher order (n = 0): the analytic cylinder with the draft equal to the depth, the
    wall 8 x 8 spans a quadrant and the waterplane 8 x 2 as the interior surface, order 3, the
    uniform mapping (the pressure drift is not available with the nonuniform one, whose profiles
    meet the waterline)."""
    if n == 0:
        rows = [(8, 8, 3, 3, 3, 4), (8, 2, 3, 3, 3, 4)]
        mesh = analytic_shape("circular_cylinder", rows, parts=["side", "waterplane"],
                        mapping="uniform", radius=1.0, draft=H_A, symmetry="xz_and_yz")  # fmt: skip
        return {"mesh": (mesh, (1, 1)), "modes": [], "interior_surface": (mesh, (2, 2))}
    pan = cylinder_panels(1.0, H_A, n_phi=n, n_z=n, n_r=0, quarter=True, cosine=True)
    return {"mesh": surface(pan, symmetry="xz_and_yz"), "modes": [], "interior_surface": "graded"}


def hydrobem(ka, n):
    k = np.asarray(ka)
    w = np.sqrt(G * k * np.tanh(k * H_A))
    # two frequencies at a time bound the memory of the larger meshes
    body = body_of(n)
    opts = {
        "diffraction": True,
        "drift_pressure_integration": True,
        "drift_momentum_conservation": True,
        "frequency_threads": 2,
        "inner_threads": max(1, threads() // 2),
        "higher_order": n == 0,
    }
    r = solve(body, w, headings=[0.0], depth=H_A, options=opts)
    x = np.abs(r.quantity("excitation_diffraction")[:, 0, 0]) / (RHO * G)
    dp = r.quantity("drift_pressure_integration")[:, 0, 0].real / (RHO * G)
    dm = r.quantity("drift_momentum_conservation")[:, 0, 0].real / (RHO * G)
    return x, dp, dm


def main(quick=False):
    ka = np.linspace(0.2, 3.0, 8 if quick else 29)
    ref = np.array([analytic(x, H_A) for x in ka])
    # the deep-water check of the series against Havelock's closed sum
    deep = [abs(analytic(x, 40.0)[1] - havelock39(x)) / havelock39(x) for x in (0.5, 1.0, 2.0, 3.0)]
    ns = [8, 16] if quick else [8, 16, 32, 48]
    res = {}
    with Timer("bottom-mounted cylinder"):
        for n in ns:
            res[n] = hydrobem(ka, n)
        ho = hydrobem(ka, 0)
    fine = res[ns[-1]]
    env = {"density": RHO, "gravity": G, "depth": H_A}
    mesh_figure.draw("bottom_mounted_cylinder", [
        (f"low order, n = {ns[1]}", model(body_of(ns[1]), [1.0], [0.0], environment=env), False),
        ("higher order, 8 × 8 spans", model(body_of(0), [1.0], [0.0], environment=env, options={"higher_order": True}),
         True)], elev=25.0)  # fmt: skip
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5))
    ax = axes[0]
    ax.plot(ka, ref[:, 0], **{**REF, "label": "MacCamy & Fuchs (1954)"})
    ax.plot(ka, fine[0], color=C[0], label=f"low order, {ns[-1]} panels a quadrant")
    ax.plot(ka, ho[0], color=C[2], ls=":", lw=2.2, label="higher order")
    ax.set_xlabel("ka")
    ax.set_ylabel("|X₁|/(ρgAa²)")
    ax.set_title(f"surge exciting force, h/a = {H_A:g}")
    ax.legend()
    ax = axes[1]
    ax.plot(ka, ref[:, 1], **{**REF, "label": "analytic (pressure integration of the series)"})
    ax.plot(ka, fine[1], color=C[0], label="HydroBEM pressure integration")
    ax.plot(ka, fine[2], color=C[1], ls="--", label="HydroBEM momentum conservation")
    ax.plot(ka, ho[1], color=C[2], ls=":", lw=2.2, label="higher order, pressure integration")
    ax.set_xlabel("ka")
    ax.set_ylabel("F̄ₓ/(ρgA²a)")
    ax.set_title(f"mean surge drift, h/a = {H_A:g}")
    ax.legend()
    out = {"depth_over_a": H_A, "meshes": ns, "series_vs_havelock39_deep": max(deep)}
    for n in ns + [0]:
        x, dp, dm = res[n] if n else ho
        out[f"n{n}"] = {"force": float(np.abs(x - ref[:, 0]).max() / ref[:, 0].max()),
                        "drift_pressure": float(np.abs(dp - ref[:, 1]).max() / np.abs(ref[:, 1]).max()),
                        "drift_momentum": float(np.abs(dm - ref[:, 1]).max() / np.abs(ref[:, 1]).max())}  # fmt: skip
    ax = axes[2]
    for j, (key, lab) in enumerate((("force", "|X₁|"), ("drift_pressure", "drift, pressure integration"),
                                    ("drift_momentum", "drift, momentum conservation"))):  # fmt: skip
        ax.loglog(ns, [out[f"n{n}"][key] for n in ns], "o-", color=C[j], ms=4, label=lab)
    nn = np.array([ns[0], ns[-1]], dtype=float)
    ax.loglog(nn, out[f"n{ns[0]}"]["drift_momentum"] * (nn / ns[0]) ** -2, color="#888888", ls=":", label="∝ n⁻²")
    ax.loglog(nn, out[f"n{ns[0]}"]["drift_pressure"] * (nn / ns[0]) ** -1, color="#888888", ls="--", label="∝ n⁻¹")
    ax.set_xticks(ns, [str(n) for n in ns])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("panels a quarter circle (and down the side) n")
    ax.set_ylabel("max difference / max reference over ka")
    ax.set_title("mesh convergence")
    ax.legend(fontsize=7)
    save(fig, "bottom_cylinder")
    print("  bottom-mounted cylinder:", out)
    record("bottom_cylinder", out)


NAME = "bottom_mounted_cylinder"
GROUPS = ("bottom_cylinder",)
INDEX = {"case": "Bottom-mounted cylinder", "checked": "surge exciting force; mean surge drift (pressure, momentum)",
         "sources": "MacCamy & Fuchs (1954); Havelock (1940); the analytic scattering series",
         "methods": "low, higher"}  # fmt: skip


def summary(n):
    d = n["bottom_cylinder"]
    lo, hi = d[f"n{d['meshes'][-1]}"], d["n0"]
    return (f"exciting force within {pc(lo['force'])} and momentum drift within {pc(lo['drift_momentum'])} "
            f"(low order, $n = {d['meshes'][-1]}$); {size(hi['force'])} and "
            f"{size(max(hi['drift_pressure'], hi['drift_momentum']))} (higher order)")  # fmt: skip


def page(n):
    d = n["bottom_cylinder"]
    rows = [[f"low, $n = {k}$", e(d[f"n{k}"]["force"]), e(d[f"n{k}"]["drift_momentum"]),
             e(d[f"n{k}"]["drift_pressure"])]
            for k in d["meshes"]]  # fmt: skip
    h = d["n0"]
    rows.append([r"higher, $8 \times 8$ spans", e(h["force"]), e(h["drift_momentum"]), e(h["drift_pressure"])])
    header = ["Method, mesh", r"$\lvert X_1 \rvert$", "drift, momentum conservation", "drift, pressure integration"]
    nf = d["meshes"][-1]
    lo = d[f"n{nf}"]
    hd = max(h["drift_pressure"], h["drift_momentum"])
    return rf"""
# Bottom-mounted cylinder

**Result.** The surge exciting force of a cylinder standing on the sea bed agrees with the closed form
of MacCamy & Fuchs (1954) within {pc(lo['force'])} of its largest value in the low order
($n = {nf}$) and {size(h['force'])} in the higher order. The mean drift force agrees with the analytic
drift within {pc(lo['drift_momentum'])} (low order, momentum conservation) and {size(hd)} (higher
order, both ways). The low order's pressure drift converges more slowly ({pc(lo['drift_pressure'])}
at $n = {nf}$), because it needs the velocities of constant panels on the wall.

## The problem

A vertical cylinder of radius $a = 1$ m stands fixed on the bottom of water of depth $h = 2a$ and
pierces the free surface. Waves come from ahead at 29 values $0.2 \le ka \le 3$. We compare the surge
exciting force $\lvert X_1 \rvert/(\rho g A a^2)$ and the mean surge drift force
$\bar{{F}}_x/(\rho g A^2 a)$.

The mean drift force is the steady force that remains when the second-order pressure is averaged
over a wave period. HydroBEM computes it in two independent ways: by pressure integration over the
hull and along the waterline, and by momentum conservation, from the flux of momentum the waves carry
to and from the body. Both are compared.

![Meshes](../figures/mesh_{NAME}.png)

*Figure 1. The two meshes, expanded to the whole body and seen from above the free surface: the
wetted wall in blue, the interior surface in the waterplane in yellow. Left: low order, $n$ panels
per quarter circle and $n$ down the side, cosine-spaced (shown $n = {d['meshes'][1]}$), with the
graded interior surface. Right: higher order, the analytic cylinder with the draft equal to the
depth (patch edges in black, span edges in grey), $8 \times 8$ spans on the wall per quadrant and the
waterplane patch ($8 \times 2$) as interior surface, order 3, with the uniform mapping (the pressure
drift is not available with the nonuniform one).*

The interior surface removes the irregular frequencies: the integral equation of the exterior
problem fails at the eigenfrequencies of the water that would fill the inside of the column, the
first at $ka = j_{{0,1}} = 2.405$, where $J_0(ka) = 0$, and a lid over the waterplane inside the body
removes them. Without it the difference near $ka = 2.4$ exceeds the values of Table 1 many times.

## The reference

The force is the closed form of MacCamy & Fuchs (1954),

$$
\lvert X_1 \rvert = \frac{{4\rho g A \tanh kh}}{{k^2 \lvert H_1^{{(2)\prime}}(ka) \rvert}}.
$$

The mean drift is derived in `validation/bottom_cylinder.py` from the same analytic scattering
solution. On $r = a$ the total potential is $(gA/\omega) Z(z) S(\theta)$ with
$Z = \cosh k(z+h)/\cosh kh$ and

$$
S = \sum_m \epsilon_m (-i)^m \frac{{-2i}}{{\pi ka H_m^{{(2)\prime}}(ka)}} \cos m\theta,
$$

where the Wronskian of $J_m$ and $Y_m$ has simplified the series. The mean force of the fixed body is

$$
\bar{{F}}_x = \frac{{\rho}}{{4}} \iint \lvert\nabla\phi\rvert^2 n_x \,\mathrm{{d}}S
- \frac{{\rho g}}{{4}} \oint \lvert\zeta\rvert^2 n_x \,\mathrm{{d}}l,
$$

with the depth integrals in closed form. In deep water this is eq. (36) of Havelock (1940); the
script checks the series against Havelock's closed sum, eq. (39), to {e(d['series_vs_havelock39_deep'])}.
The force formula is MacCamy & Fuchs's, here derived from the same series (Havelock 1940, eq. (35),
gives its deep-water form). Both references are analytic, so the differences below are those of
the panel method alone.

## Agreement with the reference

![Bottom-mounted cylinder](../figures/bottom_cylinder.png)

*Figure 2. Left: $\lvert X_1 \rvert$ against MacCamy & Fuchs. Middle: the drift by pressure integration
and by momentum conservation against the analytic drift. Right: the largest difference over $ka$ of
each against the low-order $n$.*

Table 1 gives the largest difference of each quantity on every mesh.

*Table 1. Largest difference from the reference over $0.2 \le ka \le 3$, relative to the largest
reference value of each quantity.*

{table(header, rows)}

## Does it converge?

The four low-order meshes of Table 1 show the rate of each quantity. The exciting force and the
momentum drift converge as $n^{{-2}}$. The pressure drift converges about as $n^{{-1}}$, most slowly
at the larger $ka$: it needs the velocity along the wall of constant panels, whose error falls only
as $n^{{-1}}$ where the panel heights change from one panel to the next (the cosine spacing). Its
waterline term takes the potential of each waterline panel expanded from the centroid to the
waterline; the centroid value alone, half a panel height below it, would add a difference of first
order in that height, larger than all the others here.

The higher order integrates the pressure over the patches and along their waterline sides, and its
pressure drift converges about as fast as its momentum drift.

## Reproducing this page

The example [`examples/{NAME}`](../../../examples/{NAME}) runs the same body. This page, its figures
and its numbers are written by `validation/bottom_cylinder.py` through `validation/make_report.py`.
"""


if __name__ == "__main__":
    main("--quick" in sys.argv)
