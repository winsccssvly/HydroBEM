"""A truncated vertical circular cylinder with a heave plate at its bottom (Han et al. 2024): the
exciting forces, added mass and damping in surge, heave and pitch of the cylinder with a thick plate
(their Fig. 4) and with a thin plate as a dipole sheet (their Fig. 6), in the low and the higher
order, against the lines of their matched eigenfunction expansions read from the figures
(docs/validation/cases/cylinder_with_heave_plate.md).

    python validation/cylinder_with_heave_plate.py [--quick]
"""

from __future__ import annotations

import sys

import mesh_figure
import numpy as np
from common import READ, RHO, C, G, Timer, at_read, e, pct, plt, read_differences, record, save, solve_model, table

from bodies import surface
from problems import model

NAME = "cylinder_with_heave_plate"
DEPTH = 100.0
THICK = {"name": "thick", "dc": 12.0, "dd": 24.0, "draft": 20.0, "t": 6.0, "plate": "thick"}
THIN = {"name": "thin", "dc": 7.0, "dd": 20.0, "draft": 15.5, "t": 0.1, "plate": "sheet"}
NU = {"thick": [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0],
      "thin": [0.05, 0.2, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]}  # fmt: skip
QUANTITIES = ["Fx", "Fz", "My", "a11", "a33", "a55", "b11", "b33", "b55"]
# Han et al. (2024), the lines of the analytical method read by eye from the page images ({nu: value})
FIG_4 = {
    "Fx": {0.5: 1.34, 1.0: 1.68, 2.0: 1.12}, "Fz": {0.5: 0.53, 1.0: 0.575, 2.0: 0.235},
    "My": {0.5: 1.38, 1.0: 1.56, 2.0: 0.71}, "a11": {0.05: 1.52, 0.5: 1.79, 1.0: 1.62, 2.0: 0.92, 3.0: 0.835},
    "a33": {0.05: 2.63, 0.5: 2.68, 1.0: 2.58, 2.0: 2.47, 3.0: 2.50},
    "a55": {0.05: 2.20, 0.5: 2.41, 1.0: 2.13, 2.0: 1.65, 3.0: 1.75},
    "b11": {0.5: 0.22, 1.0: 0.70, 2.0: 0.62, 3.0: 0.33}, "b33": {0.5: 0.07, 1.0: 0.16, 2.0: 0.05},
    "b55": {0.5: 0.24, 1.0: 0.61, 2.0: 0.25, 3.0: 0.05},
}  # fmt: skip
FIG_6 = {
    "Fx": {0.5: 1.14, 1.0: 1.46, 2.0: 1.08}, "Fz": {0.5: 0.77, 1.0: 0.45, 2.0: 0.07},
    "My": {0.5: 1.35, 0.75: 1.49, 2.0: 0.58}, "a11": {0.05: 1.65, 0.75: 1.87, 2.0: 1.21, 3.0: 1.08},
    "a33": {0.05: 6.88, 0.5: 6.93, 1.0: 6.78, 2.0: 6.77, 3.0: 6.78},
    "a55": {0.05: 6.01, 0.5: 6.17, 1.0: 5.98, 2.0: 5.54, 3.0: 5.62},
    "b11": {1.5: 0.67, 3.0: 0.33}, "b33": {0.5: 0.16, 1.0: 0.11}, "b55": {1.0: 0.50, 2.0: 0.17},
}  # fmt: skip
FIGURE = {"thick": FIG_4, "thin": FIG_6}


def cosine(n):
    return 0.5 * (1.0 - np.cos(np.pi * np.arange(n + 1) / n))


def meridian(points, size):
    """The vertices (r, z) of the polyline, each side divided cosine-spaced, at most about `size`."""
    r, z = [points[0][0]], [points[0][1]]
    for (r0, z0), (r1, z1) in zip(points[:-1], points[1:], strict=True):
        n = max(2, int(np.ceil(0.5 * np.pi * np.hypot(r1 - r0, z1 - z0) / size)))
        s = cosine(n)[1:]
        r += list(r0 + (r1 - r0) * s)
        z += list(z0 + (z1 - z0) * s)
    return np.array(r), np.array(z)


def sweep(r, z, n):
    phi = np.linspace(0.0, 0.5 * np.pi, n + 1)
    p = lambda i, j: [r[i] * np.cos(phi[j]), r[i] * np.sin(phi[j]), z[i]]  # noqa: E731
    return [[p(i, j), p(i + 1, j), p(i + 1, j + 1), p(i, j + 1)] for i in range(len(r) - 1) for j in range(n)]


def low_body(geo, sheet_refinement=1.0, size=1.0 / 18.0, n_theta=24):
    """The quadrant: the meridian of at most D_c/18 a side (cosine-spaced), 24 panels around; the
    thick plate part of the hull, the thin plate a dipole sheet in its mid-plane (its intervals
    `sheet_refinement` times finer)."""
    rc, rd, d, t = geo["dc"] / 2, geo["dd"] / 2, geo["draft"], geo["t"]
    size *= geo["dc"]
    parts = {}
    if geo["plate"] == "thick":
        r, z = meridian([(rc, 0.0), (rc, t - d)], size)
        rp, zp = meridian([(rc, t - d), (rd, t - d), (rd, -d), (0.0, -d)], size)
        parts["hull"] = np.r_[r, rp[1:]], np.r_[z, zp[1:]]
    else:
        zs = 0.5 * t - d
        parts["hull"] = meridian([(rc, 0.0), (rc, zs), (0.0, zs)], size)
        parts["plate"] = meridian([(rc, zs), (rd, zs)], size / sheet_refinement)
    rg, quads = {}, []
    for k, (r, z) in parts.items():
        q = sweep(r, z, n_theta)
        rg[k] = (len(quads) + 1, len(quads) + len(q))
        quads += q
    mesh = surface(np.array(quads), symmetry="xz_and_yz")
    b = {"name": "body", "mesh": (mesh, rg["hull"]), "modes": ["surge", "heave", "pitch"], "interior_surface": "graded"}
    if "plate" in rg:
        b["dipoles"] = [{"mesh": (mesh, rg["plate"])}]
    return b


def segments(geo):
    rc, rd, d, t = geo["dc"] / 2, geo["dd"] / 2, geo["draft"], geo["t"]
    if geo["plate"] == "thick":
        hull = [((rc, 0.0), (rc, t - d)), ((rc, t - d), (rd, t - d)), ((rd, t - d), (rd, -d)), ((rd, -d), (0.0, -d))]
        return [(s, "smooth", "hull") for s in hull] + [(((0.0, 0.0), (rc, 0.0)), "linear", "interior")]
    zs = 0.5 * t - d
    return [(((rc, 0.0), (rc, zs)), "smooth", "hull"), (((rc, zs), (0.0, zs)), "smooth", "hull"),
            (((0.0, 0.0), (rc, 0.0)), "linear", "interior"), (((rc, zs), (rd, zs)), "smooth", "plate")]  # fmt: skip


def higher_body(geo, size=1.0 / 8.0):
    """Revolved straight segments of the meridian as analytic patches a quadrant (exact circles),
    the spans at most D_c/8, order 3, clustered at the ends of each segment ("smooth")."""
    size *= geo["dc"]
    spans = lambda length: max(1, int(np.ceil(length / size - 1e-9)))  # noqa: E731
    rows, patches, rg = [], [], {}
    for i, (((r0, z0), (r1, z1)), profile, role) in enumerate(segments(geo)):
        rows.append([spans(0.5 * np.pi * max(r0, r1)), spans(np.hypot(r1 - r0, z1 - z0)), 3, 3, 3, 4])
        patches.append({"type": "AnalyticPatch", "map": "line", "profile": profile, "a": 1.0, "b": 1.0,
                        "theta0": 0.0, "theta1": 0.5 * np.pi, "p": [r0, z0, r1, z1]})  # fmt: skip
        first = rg.get(role, (i + 1, i + 1))[0]
        rg[role] = (first, i + 1)
    mesh = {"kind": "analytic_patches", "discretization": np.array(rows), "symmetry": "xz_and_yz", "patches": patches}
    b = {"name": "body", "mesh": (mesh, rg["hull"]), "modes": ["surge", "heave", "pitch"],
         "interior_surface": (mesh, rg["interior"])}  # fmt: skip
    if "plate" in rg:
        b["dipoles"] = [{"mesh": (mesh, rg["plate"])}]
    return b


def build(geo, nu, kind):
    w = np.sqrt(np.asarray(nu) * G / geo["dc"])
    if kind == "higher":
        b = higher_body(geo)
    else:
        b = low_body(geo, {"low": 1.0, "coarse": 2.0, "fine": 4.0}[kind])
    return model(b, w, [0.0], environment={"depth": DEPTH, "density": RHO, "gravity": G},
                 options={"higher_order": kind == "higher"})  # fmt: skip


def quantities(r, geo):
    dc = geo["dc"]
    x = r.excitation.diffraction[:, 0, :]
    out = {}
    for key, mode, n in (("Fx", "surge", 2), ("Fz", "heave", 2), ("My", "pitch", 3)):
        out[key] = np.abs(x[:, r.coordinate("body", mode)[0]]) / (RHO * G * dc**n)
    for key, mode, n in (("11", "surge", 3), ("33", "heave", 3), ("55", "pitch", 5)):
        out["a" + key] = r.coefficient("added_mass", ("body", mode), ("body", mode)) / (RHO * dc**n)
        out["b" + key] = r.coefficient("radiation_damping", ("body", mode), ("body", mode)) / (r.omegas * RHO * dc**n)
    return out


def main(quick=False):
    out, res = {}, {}
    for geo in (THICK, THIN):
        nu = np.array(NU[geo["name"]][::2] if quick else NU[geo["name"]])
        kinds = ("low", "higher") if geo["plate"] == "thick" else ("coarse", "fine", "higher")
        for kind in kinds:
            with Timer(f"heave plate: {geo['name']} {kind}"):
                res[geo["name"], kind] = quantities(solve_model(build(geo, nu, kind)), geo)
        if geo["plate"] == "sheet":
            c, f = res["thin", "coarse"], res["thin", "fine"]
            res["thin", "low"] = {k: 2 * f[k] - c[k] for k in f}  # extrapolated in the sheet's interval
        fig_read = FIGURE[geo["name"]]
        out[geo["name"]] = {"nu": nu}
        for kind in ("low", "higher") + (("fine",) if geo["plate"] == "sheet" else ()):
            out[geo["name"]][kind] = read_differences(nu, res[geo["name"], kind], fig_read)
        hi, lo = res[geo["name"], "higher"], res[geo["name"], "low"]
        out[geo["name"]]["low_vs_higher"] = {q: float(np.abs(lo[q] - hi[q]).max() / np.abs(hi[q]).max())
                                             for q in QUANTITIES}  # fmt: skip
        out[geo["name"]]["a33_first"] = {k: float(res[geo["name"], k]["a33"][0]) for k in kinds + ("low",)}
        fig, axes = plt.subplots(3, 3, figsize=(10.5, 8.0), sharex=True)
        for ax, q in zip(axes.ravel(), QUANTITIES, strict=True):
            if q in fig_read:
                x, y, _ = at_read(nu, hi[q], fig_read[q])
                ax.plot(x, y, **{**READ, "label": "Han et al. (2024), read"})
            ax.plot(nu, lo[q], color=C[0], label="low order" + (", extrapolated" if geo["plate"] == "sheet" else ""))
            ax.plot(nu, hi[q], color=C[1], ls="--", label="higher order")
            ax.set_title(q, fontsize=9)
        for ax in axes[2]:
            ax.set_xlabel("ω²D_c/g")
        axes[0, 0].legend(fontsize=7)
        save(fig, f"{NAME}_{geo['name']}")
    mesh_figure.draw(NAME, [("thick plate, low order", build(THICK, [1.0], "low"), False, {"cut": (1, -1)}),
                            ("thick plate, higher order", build(THICK, [1.0], "higher"), True, {"cut": (1, -1)}),
                            ("thin plate (sheet), low order", build(THIN, [1.0], "coarse"), False,
                             {"cut": (1, -1), "elev": 15.0}),
                            ("thin plate (sheet), higher order", build(THIN, [1.0], "higher"), True,
                             {"cut": (1, -1), "elev": 15.0})], elev=25.0, size=(3.4, 3.6))  # fmt: skip
    print("  heave plate:", out)
    record(NAME, out)


def summary(n):
    d = n[NAME]
    hi = {**{k + "_thick": v for k, v in d["thick"]["higher"].items()}, **d["thin"]["higher"]}
    worst = max(v for k, v in hi.items() if not k.startswith("b33"))
    b33 = max(d["thick"]["higher"]["b33"], d["thin"]["higher"]["b33"])
    return (f"the figures read agree within {pct(worst, 0)} of the largest value read (higher order); "
            f"$b_{{33}}$ near its small peak, read approximately, within {pct(b33, 0)}")


INDEX = {"case": "Cylinder with a heave plate",
         "checked": "$X$, $A$, $B$ in surge, heave, pitch; thick plate and thin plate (dipole sheet)",
         "sources": "Han et al. (2024) figs. 4, 6 (read)", "methods": "low, higher"}  # fmt: skip


def head(q):
    """The header of a quantity in the page's tables: Fx as F_x, a11 as a_11."""
    return f"${q[0]}_{{{q[1:]}}}$"


def page(n):
    d = n[NAME]
    rows = []
    for geo, label in (("thick", "thick plate"), ("thin", "thin plate")):
        for kind, name in (("low", "low order" + (", extrapolated" if geo == "thin" else "")),
                           ("fine", "low order, fine sheet"), ("higher", "higher order")):  # fmt: skip
            if kind in d[geo]:
                rows.append([f"{label}, {name}"] + [pct(d[geo][kind][q], 1) for q in QUANTITIES])
    rows_lh = [[label] + [e(d[geo]["low_vs_higher"][q]) for q in QUANTITIES]
               for geo, label in (("thick", "thick plate"), ("thin", "thin plate (extrapolated)"))]  # fmt: skip
    a = d["thin"]["a33_first"]
    hi = {**{k + "_thick": v for k, v in d["thick"]["higher"].items()}, **d["thin"]["higher"]}
    worst = max(v for k, v in hi.items() if not k.startswith("b33"))
    b33 = max(d["thick"]["higher"]["b33"], d["thin"]["higher"]["b33"])
    thick_lh = max(d["thick"]["low_vs_higher"].values())
    return rf"""
# Truncated cylinder with a heave plate

**Result.** Both methods reproduce the analytical lines of Han et al. (2024) for a cylinder with a
heave plate, thick or thin, to about the accuracy of reading the figures: the higher order within
{pct(worst, 0)} of the largest value read, except near the small peak of the heave damping $b_{{33}}$,
read approximately ({pct(b33, 0)}). For the thick plate the two methods agree with each other within
{pct(thick_lh, 1)}. The thin plate's sheet converges slowly in the low order and is extrapolated.

## The problem

A surface-piercing circular cylinder stands on a coaxial circular plate and is held at rest in waves.
We compare the exciting forces, added mass and damping in surge, heave and pitch, with a thick plate
and with a thin plate. The geometries are those of Han et al. (2024), their table 1:

- the thick plate (after the DeepCwind column): $D_c = 12$ m, plate diameter $D_d = 24$ m, draft
  $d = 20$ m to the plate's bottom, plate thickness 6 m;
- the thin plate (after HiPRWind): $D_c = 7$ m, $D_d = 20$ m, $d = 15.5$ m, the plate a sheet of zero
  thickness in its mid-plane $z = -15.45$ m (both faces wetted; the cylinder's bottom is part of the
  hull).

The depth is 100 m, the heading $0^\circ$ and the range $0.05 \le \omega^2 D_c/g \le 3$. The
normalization is that of the figures: $F_x$, $F_z / (\rho g D_c^2 A)$, $M_y / (\rho g D_c^3 A)$;
$a_{{11}}$, $a_{{33}} / (\rho D_c^3)$, $a_{{55}} / (\rho D_c^5)$, the damping also over $\omega$; pitch is
about the origin in the free surface on the axis.

A thin sheet stands for a plate whose thickness is negligible: instead of two nearly coincident faces,
the plate is one surface carrying the jump of the potential across it (a distribution of dipoles).

![Meshes](../figures/mesh_{NAME}.png)

*Figure 1. The four meshes, expanded to the whole body and cut open (a quarter removed) to show the
inside: the wetted surface in blue, the interior surface in the waterplane in yellow, the thin sheet
in red. From the left, the thick plate in the low and the higher order, then the thin plate in the low
and the higher order. Low order, a quadrant: the meridian divided cosine-spaced into intervals of at
most $D_c/18$, 24 panels around, the graded interior surface; the thin plate a dipole sheet (shown:
the sheet of $D_c/36$; the comparison uses $D_c/36$ and $D_c/72$ and extrapolates, the constant
panels erring in proportion to the interval across the sheet's free edge). Higher order: each
straight segment of the meridian revolved as an analytic patch (exact circles; patch edges in black,
span edges in grey), spans of at most $D_c/8$, order 3, clustered at the ends of each segment; the
waterplane disc is the interior surface, the sheet an annular patch of dipoles.*

## The reference

The reference is Han et al. (2024), figures 4 (thick plate) and 6 (thin plate): the lines of their
matched eigenfunction expansions (MEEM). The paper has no tables of results, so the lines are **read
by eye** from the page images at a few frequencies each, to about 1 % of the values. Near the peaks
of $F_z$ and $b_{{33}}$ they are read only approximately. Their eigenfunction series is truncated and
includes the thin plate's thickness (0.1 m), which the sheet leaves out.

## Agreement with the figures

![Thick plate](../figures/{NAME}_thick.png)

*Figure 2. Thick plate (Han et al. figure 4): values read (crosses), low order (solid), higher order
(dashed).*

![Thin plate](../figures/{NAME}_thin.png)

*Figure 3. Thin plate (Han et al. figure 6): values read (crosses), low order extrapolated in the
sheet's interval (solid), higher order (dashed).*

The thick plate agrees with the lines of Han et al.'s figure 4 to about the reading accuracy in both
methods (Table 1). The remaining differences of the thin plate, largest near the peaks of $F_z$ and
$b_{{33}}$ and in $a_{{55}}$, are within the reading of the figure and the truncation and thickness of
the paper's series.

*Table 1. Largest difference from Han et al.'s figures at the points read, relative to the largest
value read of each quantity.*

{table(["Method"] + [head(q) for q in QUANTITIES], rows)}

## Do the methods agree?

The figures are read to about 1 %, so they cannot show smaller differences. The two methods against
each other can (Table 2): for the thick plate they agree to a few 1e-3.

*Table 2. Largest difference between the methods over the frequencies, relative to the largest value
of the higher order.*

{table(["Low against higher order"] + [head(q) for q in QUANTITIES], rows_lh)}

The thin plate's sheet converges slowly in the low order, at the rate of the interval across its free
edge. Its $a_{{33}}$ at $\omega^2 D_c/g = 0.05$ is {a["coarse"]:.4f} ($D_c/36$), {a["fine"]:.4f}
($D_c/72$), {a["low"]:.4f} extrapolated and {a["higher"]:.4f} in the higher order, whose spans
clustered at the edge converge in one solve; the figure reads 6.88.

## Reproducing this page

The example [`examples/{NAME}`](../../../examples/{NAME}) also runs the paper's model-scale bodies with
their radiated waves and drift. This page, its figures and its numbers are written by
`validation/cylinder_with_heave_plate.py` through `validation/make_report.py`.
"""


if __name__ == "__main__":
    main("--quick" in sys.argv)
