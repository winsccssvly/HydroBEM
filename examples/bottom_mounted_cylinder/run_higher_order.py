"""A fixed bottom-mounted vertical cylinder in head seas in the higher order: the surge exciting force
against MacCamy & Fuchs (1954) and the mean surge drift force against the analytic solution.

The case: a vertical circular cylinder of radius a = 1 m standing on the sea bed in the depth
h = 2 m, fixed, in regular waves of amplitude A along +x, rho = 1000 kg/m^3, g = 9.80665 m/s^2, at
ka = 0.5, 1.0 and 2.0. The exciting force is the closed form of R. C. MacCamy & R. A. Fuchs
(1954), Wave forces on piles: a diffraction theory, Technical Memorandum No. 69, Beach Erosion
Board, U.S. Army Corps of Engineers, |X_1| = 4 rho g A tanh(kh) / (k^2 |H_1^(2)'(ka)|); the mean
drift force the pressure integration of the same scattering solution (in deep water eq. (36) of
T. H. Havelock (1940), The pressure of water waves upon a fixed obstacle, Proc. R. Soc. Lond. A
175(963), 409-421, doi:10.1098/rspa.1940.0066). The reference values below are those formulas
evaluated as in run.py (which computes them with SciPy), normalized as |X_1| / (rho g A a^2) and
F_x / (rho g A^2 a).

The higher-order mesh: the named analytic shape "circular_cylinder" of the engine with the draft
equal to the depth, the quadrant x, y >= 0 (the symmetry planes x = 0 and y = 0 give the rest) as
two exact analytic patches, the potential a B-spline of order 3 with Gauss orders 3 (outer) and 4
(inner):
- the wall from z = 0 to the sea bed, 8 spans around the quarter circle and 8 down (no bottom: the
  cylinder stands on the sea bed);
- the waterplane, the quarter disc in z = 0 inside the waterline, 8 x 2 spans, as the interior
  surface of the body (interior_surface "mesh", the second patch of the same mesh): it removes the
  irregular frequencies, the first at ka = 2.405;
- the uniform mapping: the drift by pressure integration is not available with the nonuniform
  mapping, whose profiles meet the waterline (the engine refuses the route).
The drift by both methods: pressure integration (drift_pressure_integration) and momentum
conservation (drift_momentum_conservation); in the higher order both converge about as fast as the
exciting force. One frequency is solved at a time (FREQUENCY_THREADS = 1).

Prints the model, the summary of the results and, at each ka, the exciting force and the two drift
forces beside the analytic values, with the largest difference relative to the largest analytic
value; saves out/run_higher_order/bottom_mounted_cylinder.hbr. The full comparison is in the
validation report, docs/validation/cases/bottom_mounted_cylinder.md.
"""

from pathlib import Path

import numpy as np

import hydrobem as hb

OUT = Path(__file__).resolve().parent / "out" / "run_higher_order"  # the outputs of this script
RADIUS, DEPTH = 1.0, 2.0  # m
DENSITY, GRAVITY = 1000.0, 9.80665
WALL = (8, 8, 3, 3, 3, 4)  # spans around the quarter circle and down the wall, spline orders, Gauss orders
WATERPLANE = (8, 2, 3, 3, 3, 4)  # spans around the quarter circle and along the radius, orders as the wall
FREQUENCY_THREADS = 1  # frequencies solved at once (more give the same results, faster)
# the analytic solution (MacCamy & Fuchs 1954; the pressure integration of its scattering solution)
# evaluated by run.py, h/a = 2: ka, |X1|/(rho g A a^2), Fx/(rho g A^2 a)
ANALYTIC = np.array([
    [0.5, 4.79871267, 0.44368987],
    [1.0, 4.15405097, 0.76238664],
    [2.0, 1.76072931, 0.63054096],
])  # fmt: skip
TOLERANCE = 2e-4  # of the largest analytic value of each quantity


def build() -> hb.Model:
    m = hb.Model()
    m.meshes.analytic(
        name="cylinder",
        shape="circular_cylinder",
        discretization=[WALL, WATERPLANE],
        parts=["side", "waterplane"],
        mapping="uniform",
        radius=RADIUS,
        draft=DEPTH,
        reference_length=RADIUS,
        symmetry="xz_and_yz",
        label="bottom-mounted cylinder a = 1 m, h = 2 m, analytic patches, 8 x 8 spans",
    )
    m.bodies.add(
        name="cylinder",
        mesh="cylinder",
        ranges=[(1, 1)],  # the wall patch
        rotation_center=[0.0, 0.0, 0.0],
        modes=[],
        interior_surface="mesh",
        interior_surface_mesh="cylinder",
        interior_surface_ranges=[(2, 2)],  # the waterplane patch
    )
    k = ANALYTIC[:, 0] / RADIUS
    m.environment.update(
        depth=DEPTH,
        density=DENSITY,
        gravity=GRAVITY,
        frequency_input="omegas",
        omegas=np.sqrt(GRAVITY * k * np.tanh(k * DEPTH)),
        headings=[0.0],
    )
    m.options.update(
        higher_order=True,
        frequency_threads=FREQUENCY_THREADS,
        inner_threads=1,
    )
    m.first_order.update(
        radiation=False,
        diffraction=True,
        motion=False,
    )
    m.second_order.update(
        drift_pressure_integration=True,
        drift_momentum_conservation=True,
    )
    return m


def main():
    m = build()
    print(m, "| route:", m.route())
    results = m.solve(lambda done, total: print(f"  frequency {done} of {total}"))
    path = OUT / "bottom_mounted_cylinder.hbr"
    m.save_results(
        path,
        overwrite=True,
    )
    print(results.summary())
    print("saved:", path)

    force = np.abs(
        results.coefficient(
            name="excitation_diffraction",
            row=("cylinder", "surge"),
        )[:, 0]
    )
    force /= DENSITY * GRAVITY * RADIUS**2
    scale = DENSITY * GRAVITY * RADIUS
    pressure = results.bodies["cylinder"].drift.pressure_integration[:, 0, 0].real / scale  # [F, heading pair, D]
    momentum = results.drift.momentum_conservation[:, 0, 0].real / scale  # [F, heading pair, surge sway yaw]
    ka, ref = ANALYTIC[:, 0], ANALYTIC[:, 1:]
    print("\n|X1|/(rho g A a^2) HydroBEM / MacCamy & Fuchs;  Fx/(rho g A^2 a) pressure, momentum / analytic")
    for i, x in enumerate(ka):
        print(f"  ka = {x:3.1f}:  {force[i]:.5f} / {ref[i, 0]:.5f};  "
              f"{pressure[i]:.5f}, {momentum[i]:.5f} / {ref[i, 1]:.5f}")  # fmt: skip
    diff = max(np.abs(got - r).max() / np.abs(r).max()
               for got, r in ((force, ref[:, 0]), (pressure, ref[:, 1]), (momentum, ref[:, 1])))  # fmt: skip
    print(f"check: largest difference / largest analytic value {diff:.1e} <= {TOLERANCE:.0e}:",
          "ok" if diff <= TOLERANCE else "NOT MET")  # fmt: skip


if __name__ == "__main__":
    main()
