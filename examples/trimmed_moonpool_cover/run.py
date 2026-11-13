"""A trimmed moonpool hull whose moonpool is closed by a cover that the engine generates.

The case: a floating annular cylinder (outer radius 10 m, moonpool radius 4 m, draft 6 m) given as
the whole hull above the water too, trimmed by the engine (pitched by 3 deg), its moonpool covered by
a rigid plate in the free surface: a hatch over the moonpool, or the roof of an oscillating water
column whose air chamber has collapsed. Under the pitch the moonpool wall is tilted, its waterline is
no longer the circle of the mesh, and a cover given as a mesh would leave z = 0. The cover is
therefore generated: free_surface_plates.add(inside=(x, y)) names a point inside the waterline loop
it covers, and the plate is the water of z = 0 inside that loop, triangulated, its edge on the
moonpool wall. No published solution of this body is known; the check is between two methods of the
mean drift that rest on different parts of the solution.

The parameters taken:
- the hull: outer wall, moonpool wall and bottom annulus, the walls from z = +4 m (the freeboard) to
  z = -6 m, the half y >= 0 (symmetry "xz": the pitch keeps the plane y = 0 and breaks x = 0), 24
  low-order panels around the half, 10 down each wall and 6 across the bottom; trim [0, 3, 0]
  (no sinkage, pitch 3 deg, no roll);
- the cover: generated over the moonpool, from the point (0, 0) in the body axes of the trimmed hull;
- the interior surface "waterline" (generated from the trimmed outer waterline; the covered moonpool
  has none of its own) removes the irregular frequencies of the hull;
- the motions free in surge, heave and pitch, the mass the displaced mass, the centre of gravity
  2 m below the free surface, radii of gyration 6 m (pitch) and 7 m (yaw);
- infinite depth, heading 0 deg, six frequencies from omega = 0.4 to 1.4 rad/s and omega = infinity.

What is computed with the cover (theory §9.7 of the documentation):
- A(infinity), the added mass at the infinite-frequency limit, where the free surface is phi = 0 and
  the plate a rigid thin sheet of the odd-image problem;
- the mean drift by its three methods: momentum conservation in the far field (H. Maruo (1960), The
  drift of a body floating on waves, J. Ship Res. 4(3), 1-10; J. N. Newman (1967), The drift force
  and moment on ships in waves, J. Ship Res. 11(1), 51-60, doi:10.5957/jsr.1967.11.1.51), pressure
  integration over the body (J. A. Pinkster (1980), Low frequency second order wave exciting forces
  on floating structures, PhD thesis, Delft University of Technology) and a control surface around
  the body (X. B. Chen (2007), Middle-field formulation for the computation of wave-drift loads,
  J. Eng. Math. 59(1), 61-82, doi:10.1007/s10665-006-9074-x), here an automatic one: a circle of
  radius 18 m, 10 m deep, generated about the trimmed outer waterline (the covered moonpool's
  waterline left out). They are all available because the cover's edge lies on its own body's wall;
- the quadratic QTF at the difference frequencies, by pressure integration and through the control
  surface (Pinkster 1980; Chen 2007); the diagonal of each is its mean drift;
- the potential QTF, the force of the second-order potential, by the indirect method (B. Molin
  (1979), Second-order diffraction loads upon three-dimensional bodies, Applied Ocean Research 1(4),
  197-202, doi:10.1016/0141-1187(79)90027-0): its free surface ends at the outer waterline and the
  cover, which the generated free-surface panels leave out.

The pressure drift converges slowly at the sharp bottom edges of the hull (as h^(1/3) in the panel
size h, theory §11.2): its velocity terms are singular there, most of all near the heave and pitch
resonances of a floating body. The drifts by momentum conservation and through the control surface
need no velocity on the body and converge fast: they are the ones to trust, and the check is that
these two agree. On 16, 24 and 36 panels around the half (with 6, 10, 14 down each wall and 4, 6,
8 across the bottom) the surge drift through the control surface differs from the momentum drift
by 4.3e-2, 3.4e-2 and 2.4e-2 of the latter's peak (at the heave and pitch resonance near
omega = 1 rad/s), falling with the panel size; the pressure drift by 0.56, 0.49 and 0.46: there it
is half as large again as the other two.

Prints the panels and the area of the generated cover, A(infinity), the surge drift by the three
methods, the surge difference QTF of both quadratic methods and of the potential part at three
pairs of frequencies, and checks the horizontal drift through the control surface against the
momentum drift, relative to the latter's largest value.
"""

from pathlib import Path

import numpy as np

import hydrobem as hb

OUT = Path(__file__).resolve().parent / "out" / "run"  # the outputs of this script
R_OUTER, R_INNER, DRAFT, FREEBOARD = 10.0, 4.0, 6.0, 4.0  # m
TRIM = [0.0, 3.0, 0.0]  # sinkage (m), pitch (deg), roll (deg)
DENSITY, GRAVITY = 1000.0, 9.80665
OMEGAS = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4]  # rad/s
N_AROUND, N_WALL, N_BOTTOM = 24, 10, 6  # panels around the half, down each wall, across the bottom
CONTROL_SURFACE = dict(radius=18.0, depth=10.0, panel_size=1.5)  # m
FREQUENCY_THREADS = 1  # frequencies solved at once (more give the same results, faster)
TOLERANCE = 5e-2  # the control-surface drift against the momentum drift, of the latter's peak


def half_hull():
    """The half y >= 0 of the hull, its walls up to the freeboard: outer wall (normals outward),
    moonpool wall (normals towards the axis) and bottom annulus (normals down)."""
    phi = np.linspace(0.0, np.pi, N_AROUND + 1)
    z = np.linspace(FREEBOARD, -DRAFT, N_WALL + 1)
    r = np.linspace(R_INNER, R_OUTER, N_BOTTOM + 1)

    def p(rad, k, depth):
        return [rad * np.cos(phi[k]), rad * np.sin(phi[k]), depth]

    out = []
    for k in range(N_AROUND):
        for j in range(N_WALL):
            out.append([p(R_OUTER, k, z[j]), p(R_OUTER, k, z[j + 1]), p(R_OUTER, k + 1, z[j + 1]),
                        p(R_OUTER, k + 1, z[j])])  # fmt: skip
            out.append([p(R_INNER, k, z[j]), p(R_INNER, k + 1, z[j]), p(R_INNER, k + 1, z[j + 1]),
                        p(R_INNER, k, z[j + 1])])  # fmt: skip
        for j in range(N_BOTTOM):
            out.append([p(r[j], k, -DRAFT), p(r[j], k + 1, -DRAFT), p(r[j + 1], k + 1, -DRAFT),
                        p(r[j + 1], k, -DRAFT)])  # fmt: skip
    return np.array(out)


def build() -> hb.Model:
    m = hb.Model()
    m.meshes.add(
        name="hull",
        kind="panels",
        vertices=half_hull(),
        reference_length=R_OUTER,
        symmetry="xz",
        label="moonpool hull R = 10 m, r = 4 m, T = 6 m, walls to z = +4 m",
    )
    hull = m.bodies.add(
        name="hull",
        mesh="hull",
        trim=TRIM,  # the mesh is the whole hull: placed, pitched and cut at z = 0 by the engine
        rotation_center=[0.0, 0.0, 0.0],
        modes=["surge", "heave", "pitch"],
        center_of_gravity=[0.0, 0.0, -2.0],
        mass="displaced",
        inertia_input="radii_of_gyration",
        radii_of_gyration=[6.0, 6.0, 7.0],
        interior_surface="waterline",
        control_surface="automatic",
        control_surface_outer="circle",
        control_surface_radius=CONTROL_SURFACE["radius"],
        control_surface_depth=CONTROL_SURFACE["depth"],
        control_surface_panel_size=CONTROL_SURFACE["panel_size"],
    )
    hull.free_surface_plates.add(
        inside=(0.0, 0.0),  # a point of the water in the moonpool, body axes of the trimmed hull
    )
    m.environment.update(
        depth=np.inf,
        density=DENSITY,
        gravity=GRAVITY,
        frequency_input="omegas",
        omegas=[*OMEGAS, np.inf],  # with the limit omega = infinity
        headings=[0.0],
    )
    m.options.update(
        frequency_threads=FREQUENCY_THREADS,
        inner_threads=1,
    )
    m.first_order.update(
        radiation=True,
        diffraction=True,  # in the total potential, the default with a free-surface plate
        motion=True,
    )
    m.second_order.update(
        drift_momentum_conservation=True,
        drift_pressure_integration=True,
        drift_control_surface=True,
        qtf_quadratic_pressure_integration=True,
        qtf_quadratic_control_surface=True,
        qtf_potential_indirect=True,
        qtf_difference=True,
        qtf_sum=False,
    )
    return m


def main():
    m = build()
    print(m, "| route:", m.route())
    plate = m.bodies["hull"].panels(role="free_surface_plate")
    print(f"the generated cover: {len(plate.areas)} panels, area {plate.areas.sum():.2f} m^2 "
          f"(the untrimmed moonpool pi r^2 = {np.pi * R_INNER**2:.2f} m^2)")  # fmt: skip
    results = m.solve(lambda done, total: print(f"  frequency {done} of {total}"))
    path = OUT / "trimmed_moonpool_cover.hbr"
    m.save_results(
        path,
        overwrite=True,
    )
    print(results.summary())
    coords = [("hull", "surge"), ("hull", "heave"), ("hull", "pitch")]
    a_inf = np.array([[results.coefficient("added_mass_infinite_frequency", i, j) for j in coords] for i in coords])
    print("\nA(infinity) on surge, heave, pitch [kg, kg m, kg m^2]:")
    for name, row in zip(("surge", "heave", "pitch"), a_inf, strict=True):
        print(f"  {name:6s}" + "".join(f"{x:13.4e}" for x in row))

    w = results.omegas
    momentum = results.drift.momentum_conservation.real[:, 0, :]  # [F, (surge, sway, yaw)]
    pressure = results.bodies["hull"].drift.pressure_integration.real[:, 0, :]  # [F, 6]
    control = results.bodies["hull"].drift.control_surface.real[:, 0, :]  # [F, 6]
    print("\nmean surge drift [N/m^2] (per unit wave amplitude squared):")
    print("  omega [rad/s]   momentum   control surface   pressure")
    for k, omega in enumerate(w):
        print(f"  {omega:13.2f} {momentum[k, 0]:10.1f} {control[k, 0]:17.1f} {pressure[k, 0]:10.1f}")

    quadratic_p = results.bodies["hull"].qtf.quadratic.pressure_integration.difference[:, :, 0, 0]
    quadratic_c = results.bodies["hull"].qtf.quadratic.control_surface.difference[:, :, 0, 0]
    potential = results.bodies["hull"].qtf.potential.indirect.difference[:, :, 0, 0]
    print("\nsurge difference QTF [N/m^2], |Q-(omega_a, omega_b)|:")
    print("  omega_a omega_b   quadratic (pressure)   quadratic (control surface)   potential")
    for a, b in ((1, 2), (2, 3), (3, 4)):
        print(f"  {w[a]:7.2f} {w[b]:7.2f} {abs(quadratic_p[a, b]):22.1f} {abs(quadratic_c[a, b]):29.1f} "
              f"{abs(potential[a, b]):11.1f}")  # fmt: skip

    horizontal = control[:, :2]  # surge, sway
    diff = np.abs(horizontal - momentum[:, :2]).max() / np.abs(momentum[:, :2]).max()
    print(f"check: control-surface against momentum drift, difference {diff:.1e} <= {TOLERANCE:.0e}: "
          f"{'ok' if diff <= TOLERANCE else 'NOT MET'}")  # fmt: skip
    print("saved:", path)


if __name__ == "__main__":
    main()
