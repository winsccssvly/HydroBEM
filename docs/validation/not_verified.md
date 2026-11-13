# What is not measured against an independent reference

**In short.** Most second-order loads beyond the fixed bottom-mounted cylinder, several first-order
cases whose publications give no usable numbers, and a number of features (walls, lids, dampers,
generalized modes, field outputs) have not been compared with an independent solution. They rest on
identities of the problem, on internal consistency and on convergence instead.

Either no published or analytic solution of these results is known to us, or the case's publication
gives no usable numbers. They are tested (test plan B), and they are as accurate as the parts they
are made of, as far as the checks show. A user relying on them for design should compare them with an
independent solution of the case.

## Second-order loads

- **The force of the second-order potential beyond the fixed bottom-mounted cylinder.** The one
  independent reference is Kim & Yue (1990, table 3;
  [second-order force](cases/second_order_bottom_cylinder.md)), a fixed body in finite depth. A
  floating body (the motion terms of theory §12.7), deep water, several bodies, and the quadrature
  zone and far field of theory §12.8–§12.9 rest on other evidence: the agreement of the direct and
  indirect methods (theory §12.6: one continuous problem, two discretizations), the convergence
  of the free surface ([cylinder](cases/cylinder.md)), a floating body held by a stiff spring giving
  the fixed body's coefficients, and two bodies mirroring each other. The potential QTF exists in the
  low order only.
- **One wall (theory §12.14).** It is checked against the open-sea problem of the body and its
  mirror image, an identity of the method; no solution of a body by a wall is compared.
- **Generalized modes in the second-order loads (theory §12.15).** The rigid modes given as
  generalized modes reproduce the rigid body's drift and QTFs (an identity), and the pressure drift
  of a flexible body converges to its momentum drift (a consistency of two methods). No published
  second-order load of a deformable body is compared.
- **The quadratic QTFs off the diagonal.** The diagonal is the mean drift, compared with the analytic
  drift of the bottom-mounted cylinder and with the momentum drift
  ([bottom-mounted cylinder](cases/bottom_mounted_cylinder.md)). Off the diagonal the QTFs are tested
  for their symmetries, for the drift at equal frequencies and for the continuity of the heading
  pairs. Newman's approximation is compared with the full QTF ([cylinder](cases/cylinder.md)), not
  with an independent solution.
- **The mean drift of the heave-plate cylinders and of the OWC devices.** The example of the heave
  plate computes the drift of the paper's model-scale bodies, and the paper of the OWC chamber shows a
  drift (its Fig. 11) without saying whether the device is restrained or floating; neither is
  compared.

## First-order cases without a numeric reference

- **The two-body OWC (hull and piston)** ([floating OWC piston](cases/floating_owc_piston.md)). The
  paper's curves are those of its coarse polygonal meshes and are not read; the only check is that
  the low order on fine panels converges to the higher order on the exact circles. The physical model
  (a rigid piston for the water column) is the paper's choice and is not checked.
- **A plate as part of a floating body** ([floating OWC cover](cases/floating_owc_cover.md)). It
  rests on the energy identity, the Haskind relation, the reciprocity of a hull and its cover (tests)
  and the methods against each other. The plate itself is checked against independent solutions only
  alone ([dock](cases/dock.md)).
- **The sloshing of fish tanks free in waves.** The surge and pitch RAOs of Newman (2021, Fig. 2)
  are read and compared away from their zeros and resonances
  ([fish tanks](cases/fish_tank_cylinders.md)); the sloshing amplitudes of the same figure, on its
  right axis, are not read, and the RAOs at the sloshing resonance are not compared. The discretized
  tank has a small damping (the closed tank has none, theory §10.2), so the RAOs that vanish at the
  sloshing resonance in the paper are small there but not zero.
- **The OWC chamber beyond its tables** ([OWC chamber](cases/floating_owc_chamber.md)). Several of the
  paper's results are not compared: the force of the chamber pressure (Fig. 3: its values do not agree
  with its axis label), the elevation at the walls (Table 4), the floating device's largest power
  (Table 5: its definition is not clear), the optimum turbine (Fig. 7: it needs the air volume, not
  given) and the wave profiles (Figs. 8, 9: the geometry stated does not agree with the axes).
- **The heave-plate experiments.** The measured loads of Han et al. (2024) include viscous effects
  (separation at the plate's edge) that potential flow leaves out; only their analytical lines are
  compared ([cylinder with a heave plate](cases/cylinder_with_heave_plate.md)).

## Features checked by identities only

Walls and channels, damping lids, dipole dampers, generalized modes, pressure surfaces other than an
axisymmetric chamber, the control-surface drift and the field outputs are checked by the identities
and limits of the [test plan](test-plan.md): a wall equals the body and its mirror image, a lid with
zero damping equals no lid, the control-surface drift equals the momentum drift, and so on. They are
not compared with published solutions of their own.
