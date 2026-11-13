# Examples

Scripts to read and run: each shows how to pose a problem, solve it and read its results, and ends
with a short check against a published or analytic value, or an identity of the solution (a line
`check: ... ok`). Every script stands on its own: it makes its meshes itself, imports nothing but
NumPy and `hydrobem` (and SciPy where stated) and reads no file. It writes the model with its
results (`.hbr`) to `out/<script>/` in its own folder. The full comparison of a case that has one
(all frequencies, the convergence with the mesh, both methods) is in its page of the validation
report, `docs/validation/cases/<folder>.md`, which the script's docstring names. The one
exception is `floating_owc_cover/`: the page of that name is about another body, a smaller tube
with a cover of its own body, and the docstrings of the folder name the pages of the related
cases. The sources are listed in [docs/references.md](../docs/references.md).

```
python examples/floating_hemisphere/run.py
```

## Requirements

`hydrobem` with NumPy. Two scripts also need SciPy (Bessel functions and their zeros):
`bottom_mounted_cylinder/run.py` and `fish_tank_cylinders/run_forced_tank.py`; they say so in their
docstrings. Install it with the optional extra:

```
pip install hydrobem[examples]
```

## The cases

`run.py` uses low-order panels; `run_higher_order.py`, where present, the same case on
exact-geometry patches (options.higher_order).

*Table 1. The example folders, their scripts and their cases.*

| Folder | Scripts | The case |
|---|---|---|
| `cylinder/` | `run.py` | a floating cylinder in deep water with every input of the model written out by name with its default |
| `floating_hemisphere/` | `run.py`, `run_higher_order.py` | heave and surge added mass and damping against Hulme (1982) |
| `truncated_cylinder_finite_depth/` | `run.py`, `run_higher_order.py` | heave and surge added mass and damping in finite depth against eigenfunction matching |
| `bottom_mounted_cylinder/` | `run.py` (SciPy), `run_higher_order.py` | the exciting force and the mean drift against the closed forms of MacCamy & Fuchs and Havelock |
| `second_order_bottom_cylinder/` | `run.py` | the force of the second-order potential against Kim & Yue (1990) |
| `two_truncated_cylinders/` | `run.py`, `run_higher_order.py` | two cylinders side by side in finite depth against Kagemoto & Yue (1986) |
| `three_truncated_cylinders/` | `run.py`, `run_higher_order.py` | three cylinders on an equilateral triangle against Mavrakos (1991) |
| `two_submerged_spheres/` | `run.py`, `run_higher_order.py` | two submerged spheres against Wu (1995) |
| `floating_owc_piston/` | `run.py`, `run_higher_order.py` | an oscillating water column as a hull and a piston filling its column (Raghavan et al. 2024) |
| `floating_owc_cover/` | `run.py`, `run_higher_order.py` | the same column covered by a light piston of zero thickness, a free-surface plate: the heave RAOs of piston and hull, the added mass at $\omega = \infty$, the mean drift by momentum conservation and by pressure integration, the difference-frequency QTFs |
| `floating_owc_chamber/` | `run.py`, `run_higher_order.py`, `run_open_heave.py`, `run_thin_chamber.py` | an oscillating water column with its air chamber and a linear turbine, restrained and free in heave, against Mavrakos & Konispoliatis (2012) |
| `dock/` | `run.py`, `run_higher_order.py` | a floating circular dock of zero draft in finite depth, a free-surface plate, against eigenfunction matching (Garrett 1971); its added mass at $\omega = \infty$ and its mean drift by momentum conservation |
| `trimmed_moonpool_cover/` | `run.py` | a pitched moonpool hull whose moonpool cover the engine generates from the trimmed waterline: the added mass at $\omega = \infty$, the mean drift by its three methods, the quadratic and potential QTFs |
| `cylinder_with_heave_plate/` | `run.py`, `run_higher_order.py`, `run_thin_plate.py`, `run_model_scale.py` | a truncated cylinder with a thick or a thin heave plate against Han et al. (2024) |
| `fish_tank_cylinders/` | `run.py`, `run_higher_order.py`, `run_forced_tank.py` (SciPy), `run_covered_tank.py` | floating tanks of zero wall thickness with the sloshing inside (Newman 2021; Faltinsen & Timokha 2009); a tank half and fully covered by plates on its free surface |
