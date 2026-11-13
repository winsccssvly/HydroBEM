# 2. Radiation, diffraction and coefficients

The boundary-value problem of chapter 1 is linear, so its solution splits into parts that are
solved separately: one radiation problem per mode of motion and one diffraction problem per wave
heading. This chapter defines the hydrodynamic coefficients that are built from those parts, the
added mass, the damping and the exciting force, and the two ways HydroBEM computes the exciting
force. The equations of motion that use them are in chapter 3.

## 2.1 Radiation, excitation and coefficients

### The radiation potential

A body that moves with the complex amplitude $\xi_j$ in mode $j$ has the velocity $i\omega\xi_j$
in that mode. Since $\phi_j$ is the potential of a unit velocity (1.13), the radiation potential of
the actual motion is the sum

$$
\phi_R = i\omega \sum_j \xi_j \phi_j. \tag{2.1}
$$

### Added mass and damping

The linearized pressure is $p = -i\omega\rho\phi$, with $\rho$ the density of the water. The
force on the body in mode $i$ is $-\int_{S_B} p\, n_i \,\mathrm{d}S$, the normal $n$ pointing into
the fluid (1.9). For the radiation potential of a unit velocity in mode $j$ this force, written as
an added-mass and a damping term, defines both coefficients:

$$
A_{ij} - \frac{i}{\omega} B_{ij} = -\rho \int_{S_B} n_i \phi_j \,\mathrm{d}S. \tag{2.2}
$$

The added mass $A_{ij}$ is the part in phase with the acceleration, the damping $B_{ij}$ the part
in phase with the velocity.

### The exciting force by pressure integration

The exciting force is the force on the fixed body in the incident wave. Integrating the pressure of
the total diffraction potential gives

$$
X_i = i\omega\rho \int_{S_B} n_i (\phi_I + \phi_S) \,\mathrm{d}S. \tag{2.3}
$$

It splits into the Froude–Krylov force of the undisturbed incident wave and the scattering force
of the disturbance the body makes:

$$
X_i^{\mathrm{FK}} = i\omega\rho \int n_i \phi_I \,\mathrm{d}S, \qquad
X_i^{S} = i\omega\rho \int n_i \phi_S \,\mathrm{d}S. \tag{2.4}
$$

### The exciting force by the Haskind relation

Green's second identity turns the scattering force into an integral of the radiation potential.
The exciting force then needs no diffraction solution, only the radiation potential of mode $i$:

$$
X_i = i\omega\rho \left[\int_{S_B} \left(n_i \phi_I - \phi_i\, \frac{\partial\phi_I}{\partial n}\right) \mathrm{d}S
+ (K - \kappa) \int_L \phi_i \phi_I \,\mathrm{d}S\right]. \tag{2.5}
$$

The terms of (2.5) adapt to the special parts of a model:

- With walls or a channel, the incident wave $\phi_I$ in the relation includes its reflections
  (§1.2).
- The lid term, inside the bracket, is the difference of the lid condition of §9.1 from the free
  surface; without lids there is none.
- A thin sheet (§9.5) contributes

  $$
  -i\omega\rho \int \mu_i\, \frac{\partial\phi_I}{\partial n} \,\mathrm{d}S, \qquad \mu_i = \phi_i^+ - \phi_i^-, \tag{2.6}
  $$

  with the jump $\mu_i$ of the radiation potential across the sheet; the $n_i \phi_I$ terms of
  its two faces cancel.

Which of (2.3) and (2.5) drives the motions of a body is chosen per body (§3.1).

### What is returned

All force rows are returned, one for every coordinate of the model. The added mass and damping
have the columns of the solved modes only, labelled by `radiation_mode_ids`
(reference/results).
