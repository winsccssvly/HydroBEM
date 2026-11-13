# References (public)

Public sources cited by the project, with the subject of each. Bibliographic
details are checked against the publications themselves, or against the
publishers' records where no copy was at hand; local copies of the papers are
not part of the repository.

## Theory

- J. V. Wehausen, E. V. Laitone (1960). Surface waves. In *Handbuch der Physik* (Encyclopedia of
  Physics) 9, Fluid Dynamics III, Springer, Berlin, 446–778. Free-surface Green functions,
  infinite and finite depth.
- F. John (1950). On the motion of floating bodies II. Simple harmonic motions. *Comm. Pure Appl.
  Math.* 3(1):45–101. https://doi.org/10.1002/cpa.3160030106. Finite-depth Green function
  series, irregular frequencies.
- J. N. Newman (1977). *Marine Hydrodynamics*. MIT Press (40th anniversary edition 2018). Linear
  theory, added mass, damping, Haskind relations, energy relations; the added mass of the disc in
  unbounded fluid as the limit of the oblate spheroid (figure 4.8), half of which a free-surface
  plate has at $\omega = \infty$ (theory §9.7).
- Lord Rayleigh (J. W. Strutt) (1896). *The Theory of Sound*, vol. 2, 2nd ed. Macmillan, London
  (§307, the correction for an open end). The piston in a rigid plane and its end correction
  $8a/(3\pi)$: the added mass $(8/3)\rho a^3$ of a disc in $z = 0$ at $\omega = 0$ (the free-surface plates,
  theory §9.7).
- H. Maruo (1960). The drift of a body floating on waves. *J. Ship Res.* 4(3):1–10. The far-field
  drift force.
- J. N. Newman (1967). The drift force and moment on ships in waves. *J. Ship Res.* 11(1):51–60.
  https://doi.org/10.5957/jsr.1967.11.1.51. The far-field drift force and yaw moment.
- J. A. Pinkster (1980). *Low frequency second order wave exciting forces on floating
  structures*. PhD thesis, Delft University of Technology. The mean drift force and moment by
  direct integration of the pressure over the wetted surface, and its quadratic second-order
  forces at the difference frequencies (the quadratic QTFs).
- J. N. Newman (1974). Second-order, slowly-varying forces on vessels in irregular waves. In
  *Proc. Int. Symp. on Dynamics of Marine Vehicles and Structures in Waves*, London, 182–186.
  The approximation of the difference-frequency QTF by the mean drift.
- B. Molin (1979). Second-order diffraction loads upon three-dimensional bodies. *Applied Ocean
  Research* 1(4):197–202. https://doi.org/10.1016/0141-1187(79)90027-0. The force of the
  second-order potential by Green's identity with the radiation potential (the indirect method).
- C.-H. Lee (1995). *WAMIT Theory Manual*. Report 95-2, Department of Ocean Engineering,
  Massachusetts Institute of Technology. The second-order free-surface and body forcing, the
  second-order incident wave (ch. 3).
- M.-H. Kim, D. K. P. Yue (1989). The complete second-order diffraction solution for an
  axisymmetric body. Part 1. Monochromatic incident waves. *J. Fluid Mech.* 200:235–264.
  https://doi.org/10.1017/S0022112089000649. The free surface beyond a radius by Fourier-Bessel
  series.
- M.-H. Kim, D. K. P. Yue (1990). The complete second-order diffraction solution for an
  axisymmetric body. Part 2. Bichromatic incident waves and body motions. *J. Fluid Mech.*
  211:557–593. https://doi.org/10.1017/S0022112090001690. Table 3: the semi-analytic sum- and
  difference-frequency potential forces on a bottom-mounted cylinder (a reference of the tests,
  tests/data/kim_yue1990.txt).
- Y. R. Choi, S. Y. Hong, H. S. Choi (2001). An analysis of second-order wave forces on floating
  bodies by using a higher-order boundary element method. *Ocean Engineering* 28(1):117–138.
  https://doi.org/10.1016/S0029-8018(99)00064-5. The free surface in inner, intermediate and outer
  regions.
- X. B. Chen (2007). Middle-field formulation for the computation of wave-drift loads. *J. Eng.
  Math.* 59(1):61–82. https://doi.org/10.1007/s10665-006-9074-x. The mean drift through a
  control surface around the body.

## Green function evaluation

- J. N. Newman (1985). Algorithms for the free-surface Green function. *J. Eng. Math.*
  19(1):57–67. https://doi.org/10.1007/BF00055041.
- J. N. Newman, P. D. Sclavounos (1988). The computation of wave loads on large offshore
  structures. *BOSS '88*, Trondheim. Logarithmic singularity of the wave kernel and its panel
  integration.
- P.-Y. Wuillaume (2022). Study of the free surface Green function at zero forward speed in
  finite water depth using Newman's method. *Applied Ocean Research* 120:103036.
  https://doi.org/10.1016/j.apor.2021.103036. Newman's finite-depth decomposition, the functions
  $I_m$ and their asymptotics, the zero-frequency gauge constant.
- X. B. Chen (2004). Hydrodynamics in offshore and naval applications, Part I. Keynote lecture,
  6th International Conference on HydroDynamics, Perth. Damping-lid (dissipative) free surface.
- C. M. Linton (1999). A new representation for the free-surface channel Green's function.
  *Applied Ocean Research* 21(1):17–25. https://doi.org/10.1016/S0141-1187(98)00032-7. Channel
  Green function.

## Panel methods

- J. L. Hess, A. M. O. Smith (1964). Calculation of non-lifting potential flow about arbitrary
  three-dimensional bodies. *J. Ship Res.* 8(4):22–44 (September 1964).
  https://doi.org/10.5957/jsr.1964.8.4.22.
- J. N. Newman (1986). Distributions of sources and normal dipoles over a quadrilateral panel.
  *J. Eng. Math.* 20(2):113–126. https://doi.org/10.1007/BF00042771.
- C.-H. Lee, J. N. Newman, X. Zhu (1996). An extended boundary integral equation method for the
  removal of irregular frequency effects. *Int. J. Numer. Methods Fluids* 23(7):637–660.
  https://doi.org/10.1002/(SICI)1097-0363(19961015)23:7<637::AID-FLD437>3.0.CO;2-3.
- C.-H. Lee, J. N. Newman (2005). Computation of wave effects using the panel method. In
  S. K. Chakrabarti (ed.), *Numerical Models in Fluid-Structure Interaction*, Advances in Fluid
  Mechanics 42, WIT Press, Southampton, ch. 6. The higher-order (B-spline) method and the
  hypersingular equation of thin (dipole) sheets solved with the body equation.
- H. D. Maniar (1995). *A three dimensional higher order panel method based on B-splines*. PhD
  thesis, Massachusetts Institute of Technology. https://hdl.handle.net/1721.1/11127.
- J. N. Newman (1994). Wave effects on deformable bodies. *Applied Ocean Research* 16(1):47–59.
  https://doi.org/10.1016/0141-1187(94)90013-2. Generalized modes: their normal velocities and
  hydrostatic restoring; the displacement as the sum of the mode shapes times their amplitudes and
  the generalized force as the pressure projected on the mode shape (the generalized modes of the
  second-order loads).
- R. W. Ogden (1984). *Non-Linear Elastic Deformations*. Ellis Horwood, Chichester (Dover reprint
  1997). Nanson's formula, the transformation of an oriented surface element by the deformation
  gradient (the turned and stretched surface element of a deforming body).
- A. T. Chwang (1983). A porous-wavemaker theory. *J. Fluid Mech.* 132:395–406.
  https://doi.org/10.1017/S0022112083001676. The jump of a porous sheet proportional to the
  normal velocity through it (dipole dampers).
- C.-H. Lee, J. N. Newman (2000). Wave effects on large floating structures with air cushions.
  *Marine Structures* 13(4):315–330. https://doi.org/10.1016/S0951-8339(00)00014-9.
  Prescribed-pressure surfaces, generalized free-surface modes and coupled air-cushion response.

## Analytic and reference results

- A. Hulme (1982). The wave forces acting on a floating hemisphere undergoing forced periodic
  oscillations. *J. Fluid Mech.* 121:443–463. https://doi.org/10.1017/S0022112082001980.
- C. J. R. Garrett (1971). Wave forces on a circular dock. *J. Fluid Mech.* 46(1):129–139.
  https://doi.org/10.1017/S0022112071000430. The eigenfunction matching of a truncated cylinder
  in finite depth (the reference data of `scripts/gen_cylinder_reference.py`).
- R. W. Yeung (1981). Added mass and damping of a vertical cylinder in finite-depth waters.
  *Applied Ocean Research* 3(3):119–133. https://doi.org/10.1016/0141-1187(81)90101-2.
- D. D. Bhatta, M. Rahman (1995). Wave loadings on a vertical cylinder due to heave motion.
  *Int. J. Math. Math. Sci.* 18(1):151–170. https://doi.org/10.1155/S0161171295000202.
- D. D. Bhatta (2003). Surge motion on a floating cylinder in water of finite depth. *Int. J.
  Math. Math. Sci.* 2003(57):3643–3656. https://doi.org/10.1155/S0161171203209285.
- R. C. MacCamy, R. A. Fuchs (1954). *Wave forces on piles: a diffraction theory*. Technical
  Memorandum No. 69, Beach Erosion Board, U.S. Army Corps of Engineers, Washington. The exciting
  force on a bottom-mounted vertical cylinder (the closed form of the validation report, derived
  there from the scattering series).
- T. H. Havelock (1940). The pressure of water waves upon a fixed obstacle. *Proc. R. Soc. Lond. A*
  175(963):409–421. https://doi.org/10.1098/rspa.1940.0066. The first-order force (eq. 35) and the
  mean drift force (eqs. 36, 39) on a vertical circular cylinder in deep water (the validation
  report).
- H. Kagemoto, D. K. P. Yue (1986), G. X. Wu (1995), S. A. Mavrakos (1991) (listed under
  Examples): the interaction theories behind the references of `scripts/gen_cylinder_array_reference.py`
  (eigenfunction matching of each cylinder with the multiple scattering of its propagating and
  evanescent modes, Graf's addition theorem, Abramowitz & Stegun 9.1.79) and
  `scripts/gen_sphere_array_reference.py` (the multipoles of submerged spheres with their
  free-surface images and the wave integral), the tests of several bodies.
- D. V. Evans (1982). Wave-power absorption by systems of oscillating surface pressure
  distributions. *J. Fluid Mech.* 114:481–499. https://doi.org/10.1017/S0022112082000196. The
  capture width of an axisymmetric pressure chamber, $k P_{max} / P_w = 1$ (a test of pressure
  surfaces).
- O. M. Faltinsen, A. N. Timokha (2009). *Sloshing*. Cambridge University Press, §5.4.4. The
  forced surge and pitch of a circular tank as a series (a test of internal tanks).
- R. Eatock Taylor, E. R. Jefferys (1986). Variability of hydrodynamic load predictions for a
  tension leg platform. *Ocean Engineering* 13(5):449–490.
  https://doi.org/10.1016/0029-8018(86)90033-8. The ISSC tension-leg platform (the dimensions
  of the TLP test meshes).

## Examples

The cases the scripts of `examples/` set up (each script states the parameters it takes).

- V. Raghavan, E. Loukogeorgaki, N. Mantadakis, A. V. Metrikine, G. Lavidas (2024). HAMS-MREL, a
  new open source multiple body solver for marine renewable energies: model description,
  application and validation. *Renewable Energy* 237:121577.
  https://doi.org/10.1016/j.renene.2024.121577. Its four validation cases: two truncated
  cylinders, two submerged spheres, three truncated cylinders, a floating oscillating water column
  as two bodies (`examples/two_truncated_cylinders`, `two_submerged_spheres`,
  `three_truncated_cylinders`, `floating_owc_piston`); its floating OWC with a light piston on the
  inner free surface, a free-surface plate (`examples/floating_owc_cover`, with Evans 1982).
- C. J. R. Garrett (1971), R. W. Yeung (1981) (listed under Analytic and reference
  results): the circular dock of zero draft in finite depth, its eigenfunction matching
  computed by `scripts/gen_cylinder_reference.py` with the draft 0 (`examples/dock`).
- H. Kagemoto, D. K. P. Yue (1986). Interactions among multiple three-dimensional bodies in water
  waves: an exact algebraic method. *J. Fluid Mech.* 166:189–209.
  https://doi.org/10.1017/S0022112086000101. Two truncated cylinders in finite depth (geometry 2).
- G. X. Wu (1995). The interaction of water waves with a group of submerged spheres. *Applied
  Ocean Research* 17(3):165–184. https://doi.org/10.1016/0141-1187(95)00010-0. Two submerged
  spheres (case 2).
- S. A. Mavrakos (1991). Hydrodynamic coefficients for groups of interacting vertical
  axisymmetric bodies. *Ocean Engineering* 18(5):485–515.
  https://doi.org/10.1016/0029-8018(91)90027-N. Three truncated cylinders on an equilateral
  triangle.
- S. A. Mavrakos, D. N. Konispoliatis (2012). Hydrodynamics of a free floating vertical
  axisymmetric oscillating water column device. *Journal of Applied Mathematics* 2012:142850.
  https://doi.org/10.1155/2012/142850. A thick-walled oscillating water column chamber with a
  linear air turbine, restrained and free in heave; open and thin-walled chambers
  (`examples/floating_owc_chamber`).
- S.-Y. Han, B. Bouscasse, V. Leroy, D. Le Touzé (2024). Linear diffraction and radiation theory
  of water waves by a truncated vertical circular cylinder with heave plate in deep and shallow
  drafts. *International Journal of Naval Architecture and Ocean Engineering* 16:100580.
  https://doi.org/10.1016/j.ijnaoe.2023.100580. A truncated cylinder with a thick or a thin heave
  plate (a dipole sheet), deep and shallow drafts: exciting forces, added mass, damping, radiated
  waves and mean drift (`examples/cylinder_with_heave_plate`).
- J. N. Newman (2021). Linear motions of fish tanks. *36th International Workshop on Water Waves
  and Floating Bodies* (IWWWFB), 25–28 April 2021, 4 pp. Circular cylindrical tanks with walls of
  zero thickness floating in infinite depth: the forced surge and pitch of the tank alone, checked
  against the interior series of O. M. Faltinsen, A. N. Timokha (2009), *Sloshing*, Cambridge
  University Press, §5.4.4, and the RAOs and sloshing in waves (`examples/fish_tank_cylinders`).

## Mathematics and algorithms

- M. Abramowitz, I. A. Stegun (eds.) (1964). *Handbook of Mathematical Functions with Formulas,
  Graphs, and Mathematical Tables*. National Bureau of Standards Applied Mathematics Series 55,
  U.S. Government Printing Office, Washington (tenth printing 1972, with corrections). Bessel and
  Struve functions, asymptotic expansions, Gauss–Legendre rules.
- L. N. Trefethen (2019). *Approximation Theory and Approximation Practice*, Extended Edition.
  SIAM, Philadelphia. https://doi.org/10.1137/1.9781611975949. Chebyshev interpolation and its
  convergence (ch. 2–4, 8; the tables of the Green functions).
- G. H. Golub, C. F. Van Loan (2013). *Matrix Computations*, 4th ed. Johns Hopkins University
  Press, Baltimore. The LU factorization with partial pivoting, blocked (right-looking; §3.2.11,
  §3.6.1).
- L. F. Richardson (1911). The approximate arithmetical solution by finite differences of
  physical problems involving differential equations, with an application to the stresses in a
  masonry dam. *Phil. Trans. R. Soc. Lond. A* 210:307–357. Richardson extrapolation (the
  converged reference values of the scripts).
- J. Ma, V. Rokhlin, S. Wandzura (1996). Generalized Gaussian quadrature rules for systems of
  arbitrary functions. *SIAM J. Numer. Anal.* 33(3):971–996. The rule exact for $p^k$ and
  $p^k \log p$ (the radial variable of the Duffy integrals with the log term, theory §7.1).
- M. G. Duffy (1982). Quadrature over a pyramid or cube of integrands with a singularity at a
  vertex. *SIAM J. Numer. Anal.* 19(6):1260–1262. The transformation of a corner singularity
  (near-singular patch integrals).
- A. Van Oosterom, J. Strackee (1983). The solid angle of a plane triangle. *IEEE Trans. Biomed.
  Eng.* BME-30(2):125–126. https://doi.org/10.1109/TBME.1983.325207. The solid angle of a
  triangle (the dipole integral of a panel, theory §6; field points inside a body).
- J. A. C. Weideman (1994). Computation of the complex error function. *SIAM J. Numer. Anal.*
  31(5):1497–1518. The rational approximation of the Faddeeva function (channel kernel).
- L. Piegl, W. Tiller (1997). *The NURBS Book*, 2nd ed. Springer, Berlin. B-spline basis
  evaluation (algorithms A2.1–A2.3); rational B-spline surfaces and their derivatives (ch. 4)
  and the rational quarter circle (§7.3) of the test meshes.
- D. Eberly (2002, revised 2024). *Triangulation by ear clipping*. Geometric Tools. Polygons
  with holes.
- J. Ruppert (1995). A Delaunay refinement algorithm for quality 2-dimensional mesh generation.
  *J. Algorithms* 18(3):548–585. https://doi.org/10.1006/jagm.1995.1021. J. R. Shewchuk (2002).
  Delaunay refinement algorithms for triangular mesh generation. *Comput. Geom.* 22(1–3):21–74.
  https://doi.org/10.1016/S0925-7721(01)00047-5. The refined waterline interior surface
  (theory §14).
- I. E. Sutherland, G. W. Hodgman (1974). Reentrant polygon clipping. *Comm. ACM* 17(1):32–42.
  https://doi.org/10.1145/360767.360802. The clipping of a polygon by a convex one (the overlaps
  of the surfaces of $z = 0$ with free-surface plates, reference/inputs-rules §1 rule 3).

## Software and standards

- F. Johansson et al. *mpmath*: a Python library for arbitrary-precision floating-point
  arithmetic (the reference values and the tables, at 30 to 40 digits).
- G. Fowler, L. C. Noll, K.-P. Vo, D. Eastlake. The FNV non-cryptographic hash algorithm
  (IETF Internet-Draft). FNV-1a 64, the checksums of the generated tables.
- ISO/IEC 3309 (HDLC), IEEE 802.3: the CRC-32 of the model and results files.
- H. Hinnant. *chrono-Compatible Low-Level Date Algorithms*
  (https://howardhinnant.github.io/date_algorithms.html, public domain): the civil date of a day
  count, the UTC time stamps of the written files.
