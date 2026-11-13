//! The engine's problem, built from the objects of a model (document_build.rs): its types, the
//! geometry derived from them (the roles of the elements, the origins of the meshes) and the counts
//! of its coordinates. The model rules are in rules/, the hydrostatics in hydrostatics.rs, the
//! field points in field_points.rs.

use crate::model_codes::{
    DamperCondition, Domain, Excitation, Formulation, InteriorMethod, Mass, NewmanDrifts, Origin, PressureShape,
    QtfFrequencies, QtfHeadings, QuantitySet, RestoringConvention, WaterlineTerm,
};

use crate::constants::STANDARD_GRAVITY;
use crate::error::{Diagnostic, Result};
use crate::geometry::{PLANE_TOL, Placement, Reflection, Vec3, max_of, v3};
use crate::mesh::{Mesh, Range, RecordShape};
use crate::progress::CancelToken;
use crate::route::{Route, select_route};

#[derive(Clone, Debug)]
pub struct PressureMode {
    pub name: String,
    pub shape: PressureShape,
    pub stiffness: f64,
    pub damping: f64,
}

impl Default for PressureMode {
    fn default() -> Self {
        PressureMode { name: String::new(), shape: PressureShape::Uniform, stiffness: 0.0, damping: 0.0 }
    }
}

/// A dipole damping surface: thin-sheet elements of the body mesh with the jump (i epsilon / omega) psi_n
/// (theory §9.5): a damper of the body (porous, psi_n = phi_n - v_n, in the body forces), or a fixed
/// damper placed in the body's frame (at rest, psi_n = phi_n, in no force).
#[derive(Clone, Debug)]
pub struct Damper {
    pub panels: Vec<Range>,
    pub epsilon: f64,
    pub condition: DamperCondition,
    pub index: i32, // its zero-based place in its list: the dipole dampers of the body, of its tank, or the fixed dampers
    pub name: String, // its name in its list ("" none)
    pub label: String, // the damper in the messages, by its list ("" none: by its place among the body's dampers)
}

impl Default for Damper {
    fn default() -> Self {
        Damper {
            panels: Vec::new(),
            epsilon: 0.0,
            condition: DamperCondition::FreeImpervious,
            index: -1,
            name: String::new(),
            label: String::new(),
        }
    }
}

impl Damper {
    /// not in the body forces (a fixed damper)
    pub fn free(&self) -> bool {
        self.condition == DamperCondition::FreeImpervious
    }
    /// driven by the body velocity (a damper of the body)
    pub fn porous(&self) -> bool {
        self.condition == DamperCondition::FixedPorous
    }
}

/// A fixed damper at rest in global coordinates, a sheet of no body: every element of
/// its mesh, the jump (i epsilon / omega) phi_n, in no force (theory §9.5).
#[derive(Clone, Debug, Default)]
pub struct FixedDamper {
    pub mesh: i32,
    pub epsilon: f64,
    pub index: i32,   // its place in the fixed dampers
    pub name: String, // its name in that list ("" none)
    pub place: i32,   // its zero-based place in the model's list, for the messages (-1: index)
}

/// A generalized mode: the displacement (u, v, w) at a body-local point of an
/// element is the sum of the polynomial pieces that select the element.
#[derive(Clone, Copy, Debug, Default)]
pub struct ModeTerm {
    pub component: i32,
    pub px: i32,
    pub py: i32,
    pub pz: i32,
    pub c: f64,
}
#[derive(Clone, Debug, Default)]
pub struct ModePiece {
    pub panels: Vec<Range>, // empty: every element
    pub terms: Vec<ModeTerm>,
}
#[derive(Clone, Debug)]
pub struct GeneralizedMode {
    pub name: String,
    pub parity: i32, // parity of this rigid mode (1..6)
    pub pieces: Vec<ModePiece>,
}

impl Default for GeneralizedMode {
    fn default() -> Self {
        GeneralizedMode { name: String::new(), parity: 1, pieces: Vec::new() }
    }
}

impl GeneralizedMode {
    /// The displacement at body-local x on the one-based element rec.
    pub fn displacement(&self, x: &Vec3, rec: i32) -> Vec3 {
        let mut pw = [[0.0f64; 9]; 3]; // the powers 0..8 of x, y, z
        for k in 0..3 {
            pw[k][0] = 1.0;
            for e in 1..9 {
                pw[k][e] = pw[k][e - 1] * x[k];
            }
        }
        let mut d = Vec3::ZERO;
        for p in &self.pieces {
            let mut inside = p.panels.is_empty();
            for r in &p.panels {
                inside = inside || (rec >= r.first && rec <= r.last);
            }
            if !inside {
                continue;
            }
            for t in &p.terms {
                d[t.component as usize] += t.c * pw[0][t.px as usize] * pw[1][t.py as usize] * pw[2][t.pz as usize];
            }
        }
        d
    }

    /// The gradient of the displacement at body-local x on the one-based element rec:
    /// out[i][k] = ∂u_i/∂x_k, of the polynomial pieces that select the element.
    pub fn gradient(&self, x: &Vec3, rec: i32) -> [[f64; 3]; 3] {
        let mut pw = [[0.0f64; 9]; 3]; // the powers 0..8 of x, y, z
        for k in 0..3 {
            pw[k][0] = 1.0;
            for e in 1..9 {
                pw[k][e] = pw[k][e - 1] * x[k];
            }
        }
        let mut d = [[0.0f64; 3]; 3];
        for p in &self.pieces {
            let mut inside = p.panels.is_empty();
            for r in &p.panels {
                inside = inside || (rec >= r.first && rec <= r.last);
            }
            if !inside {
                continue;
            }
            for t in &p.terms {
                let e = [t.px as usize, t.py as usize, t.pz as usize];
                for k in 0..3 {
                    if e[k] == 0 {
                        continue;
                    }
                    let mut v = t.c * e[k] as f64;
                    for l in 0..3 {
                        v *= if l == k { pw[l][e[l] - 1] } else { pw[l][e[l]] };
                    }
                    d[t.component as usize][k] += v;
                }
            }
        }
        d
    }
}

/// An internal tank: elements of the body mesh around a fluid domain of its own.
#[derive(Clone, Debug)]
pub struct Tank {
    pub panels: Vec<Range>,         // its walls, the sheets inside it and its plates on the body mesh
    pub density: f64,               // relative to Environment::density
    pub surface_level: Option<f64>, // body-local z of the tank free surface; None: the highest vertex
    pub name: String,               // its name in the tanks of the body ("" none)
    pub place: i32,                 // its zero-based place in the model's list, for the messages (-1: its index)
    // the plates of its free surface (theory §10.2), moving with the body: elements of the body
    // mesh (also in panels) or a mesh of their own (body coordinates), at the tank's surface level
    pub plates: Vec<Plate>,
}

impl Default for Tank {
    fn default() -> Self {
        Tank {
            panels: Vec::new(),
            density: 1.0,
            surface_level: None,
            name: String::new(),
            place: -1,
            plates: Vec::new(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct PressureSurface {
    pub panels: Vec<Range>,
    pub mesh: Option<usize>, // a mesh of its own; None: elements of the body mesh
    pub modes: Vec<PressureMode>,
    pub chamber_coupling: bool,
    pub convention: RestoringConvention,
    pub name: String, // its name in the pressure surfaces of the body ("" none)
    pub place: i32,   // its zero-based place in the model's list, for the messages (-1: its index)
}

impl Default for PressureSurface {
    fn default() -> Self {
        PressureSurface {
            panels: Vec::new(),
            mesh: None,
            modes: Vec::new(),
            chamber_coupling: false,
            convention: RestoringConvention::Variational,
            name: String::new(),
            place: -1,
        }
    }
}

/// A massless rigid plate in the free surface z = 0 (theory §9.7): elements of the body mesh or a
/// mesh of its own (body coordinates), the water below it only, moving with the body's rigid modes
/// and with the generalized modes whose pieces name its elements.
#[derive(Clone, Debug)]
pub struct Plate {
    pub panels: Vec<Range>,
    pub mesh: Option<usize>, // a mesh of its own, None: elements of the body mesh
    pub name: String,        // its name in the free-surface plates of the body ("" none)
    pub place: i32,          // its zero-based place in the model's list, for the messages (-1: its index)
}

impl Default for Plate {
    fn default() -> Self {
        Plate { panels: Vec::new(), mesh: None, name: String::new(), place: -1 }
    }
}

#[derive(Clone, Debug)]
pub struct Body {
    pub name: String,
    pub mesh: i32,
    pub position: Vec3,
    pub yaw: f64,
    pub rotation_center: Vec3,
    pub modes: Vec<i32>,
    pub has_cg: bool,
    pub cg: Vec3,
    pub mass: Option<Mass>, // a mass given: a number or "displaced"; None: no mass
    pub has_inertia: bool,  // inertia given, as components or as radii of gyration
    pub inertia: [f64; 6],
    pub has_gyration: bool, // inertia m r_ij |r_ij| from the radii of gyration
    pub gyration: [f64; 9],
    pub inertia_origin: Origin,
    pub inertia_point: Vec3,
    pub stiffness: Vec<f64>,
    pub damping: Vec<f64>,
    pub stiffness_rows: i32,
    pub stiffness_cols: i32,
    pub damping_rows: i32,
    pub damping_cols: i32,
    pub stiffness_origin: Origin,
    pub damping_origin: Origin,
    pub stiffness_point: Vec3,
    pub damping_point: Vec3,
    pub excitation: Excitation,
    pub interior_method: InteriorMethod,
    pub interior_panels: Vec<Range>,
    pub interior_mesh: Option<usize>, // the mesh of an external interior surface
    pub dipole_panels: Vec<Range>,
    pub dipole_mesh: Option<usize>, // the mesh of its dipoles when they have one of their own
    pub dampers: Vec<Damper>,       // dipole damping surfaces
    pub generalized: Vec<GeneralizedMode>, // after the six rigid modes
    pub mode_mass: Vec<f64>,        // (6 + G)^2 row-major, or empty
    pub mode_damping: Vec<f64>,
    pub mode_stiffness: Vec<f64>,
    pub tanks: Vec<Tank>,                           // internal tanks
    pub control_surface: Option<usize>,             // mesh of the control surface
    pub control_surface_alternative: WaterlineTerm, // the CL term, or the WL and S_f terms (theory §11.3)
    pub pressure: Vec<PressureSurface>,
    pub plates: Vec<Plate>,        // free-surface plates
    pub body_points: Vec<Vec3>,    // specified points of the body outputs, body-local
    pub body_point_neighbors: i32, // low order: centroids interpolated at body points
    // the dipoles on the body's mesh (of the body, then of its tanks): for the queries, the owner
    // of a dipole element (its place in the dipoles of the body or of its tank)
    pub dipole_owners: Vec<DipoleOwner>,
    // per element of the mesh of the dipoles of another mesh (dipole_mesh, or the mesh the higher
    // order joined): its dipole's place in the dipoles of the body
    pub dipole_mesh_owner: Vec<i32>,
}

/// The elements of one dipole on a body's mesh and its place in its list (the dipoles of the body,
/// or of its tank).
#[derive(Clone, Debug, Default)]
pub struct DipoleOwner {
    pub panels: Vec<Range>,
    pub index: i32,
}

impl Default for Body {
    fn default() -> Self {
        Body {
            name: String::new(),
            mesh: -1,
            position: Vec3::ZERO,
            yaw: 0.0,
            rotation_center: Vec3::ZERO,
            modes: Vec::new(),
            has_cg: false,
            cg: Vec3::ZERO,
            mass: None,
            has_inertia: false,
            inertia: [0.0; 6],
            has_gyration: false,
            gyration: [0.0; 9],
            inertia_origin: Origin::CenterOfGravity,
            inertia_point: Vec3::ZERO,
            stiffness: Vec::new(),
            damping: Vec::new(),
            stiffness_rows: 0,
            stiffness_cols: 0,
            damping_rows: 0,
            damping_cols: 0,
            stiffness_origin: Origin::RotationCenter,
            damping_origin: Origin::RotationCenter,
            stiffness_point: Vec3::ZERO,
            damping_point: Vec3::ZERO,
            excitation: Excitation::Pressure,
            interior_method: InteriorMethod::None,
            interior_panels: Vec::new(),
            interior_mesh: None,
            dipole_panels: Vec::new(),
            dipole_mesh: None,
            dampers: Vec::new(),
            generalized: Vec::new(),
            mode_mass: Vec::new(),
            mode_damping: Vec::new(),
            mode_stiffness: Vec::new(),
            tanks: Vec::new(),
            control_surface: None,
            control_surface_alternative: WaterlineTerm::ControlSurface,
            pressure: Vec::new(),
            plates: Vec::new(),
            body_points: Vec::new(),
            body_point_neighbors: 4,
            dipole_owners: Vec::new(),
            dipole_mesh_owner: Vec::new(),
        }
    }
}

impl Body {
    pub fn has_sheets(&self) -> bool {
        !self.dipole_panels.is_empty() || self.dipole_mesh.is_some() || !self.dampers.is_empty()
    }
    pub fn has_free_damper(&self) -> bool {
        for d in &self.dampers {
            if d.free() {
                return true;
            }
        }
        false
    }
}

#[derive(Clone, Debug)]
pub struct Lid {
    pub name: String,
    pub body: i32,
    pub panels: Vec<Range>,
    pub mesh: Option<usize>, // a mesh of its own (global coordinates); None: elements of its body's mesh
    pub damping_factor: f64,
}

impl Default for Lid {
    fn default() -> Self {
        Lid { name: String::new(), body: -1, panels: Vec::new(), mesh: None, damping_factor: 0.0 }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Environment {
    pub depth: f64,
    pub gravity: f64,
    pub density: f64,
}

impl Default for Environment {
    fn default() -> Self {
        Environment { depth: f64::INFINITY, gravity: STANDARD_GRAVITY, density: 1000.0 }
    }
}

/// The cancel request of a model (its CancelToken, model.rs): the token is shared with the solve
/// (the SolveContext, progress.rs), and a copy of a model starts uncancelled.
#[derive(Debug, Default)]
pub struct CancelFlag {
    token: CancelToken,
}

impl Clone for CancelFlag {
    fn clone(&self) -> Self {
        CancelFlag::default()
    }
}

impl CancelFlag {
    /// The flag of a cancel token.
    pub fn shared(token: CancelToken) -> CancelFlag {
        CancelFlag { token }
    }
    pub fn get(&self) -> bool {
        self.token.is_cancelled()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub diffraction: bool,
    pub radiation: bool,
    pub rao: bool,
    pub logarithmic: bool,
    pub higher_order: bool,
    pub formulation: Formulation,
    pub domain: Domain,
    pub channel_width: f64,
    pub channel_reflect: f64,                  // the factor of the wall images
    pub control_surface_drift: bool,           // the momentum flux through the control surfaces
    pub drift_angles: i32,                     // > 0: Kochin functions on this many angles (mean drift, theory §11.1)
    pub pressure_drift: bool,                  // mean drift by pressure integration (theory §11.2)
    pub pressure_qtf: QtfHeadings,             // its quadratic QTFs
    pub control_surface_qtf: bool,             // the control-surface QTFs, in the headings of pressure_qtf
    pub potential_qtf: QtfHeadings,            // the potential QTFs
    pub potential_qtf_radius: Option<f64>,     // the inner radius, about the centre of the bodies; None: auto
    pub potential_qtf_panel_size: Option<f64>, // the size of the generated panels (or the cells); None: auto
    pub potential_qtf_direct: bool,            // the direct method of the second-order potential
    pub potential_qtf_panels: bool, // the inner free surface by generated panels (one point each), not the cells
    // Newman's approximation (theory §12.13) from the drifts by pressure integration, control
    // surface, momentum conservation; the geometric mean (else the arithmetic one)
    pub newman: NewmanDrifts,
    pub newman_geometric: bool,
    // the zones beyond the inner free surface (theory §12.9): the quadrature annuli (0: none), their
    // radial step (None: auto), Gauss-Legendre nodes per annulus, azimuthal nodes a quadrant (None: auto);
    // the points of the outer circle's line integral (None: the fit's ring); the order of the
    // far-field series (None: auto)
    pub potential_qtf_annuli: i32,
    pub potential_qtf_annulus_step: Option<f64>,
    pub potential_qtf_radial_nodes: i32,
    pub potential_qtf_azimuthal_nodes: Option<i32>,
    pub potential_qtf_segments: Option<i32>,
    pub potential_qtf_order: Option<i32>,
    pub qtf_frequencies: QtfFrequencies,
    // the quantities computed for others but not asked for, dropped from the results at the end of
    // the solve (SolveJob::run): the pressure drift and the pressure QTFs the quadratic parts rest
    // on (the control surface's hull values), the QTFs of the bidirectional drifts, the QTFs of the
    // frequencies not asked for
    pub hidden: QuantitySet,
    // the quadratic QTFs at equal frequencies only (computed for the bidirectional drifts alone)
    pub qtf_diagonal: bool,
    pub drift_pairs: bool,                // the drifts of every heading pair (bidirectional)
    pub qtf_pairs: bool,                  // the QTFs of every heading pair (bidirectional)
    pub potential_qtf_both: bool,         // the potential QTFs by both methods (sharing the sub-problems)
    pub free_surface_mesh: Option<usize>, // the mesh of the potential QTF's free surface (second_order.free_surface_*); None: generated
    pub free_surface_points: i32,         // its Gauss points per panel side
    pub field_pressure: bool,             // elevation / pressure at the field points (theory §4.1)
    pub log_tolerance: f64,               // the bound of the log-integral correction (theory §6); 0: every pair
    pub field_velocity: bool,             // fluid velocity at the field points (theory §4.1)
    // internal (no model item): the field velocities only at the field points from this index on
    // (the potential QTF asks for them next to the waterlines only); 0: at every field point
    pub field_velocity_from: usize,
    pub body_pressure: bool, // at the body panels (theory §4.1)
    pub body_velocity: bool,
    pub frequency_threads: i32, // threads over the frequencies
    pub inner_threads: i32,     // threads inside each frequency: assembly, LU
    pub memory_check: bool,     // the memory estimate against the available memory before a solve
}

impl Default for Options {
    fn default() -> Self {
        Options {
            diffraction: true,
            radiation: true,
            rao: false,
            logarithmic: true,
            higher_order: false,
            formulation: Formulation::Scattered,
            domain: Domain::OpenSea,
            channel_width: f64::NAN,
            channel_reflect: 1.0,
            control_surface_drift: false,
            drift_angles: 0,
            pressure_drift: false,
            pressure_qtf: QtfHeadings::None,
            control_surface_qtf: false,
            potential_qtf: QtfHeadings::None,
            potential_qtf_radius: None,
            potential_qtf_panel_size: None,
            potential_qtf_direct: false,
            potential_qtf_panels: true,
            newman: NewmanDrifts::default(),
            newman_geometric: true,
            potential_qtf_annuli: 0,
            potential_qtf_annulus_step: None,
            potential_qtf_radial_nodes: 12,
            potential_qtf_azimuthal_nodes: None,
            potential_qtf_segments: None,
            potential_qtf_order: None,
            qtf_frequencies: QtfFrequencies::Both,
            hidden: QuantitySet::default(),
            qtf_diagonal: false,
            drift_pairs: false,
            qtf_pairs: false,
            potential_qtf_both: false,
            free_surface_mesh: None,
            free_surface_points: 1,
            field_pressure: false,
            log_tolerance: 1e-4,
            field_velocity: false,
            field_velocity_from: 0,
            body_pressure: false,
            body_velocity: false,
            frequency_threads: 1,
            inner_threads: 1,
            memory_check: true,
        }
    }
}

impl Options {
    /// the statics
    pub fn total_threads(&self) -> i32 {
        self.frequency_threads * self.inner_threads
    }
}

/// Role of an element of the mesh a body uses, after its mesh references.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Role {
    #[default]
    Wet,
    Interior,
    Dipole,
    Pressure,
    Lid,
    Tank,
    Plate,
}

#[derive(Clone, Debug)]
pub struct BodyDerived {
    pub role: Vec<Role>, // per element of the mesh the body uses
    pub owner: Vec<i32>, // pressure surface / lid / plate index per element, -1
    pub wet: i32,
    pub plate: i32, // elements of the free-surface plates (of the body mesh and of their own meshes)
    pub sheet: i32,
    pub interior: i32,
    pub symmetry: i32,
    pub pressure_modes: i32,
    pub generalized: i32,
    pub coordinate_offset: i32, // zero-based first global coordinate (6 rigid, G generalized, pressure)
    pub tank: Vec<i32>,         // per element of the mesh the body uses: its tank, else -1
    pub tank_z: Vec<f64>,       // per tank: the body-local z of its free surface
    pub wavemaker: bool,        // every element in a wall plane: a wavemaker in the wall
}

impl Default for BodyDerived {
    fn default() -> Self {
        BodyDerived {
            role: Vec::new(),
            owner: Vec::new(),
            wet: 0,
            plate: 0,
            sheet: 0,
            interior: 0,
            symmetry: 1,
            pressure_modes: 0,
            generalized: 0,
            coordinate_offset: 0,
            tank: Vec::new(),
            tank_z: Vec::new(),
            wavemaker: false,
        }
    }
}

/// Where the elements of a mesh of the problem come from: the number of stored elements of the mesh
/// object of the model it was made from, and per element (in the problem mesh's order) the
/// one-based stored element it comes from: an element cut by a trim the element it was cut from, 0
/// an element the engine generated (the interior surface inside the waterline of a trimmed patch
/// mesh); the elements of other meshes the higher order appended to a body's mesh have the stored
/// elements of their own mesh and are listed in `joined`. Of a trimmed mesh also per element the
/// copy (sx, sy) of the stored element it was cut from and its part (the order of the trim among
/// the pieces of that copy); empty: (1, 1) and 0. Of a generated element also the free-surface
/// plate generated with the trim it is part of (`plate`, its place in the body's plates; -1 or
/// empty: none, the interior surface), each generated surface numbered on its own.
#[derive(Clone, Debug, Default)]
pub struct MeshOrigin {
    pub stored: i32,
    pub records: Vec<i32>,
    pub copy: Vec<[i32; 2]>,
    pub part: Vec<i32>,
    pub joined: Vec<Joined>,
    pub plate: Vec<i32>,
}

/// Elements of another mesh appended to a body's patch mesh (the higher order): the zero-based
/// first element and the count in the body's mesh, the mesh of the problem they come from (kept, as
/// given), and the lid they are when it was given in global coordinates (else -1).
#[derive(Clone, Copy, Debug, Default)]
pub struct Joined {
    pub first: i32,
    pub count: i32,
    pub mesh: i32,
    pub lid: i32,
}

#[derive(Clone, Debug, Default)]
pub struct Problem {
    pub meshes: Vec<Mesh>,
    /// per mesh where its elements come from; a mesh without one (made by the engine, or beyond the
    /// list) has the stored elements of its own (stored_record)
    pub origins: Vec<MeshOrigin>,
    pub bodies: Vec<Body>,
    pub lids: Vec<Lid>,
    pub fixed_dampers: Vec<FixedDamper>, // the fixed dampers in no body's frame
    pub env: Environment,
    pub opt: Options,
    pub omegas: Vec<f64>,
    pub headings: Vec<f64>,
    pub has_field_points: bool,
    pub field_points: Vec<Vec3>,
    // per field point, or empty: 0 the exterior, k >= 1 the k-th tank of the bodies in order
    pub field_tanks: Vec<i32>,
    pub tank_domain: bool, // a tank problem: its field points lie inside its waterline
    // a tank problem: its fluid is bounded, so where its free surface is rigid (omega = 0) or
    // covered by plates a constant potential solves the homogeneous problem (theory §10.2)
    pub interior_domain: bool,
    pub cancel: CancelFlag,
    pub warnings: Vec<Diagnostic>, // of the last rule check
    // internal: the static rows of the field points shared by solves of one geometry and one set
    // of field points (the potential QTF's sub-problems): the first solve stores them
    pub field_statics: Option<SharedStatics>,
    // internal: the extra right-hand sides of the direct method of the second-order potential, per
    // ordinary frequency of this (radiation) sub-problem
    pub direct: Option<std::sync::Arc<DirectColumns>>,

    pub(crate) rules_ok_: bool,
    pub(crate) derived_: Vec<BodyDerived>,
}

/// Extra right-hand sides of one frequency of a radiation sub-problem (the direct method of the
/// second-order potential, theory §12.5): column c solves 2πφ − ∫φ ∂G/∂n = −∫G u + (1/g) E with
/// u the normal velocity on the body points (the wetted panels in their order, `u[c Nb + q]`) and
/// E_k = Σ_q [ws_q,c S_qk + wg_q,c · ∇_h S_qk] / A_k over the field points q (S_qk the source
/// integral over panel k at q, A_k its area; wg for the points with velocity rows, from
/// field_velocity_from on). A floating body adds −Σ_q bt_q,c · ∇S_kq-like terms of its Stokes
/// transform: −Σ_q bt_q,c · V_qk / A_k with V_qk the gradient of the source integral over panel k
/// at the body point q (global axes; empty for fixed bodies). The solution on the body points is
/// written to `out` [c Nb + q].
#[derive(Default)]
pub struct DirectFreq {
    pub ncols: usize,
    pub u: Vec<num_complex::Complex64>,
    pub ws: Vec<num_complex::Complex64>,      // [P][ncols]
    pub wg: Vec<[num_complex::Complex64; 2]>, // [P - field_velocity_from][ncols]
    pub bt: Vec<[num_complex::Complex64; 3]>, // [Nb][ncols], empty when no body moves
    pub out: std::sync::Mutex<Vec<num_complex::Complex64>>,
}

/// The extra right-hand sides of every ordinary frequency of a sub-problem.
#[derive(Default)]
pub struct DirectColumns {
    pub freqs: Vec<DirectFreq>,
}

impl std::fmt::Debug for DirectColumns {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DirectColumns({} frequencies)", self.freqs.len())
    }
}

/// The static rows of the field points of a solve, kept for the next solve of the same geometry and
/// field points (Static of the solve's real type, behind Any).
#[derive(Clone, Default)]
pub struct SharedStatics(pub std::sync::Arc<std::sync::OnceLock<std::sync::Arc<dyn std::any::Any + Send + Sync>>>);

impl std::fmt::Debug for SharedStatics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SharedStatics({})", if self.0.get().is_some() { "set" } else { "empty" })
    }
}

// ---------------------------------------------------------------- helpers

pub(crate) fn finite3(v: &Vec3) -> bool {
    v[0].is_finite() && v[1].is_finite() && v[2].is_finite()
}

fn selection_size(s: &[Range]) -> i32 {
    let mut n = 0;
    for r in s {
        n += r.last - r.first + 1;
    }
    n
}

pub(crate) fn copies(m: &Mesh) -> i32 {
    (if m.isx != 0 { 2 } else { 1 }) * (if m.isy != 0 { 2 } else { 1 })
}

pub(crate) fn plane_tol(m: &Mesh, s: &RecordShape) -> f64 {
    PLANE_TOL * max_of(m.reference_length, s.size)
}

// Expanded, placed hull points and normal of an element.
pub(crate) struct Placed {
    pub hull: Vec<Vec3>,
    pub normal: Vec3,
    pub tol: f64,
}

pub(crate) fn placed_records(m: &Mesh, records: &[i32], place: &Placement, refl: &[Reflection]) -> Vec<Placed> {
    let mut out = Vec::new();
    for r in refl {
        for &rec in records {
            let s = &m.shapes[rec as usize];
            let mut hull = Vec::with_capacity(s.hull.len());
            for h in &s.hull {
                hull.push(place.apply(v3(r.sx * h[0], r.sy * h[1], h[2])));
            }
            let normal = place.rotate(v3(r.sx * s.normal[0], r.sy * s.normal[1], s.normal[2]));
            out.push(Placed { hull, normal, tol: plane_tol(m, s) });
        }
    }
    out
}

pub(crate) fn all_records(m: &Mesh) -> Vec<i32> {
    (0..m.count).collect()
}

pub(crate) fn records_of(sel: &[Range]) -> Vec<i32> {
    let mut r = Vec::new();
    for x in sel {
        for i in x.first..=x.last {
            r.push(i - 1);
        }
    }
    r
}

pub(crate) fn placement_of(b: &Body) -> Placement {
    Placement { position: b.position, yaw_deg: b.yaw }
}

/// Problem::element_key of the origins of the meshes (the rules name an element by it while the
/// derived geometry is being made).
pub(crate) fn element_key_of(origins: &[MeshOrigin], mi: i32, rec: i32) -> (i32, i32, [i32; 2]) {
    let Some(o) = origins.get(mi as usize) else { return (rec + 1, 0, [1, 1]) };
    let r = rec as usize;
    let mut element = o.records[r];
    if element == 0 {
        let plate = |k: usize| o.plate.get(k).copied().unwrap_or(-1);
        element = 1 + (0..r).filter(|&k| o.records[k] == 0 && plate(k) == plate(r)).count() as i32;
    }
    (element, o.part.get(r).copied().unwrap_or(0), o.copy.get(r).copied().unwrap_or([1, 1]))
}

impl Problem {
    // ---------------------------------------------------------------- rules

    /// The model rules, then the solver-route list.
    /// Fails with ErrorKind::Unsupported when no route exists.
    pub fn validate(&mut self) -> Result<()> {
        self.check_rules()?;
        self.route()?;
        Ok(())
    }
    /// The route of the problem (validate() first).
    pub fn route(&mut self) -> Result<Route> {
        self.check_rules()?;
        select_route(self)
    }
    pub fn invalidate(&mut self) {
        self.rules_ok_ = false;
    }

    pub fn derived(&self) -> &Vec<BodyDerived> {
        &self.derived_
    }

    /// The one-based stored element of the zero-based element rec of mesh mi (MeshOrigin; 0: an element
    /// the engine generated).
    pub fn stored_record(&self, mi: i32, rec: i32) -> i32 {
        match self.origins.get(mi as usize) {
            Some(o) => o.records[rec as usize],
            None => rec + 1,
        }
    }
    /// The key of the zero-based element rec of mesh mi in the queries: its one-based element (the
    /// stored element; an element the engine generated numbered from 1 among the generated ones of
    /// its surface: the interior surface, or one generated plate), its part and the copy (sx, sy)
    /// of the stored element it comes from.
    pub fn element_key(&self, mi: i32, rec: i32) -> (i32, i32, [i32; 2]) {
        element_key_of(&self.origins, mi, rec)
    }
    /// The number of stored elements of the mesh object mesh mi was made from.
    pub fn stored_records(&self, mi: i32) -> i32 {
        self.origins.get(mi as usize).map_or(self.meshes[mi as usize].count, |o| o.stored)
    }
    /// The zero-based elements of mesh mi that come from the one-based stored element s (of its own
    /// mesh object, not of a joined mesh).
    pub fn records_of_stored(&self, mi: i32, s: i32) -> Vec<i32> {
        let n = self.meshes[mi as usize].count;
        (0..n).filter(|&r| self.joined_at(mi, r).is_none() && self.stored_record(mi, r) == s).collect()
    }
    /// The block of another mesh that the zero-based element rec of mesh mi was appended from.
    pub fn joined_at(&self, mi: i32, rec: i32) -> Option<&Joined> {
        self.origins.get(mi as usize)?.joined.iter().find(|j| rec >= j.first && rec < j.first + j.count)
    }
    /// The meshes appended to mesh mi (none for a mesh that was not merged).
    pub fn joined(&self, mi: i32) -> &[Joined] {
        self.origins.get(mi as usize).map_or(&[], |o| &o.joined)
    }

    // ---------------------------------------------------------------- counts

    pub fn coordinates(&self) -> i32 {
        let mut d = 0;
        for b in self.bodies.iter() {
            d += 6 + b.generalized.len() as i32;
            for p in &b.pressure {
                d += p.modes.len() as i32;
            }
        }
        d
    }
    /// The zero-based first coordinate of every body (its surge), the bodies in order: each body
    /// has its six rigid coordinates, then those of its generalized modes and of its pressure modes.
    pub fn first_coordinates(&self) -> Vec<usize> {
        self.bodies
            .iter()
            .scan(0usize, |off, b| {
                let first = *off;
                *off += 6 + b.generalized.len() + b.pressure.iter().map(|p| p.modes.len()).sum::<usize>();
                Some(first)
            })
            .collect()
    }
    /// The zero-based coordinates of the six rigid modes of every body, the bodies in order (6 B).
    pub fn rigid_coordinates(&self) -> Vec<usize> {
        self.first_coordinates().into_iter().flat_map(|f| f..f + 6).collect()
    }
    /// The zero-based coordinate of every generalized mode, the bodies in order.
    pub fn generalized_coordinates(&self) -> Vec<usize> {
        self.first_coordinates()
            .into_iter()
            .zip(&self.bodies)
            .flat_map(|(f, b)| (0..b.generalized.len()).map(move |k| f + 6 + k))
            .collect()
    }
    pub fn radiation_columns(&self) -> i32 {
        self.radiation_mode_ids().len() as i32
    }
    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    pub fn radiation_mode_ids(&self) -> Vec<i32> {
        let mut ids = Vec::new();
        if !self.opt.radiation {
            return ids;
        }
        let mut offset = 0;
        for b in self.bodies.iter() {
            for &j in &b.modes {
                ids.push(offset + j);
            }
            // the generalized modes are always free, then the pressure modes
            let G = b.generalized.len() as i32;
            for k in 0..G {
                ids.push(offset + 6 + 1 + k);
            }
            let mut pm = 0;
            for p in &b.pressure {
                for _k in 0..p.modes.len() {
                    ids.push(offset + 6 + G + 1 + pm);
                    pm += 1;
                }
            }
            offset += 6 + G + pm;
        }
        ids
    }
    pub fn lid_panel_counts(&self) -> Vec<i32> {
        let mut n = Vec::new();
        for l in &self.lids {
            if l.body >= 0 {
                n.push(selection_size(&l.panels) * copies(&self.meshes[self.bodies[l.body as usize].mesh as usize]));
            } else {
                let lm = &self.meshes[l.mesh.expect("a lid of its own has its mesh (rule_lids)")];
                n.push(lm.count * copies(lm));
            }
        }
        n
    }
    pub fn pressure_panels(&self) -> i32 {
        let mut q = 0;
        for b in &self.bodies {
            for p in &b.pressure {
                if let Some(pi) = p.mesh {
                    let pm = &self.meshes[pi];
                    q += pm.count * copies(pm);
                } else {
                    q += selection_size(&p.panels) * copies(&self.meshes[b.mesh as usize]);
                }
            }
        }
        q
    }
    pub fn ordinary_frequencies(&self) -> i32 {
        let mut n = 0;
        for &w in &self.omegas {
            if w > 0.0 && w.is_finite() {
                n += 1;
            }
        }
        n
    }
    pub fn has_zero(&self) -> bool {
        self.omegas.contains(&0.0)
    }
    pub fn has_infinite(&self) -> bool {
        self.omegas.iter().any(|w| w.is_infinite())
    }
    /// Radiation and diffraction off: the hydrostatics alone, no frequency work.
    pub fn hydrostatics_only(&self) -> bool {
        !self.opt.radiation && !self.opt.diffraction
    }
}

/// The path of the body in the model.
pub fn body_label(b: &Body) -> String {
    format!("bodies/{}", b.name)
}

/// The zero-based place of an object in the messages: its place in the model's list (a copy
/// without the disabled objects keeps it), or its index k in the problem without one.
pub fn model_place(place: i32, k: usize) -> usize {
    usize::try_from(place).unwrap_or(k)
}

/// An index of the engine's lists given as an i32 with -1 for none (the build's own counters):
/// None for a negative one.
pub(crate) fn some_index(i: i32) -> Option<usize> {
    usize::try_from(i).ok()
}
