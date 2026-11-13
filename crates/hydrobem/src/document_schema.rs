//! The schema of the model: every object type with its items and lists, the model
//! first; TypeDef::item and TypeDef::list. The object types, their items (in the field order of
//! reference/inputs-*), their kinds, units, defaults and checks.

use std::sync::LazyLock;

use crate::document::{DType, ItemDef, ItemKind, ListDef, TypeDef};
use crate::document_rules::ItemRule;

use DType as T;
use ItemKind as K;

type Def = Option<&'static str>;

fn item(n: &'static str, kind: ItemKind) -> ItemDef {
    ItemDef { name: n, kind, ..ItemDef::default() }
}

fn b(n: &'static str, def: impl Into<Def>, desc: &'static str) -> ItemDef {
    let mut d = item(n, K::Bool);
    d.fallback = def.into();
    d.description = desc;
    d
}

fn i(n: &'static str, def: impl Into<Def>, rule: ItemRule, lo: f64, hi: f64, desc: &'static str) -> ItemDef {
    let mut d = item(n, K::Int);
    d.fallback = def.into();
    d.rule = rule;
    d.lo = lo;
    d.hi = hi;
    d.description = desc;
    d
}

fn r(n: &'static str, unit: &'static str, def: impl Into<Def>, rule: ItemRule, desc: &'static str) -> ItemDef {
    let mut d = item(n, K::Real);
    d.unit = unit;
    d.fallback = def.into();
    d.rule = rule;
    d.description = desc;
    d
}

// a number or one of the names of choices
fn either(mut d: ItemDef, choices: Vec<&'static str>) -> ItemDef {
    d.choices = choices;
    d
}

fn c(n: &'static str, choices: Vec<&'static str>, def: impl Into<Def>, desc: &'static str) -> ItemDef {
    let mut d = item(n, K::Choice);
    d.choices = choices;
    d.fallback = def.into();
    d.description = desc;
    d
}

fn a(
    n: &'static str,
    t: DType,
    shape: &'static str,
    unit: &'static str,
    def: impl Into<Def>,
    rule: ItemRule,
    desc: &'static str,
) -> ItemDef {
    let mut d = item(n, K::Array);
    d.dtype = t;
    d.shape = shape;
    d.unit = unit;
    d.fallback = def.into();
    d.rule = rule;
    d.description = desc;
    d
}

fn ref_(n: &'static str, target: &'static str, desc: &'static str) -> ItemDef {
    let mut d = item(n, K::Reference);
    d.target = target;
    d.description = desc;
    d
}

// the item enabled of an object that can be suppressed: kept in the model, skipped by the solve
fn enabled(what: &'static str) -> ItemDef {
    let text: &'static str = Box::leak(
        format!(
            "Whether the solve takes {what}; false: kept in the model with all its items, but solved as if it were absent."
        )
        .into_boxed_str(),
    );
    b("enabled", "true", text)
}

// an item of a surface taken by some of its methods: the values of its switch (the method, set by
// `switches` from the item's prefix) that take it ("automatic/circle": the method automatic with
// control_surface_outer "circle")
fn on(mut d: ItemDef, values: &'static str) -> ItemDef {
    d.switch_values = values;
    d
}

// the mesh of a surface: the name of a mesh of the model
fn mesh_of(desc: &'static str) -> ItemDef {
    ref_("mesh", "meshes", desc)
}

// the elements of a surface: ranges of the stored elements of its mesh (None: every element)
fn ranges(desc: &'static str) -> ItemDef {
    a("ranges", T::Int, "N,2", "", None, ItemRule::Ranges, desc)
}

const RANGES: &str = "The elements of the mesh: [(first, last), ...], one-based ranges of its stored elements, first <= last, without overlaps; None: every element.";

fn origin(n: &'static str, def: impl Into<Def>, desc: &'static str) -> ItemDef {
    let mut d = item(n, K::Origin);
    d.choices = vec!["center_of_gravity", "rotation_center"];
    d.shape = "3";
    d.unit = "m";
    d.fallback = def.into();
    d.rule = ItemRule::Finite;
    d.description = desc;
    d
}

fn name(def: impl Into<Def>, desc: &'static str) -> ItemDef {
    let mut d = item("name", K::Text);
    d.fallback = def.into();
    d.rule = ItemRule::Name;
    d.description = desc;
    d
}

fn terms(n: &'static str, desc: &'static str) -> ItemDef {
    let mut d = item(n, K::Terms);
    d.shape = "N,4";
    d.unit = "m";
    d.fallback = Some("[]");
    d.rule = ItemRule::Terms;
    d.description = desc;
    d
}

// a plain text item (label, source_file, source_spl, Tool.text)
fn txt(n: &'static str, def: &'static str, desc: &'static str) -> ItemDef {
    let mut d = item(n, K::Text);
    d.fallback = Some(def);
    d.description = desc;
    d
}

fn mesh_item(mut d: ItemDef, kinds: &'static str) -> ItemDef {
    d.kinds = kinds;
    d
}

fn ld(n: &'static str, types: Vec<&'static str>, named: bool, single: bool, always: bool) -> ListDef {
    ListDef { name: n, types, named, optional_names: false, single, always, description: "" }
}

// a list whose objects may have names (unique within the list), or none
fn lo(n: &'static str, types: Vec<&'static str>) -> ListDef {
    ListDef { optional_names: true, ..ld(n, types, false, false, false) }
}

fn ty(n: &'static str, items: Vec<ItemDef>, lists: Vec<ListDef>) -> TypeDef {
    TypeDef { name: n, items, lists }
}

const MODES: [&str; 6] = ["surge", "sway", "heave", "roll", "pitch", "yaw"];
const ALL_MODES: &str = "[\"surge\", \"sway\", \"heave\", \"roll\", \"pitch\", \"yaw\"]";

#[allow(clippy::vec_init_then_push)] // one statement per type of the schema, with its comments
fn build() -> Vec<TypeDef> {
    let mut t: Vec<TypeDef> = Vec::new();
    t.push(ty(
        "Model",
        vec![],
        vec![
            ld("environment", vec!["Environment"], false, true, true),
            ld("options", vec!["Options"], false, true, true),
            ld("first_order", vec!["FirstOrder"], false, true, true),
            ld("second_order", vec!["SecondOrder"], false, true, true),
            ld("bodies", vec!["Body"], true, false, false),
            ld("lids", vec!["DampingLid"], true, false, false),
            lo("fixed_dampers", vec!["FixedDamper"]),
            ld("meshes", vec!["Mesh"], true, false, false),
            ld("tool", vec!["Tool"], true, false, false),
        ],
    ));
    t.push(ty(
        "Environment",
        vec![
            r(
                "depth",
                "m",
                "inf",
                ItemRule::Depth,
                "The water depth h, > 0; inf for infinite depth. The sea bed is z = -h.",
            ),
            r("gravity", "m/s^2", "9.80665", ItemRule::Positive, "The acceleration of gravity g."),
            r("density", "kg/m^3", "1000.0", ItemRule::Positive, "The water density rho."),
            c("domain", vec!["open_sea", "x_wall", "y_wall", "orthogonal_walls", "channel"], "\"open_sea\"", "The walls of the fluid domain: none, a wall at x = 0, a wall at y = 0, both walls, or a channel with walls at y = +-channel_width / 2 (finite depth)."),
            r("channel_width", "m", None, ItemRule::Positive, "The width W of the channel, > 0; for the domain channel only."),
            r("channel_reflect", "", "1.0", ItemRule::Unit, "The factor R in 0..1 of every wall image of the channel: 1 reflecting walls, 0 the open sea of the same depth."),
            c("frequency_input", vec!["omegas", "periods"], "\"omegas\"", "Which of omegas and periods gives the frequencies of the solve; the other is kept unused."),
            a("omegas", T::Real, "F", "rad/s", "[]", ItemRule::Omegas, "The radian frequencies of the solve (frequency_input omegas), >= 0. 0 and inf request the limits of the added mass (zero and infinite frequency)."),
            a("periods", T::Real, "F", "s", "[]", ItemRule::Omegas, "The wave periods of the solve (frequency_input periods), >= 0; a period of 0 is omega = inf."),
            a("headings", T::Real, "H", "deg", "[]", ItemRule::Finite, "The directions of propagation of the incident waves, counter-clockwise from +x (0: waves travelling towards +x). Needed by the diffraction problem only."),
        ],
        vec![],
    ));
    t.push(ty(
        "Options",
        vec![
            b("higher_order", "false", "The higher-order (B-spline) panel method; every body then has a patch mesh."),
            b("logarithmic_integration", "true", "Integrate the logarithmic singularity of the wave Green function near the free surface analytically (the pairs of log_tolerance in the low order); false: plain quadrature of the wave kernel."),
            r("log_tolerance", "", "0.0001", ItemRule::Nonnegative, "Low order: the analytic log integral of a panel pair is applied where its correction may exceed this fraction of the panel area; 0: every pair."),
            i("frequency_threads", "1", ItemRule::IntRange, 1.0, 1024.0, "Engine threads over the frequencies; each holds the matrices of one frequency. The results do not depend on it."),
            i("inner_threads", "1", ItemRule::IntRange, 1.0, 1024.0, "Engine threads inside each frequency (assembly, LU); the statics use frequency_threads x inner_threads. The results do not depend on it."),
            b("memory_check", "true", "Before the solve, compare the resident peak of the memory estimate with the physical memory available and refuse a solve that needs more (a value error giving both and the remedies); false: solve without the check."),
        ],
        vec![
        ],
    ));
    t.push(ty(
        "FirstOrder",
        vec![
            b("radiation", "true", "Solve the radiation problems of the modes of the bodies (and of their generalized and pressure modes): the added mass and damping."),
            b("diffraction", "true", "Solve the diffraction problem of every heading: the exciting forces, the diffracted fields."),
            c("diffraction_formulation", vec!["auto", "scattered", "total"], "\"auto\"", "The unknown of the diffraction problem: the scattered or the total potential (both give the same forces; higher-order lids need total, low-order thin sheets scattered); auto: total when the model has a lid or a free-surface plate (scattered with low-order thin sheets), scattered otherwise."),
            b("motion", "false", "Solve the motions of the bodies (the RAOs); needs radiation and diffraction, and the mass and inertia of every moving body."),
            b("field_pressure", "false", "At the field points: the free-surface elevation (z = 0) and the pressure p / (rho g) below it."),
            b("field_velocity", "false", "The fluid velocity at the field points."),
            b("body_pressure", "false", "The pressure p / (rho g) on the bodies: at the panels or patch points, or at the body_points of the bodies."),
            b("body_velocity", "false", "The fluid velocity on the bodies, at the same points as body_pressure."),
            a("field_points", T::Real, "N,3", "m", None, ItemRule::Points, "The points (N, 3) of the field outputs in the sea, global coordinates (x, y, z), at z = 0 (the free-surface elevation) or below, in the fluid; the points inside a tank are the tank's field_points. Used when field_pressure or field_velocity is on."),
        ],
        vec![],
    ));
    t.push(ty(
        "SecondOrder",
        vec![
            b("drift_pressure_integration", "false", "The mean drift force and moment by pressure integration over every body's surface (six components, body axes, the moment about the rotation centre)."),
            b("drift_control_surface", "false", "The mean drift force and moment by the momentum flux through every body's control_surface (six components, body axes, the moment about the rotation centre); every body needs its control surface."),
            b("drift_momentum_conservation", "false", "The mean drift by momentum conservation in the far field: the surge, sway force and yaw moment (about the global z axis) of all bodies, global axes."),
            b("drift_bidirectional", "false", "The heading pairs of the drifts: false the equal headings, true every pair of headings (the coefficients of two waves of the same frequency and different headings)."),
            either(i("drift_momentum_conservation_angles", "\"auto\"", ItemRule::IntRange, 1.0, 65536.0, "The number of Kochin angles of the drift by momentum conservation, 1..65536, or auto: chosen by the engine from the wave number and the size of the bodies."), vec!["auto"]),
            b("qtf_quadratic_pressure_integration", "false", "The quadratic part of the second-order force and moment of every body (QTFs, theory §12.2; the products of the first-order solution; six components, body axes, the moment about the rotation centre) by pressure integration over the body."),
            b("qtf_quadratic_control_surface", "false", "The quadratic part of the QTFs over the control surface of every body (the momentum flux through it, the time derivatives over the free surface inside it and over the hull)."),
            b("qtf_potential_direct", "false", "The force of the second-order potential of every body (QTFs) by the direct method: the second-order potential solved at the pair frequencies. Low order, bodies fixed or floating (with motion), also with generalized modes, in the open sea or by one wall (x_wall, y_wall); not with pressure surfaces, tanks, thin sheets, dipole dampers, damping lids or fixed dampers."),
            b("qtf_potential_indirect", "false", "The force of the second-order potential by the indirect method: the radiation potentials at the pair frequencies and the free-surface forcing; available where qtf_potential_direct is."),
            b("qtf_newman", "false", "Newman's approximation of the difference-frequency QTF (theory §12.13) from every drift switched on (drift_pressure_integration, drift_control_surface, drift_momentum_conservation): the drift on the diagonal, the mean of the two drifts off it, the equal headings; needs at least one drift."),
            c("qtf_newman_average", vec!["geometric", "arithmetic"], "\"geometric\"", "The mean of Newman's approximation: geometric, sign(T_ii) sqrt(T_ii T_jj) where the two drifts have the same sign and 0 where not (Newman 1974), or arithmetic, (T_ii + T_jj)/2."),
            b("qtf_bidirectional", "false", "The heading pairs of the QTFs: false the equal headings, true every pair of headings."),
            b("qtf_difference", "true", "The QTFs at the difference frequencies (the slow drift)."),
            b("qtf_sum", "true", "The QTFs at the sum frequencies (without them the potential part needs no radiation problems at the sum frequencies)."),
            // the free surface of the potential QTF: its panelled zone, quadrature zone, outer circle
            // and far field
            c("free_surface_method", vec!["panels", "cells", "mesh"], "\"panels\"", "The panelled zone of the potential QTF's free surface, from the bodies' waterlines to the inner radius: panels the engine generates, of about free_surface_panel_size, each integrated by its centroid (a square grid about the waterlines with triangles to them, spokes and rings beyond; theory §12.8); cells of that size with 4 x 4 Gauss points each; or mesh, a given mesh of panels in z = 0 (free_surface_mesh, free_surface_ranges, free_surface_quadrature_points)."),
            on(either(r("free_surface_panel_size", "m", "\"auto\"", ItemRule::Positive, "(panels, cells) The size of the generated panels or cells. auto: a third of the shortest combined wavelength 2 pi/(2 k + k_Omega) of the frequencies, at most a quarter of the largest distance a from the centre of the body origins to a waterline."), vec!["auto"]), "panels cells"),
            on(either(r("free_surface_inner_radius", "m", "\"auto\"", ItemRule::Positive, "(panels, cells) The radius R of the generated panelled zone about the centre of the body origins. auto: 4 a."), vec!["auto"]), "panels cells"),
            on(ref_("free_surface_mesh", "meshes", "(mesh) The given panelled zone: the name of a panels mesh in z = 0, global coordinates, from the bodies' waterlines to the circle about the global origin through its farthest vertex (the inner radius); any panels mesh (a GDF or FDF file's panels, panels made in Python)."), "mesh"),
            on(a("free_surface_ranges", T::Int, "N,2", "", None, ItemRule::Ranges, "(mesh) The elements of free_surface_mesh: [(first, last), ...], one-based ranges of its stored elements, first <= last, without overlaps; unset: every element."), "mesh"),
            on(i("free_surface_quadrature_points", "1", ItemRule::IntRange, 1.0, 8.0, "(mesh) The Gauss points per side of each panel of the given mesh (n x n on its bilinear map); 1: its centroid with its area."), "mesh"),
            i("quadrature_annuli", "0", ItemRule::IntRange, 0.0, 1024.0, "The annuli of the quadrature zone between the inner radius and the outer circle (theory §12.9); 0: none (the outer circle at the inner radius)."),
            either(r("quadrature_radius_step", "m", "\"auto\"", ItemRule::Positive, "The radial width of each annulus. auto: twice the shortest wavelength of the frequencies and the pair frequencies."), vec!["auto"]),
            i("quadrature_radial_nodes", "12", ItemRule::IntRange, 1.0, 64.0, "The Gauss-Legendre nodes across each annulus."),
            either(i("quadrature_azimuthal_nodes", "\"auto\"", ItemRule::IntRange, 1.0, 65536.0, "The equally spaced nodes at each radius of the annuli: on the circle, or a half circle or a quadrant where the whole model has one or two symmetry planes. auto: from the outer radius and the shortest combined wavelength."), vec!["auto"]),
            either(i("outer_circle_segments", "\"auto\"", ItemRule::IntRange, 1.0, 65536.0, "The equal segments of the line integral around the outer circle (a point at the middle of each), counted as the azimuthal nodes. auto: the 4 M + 8 points of the far-field fit."), vec!["auto"]),
            either(i("expansion_order", "\"auto\"", ItemRule::IntRange, 1.0, 160.0, "The highest angular order M of the far-field series beyond the outer circle (of the disturbances, of the Green function or radiation potentials at the pair frequency, and of the incident waves, all truncated at M). auto: k R + 15 (k the largest wavenumber, R the outer radius), the incident waves to 2 M + 1."), vec!["auto"]),
        ],
        vec![],
    ));
    t.push(ty(
        "Body",
        vec![
            mesh_of("The wetted surface: the name of a mesh of the model, body coordinates (with ranges its elements). A low-order mesh, or a patch mesh with options.higher_order; with trim the hull, cut at the free surface by the engine. Unset: a body of dipoles or of free-surface plates only."),
            ranges(RANGES),
            name("\"body\"", "The unique name of the body."),
            a("position", T::Real, "3", "m", "[0.0, 0.0, 0.0]", ItemRule::Finite, "The global position of the body origin."),
            r("yaw", "deg", "0.0", ItemRule::Finite, "The rotation of the body about the global z axis, applied before the translation."),
            a("trim", T::Real, "3", "", None, ItemRule::Finite, "The elevation (m), pitch and roll (deg) of the hull in body coordinates: the mesh is moved to x = R(pitch, roll) xi + (0, 0, elevation) and every surface of the body cut at the free surface when the problem is built (tanks at their own surface level); unset: the meshes are the wetted surfaces as given."),
            a("rotation_center", T::Real, "3", "m", "[0.0, 0.0, 0.0]", ItemRule::Finite, "The motion reference point R, body-local: the first-order results (forces, matrices, motions) are about R."),
            a("modes", T::Text, "N", "", ALL_MODES, ItemRule::Modes, "The rigid modes of the body, in order (surge, sway, heave, roll, pitch, yaw): its radiation columns and free motions; [] a fixed body that still interacts."),
            a("center_of_gravity", T::Real, "3", "m", None, ItemRule::Finite, "The centre of gravity G, body-local; needed for the motions and the weight."),
            either(r("mass", "kg", None, ItemRule::Nonnegative, "The mass of the body (0 a massless body, such as a light piston or lid, whose inertia is its added mass), or displaced: the density times the displaced volume, less the fluid of its tanks. Needed with center_of_gravity for the weight, and for the motions."), vec!["displaced"]),
            c("inertia_input", vec!["components", "radii_of_gyration"], "\"radii_of_gyration\"", "Which inertia is used: components (Ixx ... Izz) or radii_of_gyration; the other is kept unused."),
            r("Ixx", "kg m^2", "0.0", ItemRule::Float32, "The inertia component Ixx about inertia_origin, body axes (read with inertia_input components)."),
            r("Ixy", "kg m^2", "0.0", ItemRule::Float32, "The inertia component Ixy about inertia_origin, body axes, without a sign change."),
            r("Ixz", "kg m^2", "0.0", ItemRule::Float32, "The inertia component Ixz about inertia_origin, body axes, without a sign change."),
            r("Iyy", "kg m^2", "0.0", ItemRule::Float32, "The inertia component Iyy about inertia_origin, body axes."),
            r("Iyz", "kg m^2", "0.0", ItemRule::Float32, "The inertia component Iyz about inertia_origin, body axes, without a sign change."),
            r("Izz", "kg m^2", "0.0", ItemRule::Float32, "The inertia component Izz about inertia_origin, body axes."),
            a("radii_of_gyration", T::Real, "3|3,3", "m", "[0.0, 0.0, 0.0]", ItemRule::Gyration, "The radii of gyration about inertia_origin (inertia_input radii_of_gyration), three values (the diagonal) or a symmetric 3 x 3 matrix; I = m r |r| with the mass in use."),
            origin("inertia_origin", "\"center_of_gravity\"", "The point of the inertia: center_of_gravity, rotation_center or a body-local point."),
            a("additional_stiffness", T::Real, "6,6", "SI", None, ItemRule::Float32, "An external 6 x 6 stiffness matrix of the body's rigid modes, about stiffness_origin; the coupling to another body is a coupling."),
            a("additional_damping", T::Real, "6,6", "SI", None, ItemRule::Float32, "An external 6 x 6 damping matrix of the body's rigid modes, about damping_origin; the coupling to another body is a coupling."),
            origin("stiffness_origin", "\"rotation_center\"", "The point of additional_stiffness, as inertia_origin."),
            origin("damping_origin", "\"rotation_center\"", "The point of additional_damping, as inertia_origin."),
            c("excitation_method", vec!["diffraction", "haskind"], "\"diffraction\"", "The exciting force in the body's motion equation: by pressure integration of the diffraction potential (excitation_diffraction) or by the Haskind relation (excitation_haskind)."),
            a("mode_mass", T::Real, "M,M", "SI", None, ItemRule::Finite, "The (6 + G) x (6 + G) mass matrix of the rigid (about R) and generalized coordinates; the rigid block zero."),
            a("mode_damping", T::Real, "M,M", "SI", None, ItemRule::Finite, "The (6 + G) x (6 + G) damping matrix of the rigid and generalized coordinates; the rigid block zero."),
            a("mode_stiffness", T::Real, "M,M", "SI", None, ItemRule::Finite, "The (6 + G) x (6 + G) stiffness matrix of the rigid and generalized coordinates; the rigid block zero."),
            a("body_points", T::Real, "N,3", "m", None, ItemRule::Points, "Body-local points (N, 3) where the body outputs are given instead of the panels; for every body or none."),
            i("body_point_neighbors", "4", ItemRule::IntRange, 1.0, 2147483647.0, "Low order with body_points: the number of nearest panel centroids averaged by inverse distance at each point."),
            // the interior surface (irregular frequencies) and the control surface (control-surface
            // drift): a method each, and the items of the methods, prefixed by its name
            c("interior_surface", vec!["none", "detect", "waterline", "graded", "projected", "mesh"], "\"detect\"", "The interior surface inside the waterline, which removes the irregular frequencies: none; detect, the elements of the wetted surface lying in z = 0 (none if there are none); a surface the engine generates: waterline (the regions inside the waterline triangulated; higher order: of a trimmed body), graded (a fan from the centroid of each region, graded towards the waterline) or projected (the downward-facing wetted elements projected to z = 0); or mesh, a given surface (interior_surface_mesh, interior_surface_ranges)."),
            on(ref_("interior_surface_mesh", "meshes", "(mesh) The given interior surface: the name of a mesh, body coordinates, at z = 0 with upward normals."), "mesh"),
            on(a("interior_surface_ranges", T::Int, "N,2", "", None, ItemRule::Ranges, "(mesh) The elements of interior_surface_mesh: [(first, last), ...], one-based ranges of its stored elements, first <= last, without overlaps; unset: every element."), "mesh"),
            c("control_surface", vec!["none", "mesh", "cylinder", "ellipsoid", "automatic"], "\"none\"", "The control surface of the control-surface drift (needed by second_order.drift_control_surface): none; mesh, a given surface (control_surface_mesh, control_surface_ranges); or one the engine generates when the problem is built, each with control_surface_panel_size: cylinder (radius, depth, inner_radius), ellipsoid (semi_axes, inner_semi_axes) or automatic around the body's waterline with its free-surface part (depth, partitions, outer with radius or boundary)."),
            on(ref_("control_surface_mesh", "meshes", "(mesh) The given control surface: the name of a mesh, body coordinates, around the body, normals away from it."), "mesh"),
            on(a("control_surface_ranges", T::Int, "N,2", "", None, ItemRule::Ranges, "(mesh) The elements of control_surface_mesh: [(first, last), ...], one-based ranges of its stored elements, first <= last, without overlaps; unset: every element."), "mesh"),
            on(c("control_surface_waterline_term", vec!["control_surface", "body"], "\"control_surface\"", "(every method) Where the control-surface drift takes the zeta^2 term of the waterline: control_surface on the waterline of the control surface; body on the body waterline with a free-surface integral (a control surface close to the body)."), "mesh cylinder ellipsoid automatic"),
            on(r("control_surface_radius", "m", None, ItemRule::Positive, "(cylinder) The radius of the cylinder; (automatic with outer circle) the radius of the circular outer surface about the body origin."), "cylinder automatic/circle"),
            on(r("control_surface_depth", "m", None, ItemRule::Positive, "(cylinder, automatic) The depth of the surface below the free surface."), "cylinder automatic"),
            on(r("control_surface_inner_radius", "m", None, ItemRule::Positive, "(cylinder) The radius of the circular waterline inside, where the free surface between it and the cylinder begins."), "cylinder"),
            on(a("control_surface_semi_axes", T::Real, "3", "m", None, ItemRule::Positive, "(ellipsoid) The semi-axes a, b, c of the lower half-ellipsoid."), "ellipsoid"),
            on(a("control_surface_inner_semi_axes", T::Real, "2", "m", None, ItemRule::Positive, "(ellipsoid) The semi-axes of the elliptic waterline inside, where the free surface between it and the ellipsoid begins."), "ellipsoid"),
            on(c("control_surface_outer", vec!["circle", "polygon"], "\"circle\"", "(automatic) The outer surface: a circle (control_surface_radius) or a polygon (control_surface_boundary)."), "automatic"),
            on(a("control_surface_boundary", T::Real, "N,2", "m", None, ItemRule::Finite, "(automatic with outer polygon) The (x, y) vertices of the outer polygon in the stored sector, counter-clockwise about the body origin, body coordinates."), "automatic/polygon"),
            on(a("control_surface_partitions", T::Real, "N,2", "m", None, ItemRule::Real, "(automatic) Further (x, y) polylines of the free surface, one after the other, each ended by a row of NaN (the last may be left open); unset: none."), "automatic"),
            on(r("control_surface_panel_size", "m", None, ItemRule::Positive, "(cylinder, ellipsoid, automatic) The panel size of the generated surface."), "cylinder ellipsoid automatic"),
            enabled("the body with everything under it"),
        ],
        vec![
            lo("dipoles", vec!["Dipole"]),
            lo("dipole_dampers", vec!["DipoleDamper"]),
            lo("tanks", vec!["Tank"]),
            lo("pressure_surfaces", vec!["PressureSurface"]),
            lo("free_surface_plates", vec!["FreeSurfacePlate"]),
            ld("generalized_modes", vec!["GeneralizedMode"], true, false, false),
            lo("couplings", vec!["Coupling"]),
        ],
    ));
    t.push(ty(
        "Coupling",
        vec![
            ref_("body", "bodies", "The other body, by name (required; not the body itself): the columns are its six rigid motions."),
            a("stiffness", T::Real, "6,6", "SI", None, ItemRule::Float32, "The 6 x 6 stiffness between the bodies: the rows the six forces of this body about its stiffness_origin, the columns the six motions of the other body about that body's stiffness_origin."),
            a("damping", T::Real, "6,6", "SI", None, ItemRule::Float32, "The 6 x 6 damping between the bodies: the rows the six forces of this body about its damping_origin, the columns the six motions of the other body about that body's damping_origin."),
            enabled("the coupling (one that names a disabled body is skipped as well)"),
        ],
        vec![],
    ));
    t.push(ty(
        "Dipole",
        vec![
            mesh_of("The sheet: the name of a mesh, body coordinates; zero-thickness elements moving with the body (both faces at its normal velocity)."),
            ranges(RANGES),
            enabled("the dipole"),
        ],
        vec![],
    ));
    t.push(ty(
        "DipoleDamper",
        vec![
            mesh_of("The sheet: the name of the body's mesh, body coordinates; porous, moving with the body, in its forces."),
            ranges(RANGES),
            r(
                "damping_factor",
                "m/s",
                "0.0",
                ItemRule::Nonnegative,
                "The damping factor epsilon >= 0 of the jump -(i epsilon / omega) psi_n. epsilon = 0: no jump, as no sheet; epsilon -> infinity: the rigid sheet (a dipole).",
            ),
            enabled("the damper"),
        ],
        vec![],
    ));
    t.push(ty(
        "FixedDamper",
        vec![
            mesh_of("The sheet: the name of a mesh, global coordinates; porous, at rest, in no force."),
            ranges(RANGES),
            r(
                "damping_factor",
                "m/s",
                "0.0",
                ItemRule::Nonnegative,
                "The damping factor epsilon >= 0 of the jump -(i epsilon / omega) psi_n. epsilon = 0: no jump, as no sheet; epsilon -> infinity: the rigid sheet (a dipole).",
            ),
            enabled("the damper"),
        ],
        vec![],
    ));
    t.push(ty(
        "GeneralizedMode",
        vec![
            name(None, "The name of the mode, unique among the body's generalized and pressure modes."),
            c(
                "parity",
                MODES.to_vec(),
                "\"surge\"",
                "The rigid mode whose reflection parity the mode shares (how it reflects about the symmetry planes of the mesh).",
            ),
        ],
        vec![ld("pieces", vec!["ModePiece"], false, false, false)],
    ));
    t.push(ty(
        "ModePiece",
        vec![
            terms("u", "The terms c x^px y^py z^pz of the x displacement, as rows (c, px, py, pz), body-local."),
            terms("v", "The terms of the y displacement, as u."),
            terms("w", "The terms of the z displacement, as u."),
            ranges("The elements of the body's mesh where the piece applies: [(first, last), ...], one-based ranges of its stored elements; unset: every element."),
        ],
        vec![],
    ));
    t.push(ty(
        "Tank",
        vec![
            mesh_of("The walls of the tank (normals into the tank fluid): the name of the body's mesh, body coordinates, with ranges its elements; the sheets inside the tank are its dipoles and dipole dampers."),
            ranges(RANGES),
            r(
                "density_ratio",
                "",
                "1.0",
                ItemRule::Nonnegative,
                "The density of the tank fluid relative to environment.density; 0: the tank empty.",
            ),
            either(
                r(
                    "surface_level",
                    "m",
                    "\"walls_top\"",
                    ItemRule::Finite,
                    "The body-local z of the tank's free surface, or walls_top: the highest vertex of the walls.",
                ),
                vec!["walls_top"],
            ),
            a("field_points", T::Real, "N,3", "m", None, ItemRule::Points, "The points (N, 3) of the field outputs inside the tank (its sloshing), body coordinates (x, y, z), at or below its surface level."),
            enabled("the tank with its sheets and field points"),
        ],
        vec![
            lo("dipoles", vec!["Dipole"]),
            lo("dipole_dampers", vec!["DipoleDamper"]),
            lo("free_surface_plates", vec!["FreeSurfacePlate"]),
        ],
    ));
    t.push(ty(
        "PressureSurface",
        vec![
            mesh_of("The surface where the pressure is prescribed: the name of a mesh, body coordinates."),
            ranges(RANGES),
            b("chamber_coupling", "false", "The restoring coupling of a constant-volume chamber (an air cushion)."),
            c("response_convention", vec!["variational", "numeric_compatibility"], "\"variational\"", "The restoring of the pressure modes: variational, or numeric_compatibility (the overlap stiffness with the opposite sign, the roof coupling in the rigid-mode rows only)."),
            enabled("the pressure surface with its pressure modes"),
        ],
        vec![
            ld("pressure_modes", vec!["PressureMode"], true, false, false),
        ],
    ));
    t.push(ty(
        "FreeSurfacePlate",
        vec![
            mesh_of("The plate: the name of a mesh, body coordinates; massless rigid elements lying in the free surface z = 0 (the body placed; a plate of a tank at the tank's surface_level), the water below them only, moving with the body's rigid modes and with the generalized modes whose pieces name its elements. Unset with inside: the plate is generated (not a plate of a tank)."),
            ranges(RANGES),
            a("inside", T::Real, "2", "m", None, ItemRule::Finite, "A generated plate: a point (x, y) of the free surface inside a waterline loop of a trimmed hull, in the body coordinates of that hull (after its trim); the plate is the water of z = 0 inside the loop (a moonpool, the water column of a tube), less the waterplanes of hulls within it, triangulated by the engine, its edges on the waterline. With the symmetry of the hull the mirror images of the region too. Unset: the plate is given by mesh."),
            ref_("body", "bodies", "The trimmed body whose waterline loop the generated plate covers (a piston body over a trimmed tube); unset: the plate's own body, which then has a trim."),
            enabled("the plate"),
        ],
        vec![],
    ));
    t.push(ty(
        "PressureMode",
        vec![
            name(None, "The name of the mode, unique among the body's generalized and pressure modes."),
            c(
                "shape",
                vec!["uniform", "sign_x", "sign_y"],
                "\"uniform\"",
                "The distribution n(x, y) of the pressure: uniform, the sign of x, or the sign of y.",
            ),
            r("additional_stiffness", "N/m", "0.0", ItemRule::Finite, "An additional stiffness of the coordinate."),
            r("additional_damping", "N s/m", "0.0", ItemRule::Finite, "An additional damping of the coordinate."),
        ],
        vec![],
    ));
    t.push(ty(
        "DampingLid",
        vec![
            name(None, "The unique name of the lid."),
            mesh_of("The lid at z = 0: the name of a mesh, global coordinates, or with placed_with elements of that body's mesh."),
            ranges(RANGES),
            ref_("placed_with", "bodies", "The name of a body whose position and yaw place the lid, taken from that body's mesh (unset: the lid mesh is in global coordinates)."),
            r(
                "damping_factor",
                "m/s",
                "0.0",
                ItemRule::Nonnegative,
                "The damping factor epsilon >= 0 of the free-surface condition on the lid.",
            ),
            enabled("the lid"),
        ],
        vec![],
    ));
    t.push(ty(
        "Mesh",
        vec![
            c("kind", vec!["panels", "flat_patches", "spline_patches", "analytic_patches", "trimmed_patches"], "\"panels\"", "The kind of the mesh: panels (low order), flat_patches, spline_patches or analytic_patches (higher order), or trimmed_patches (a patch mesh trimmed at the waterline)."),
            mesh_item(a("vertices", T::Real, "P,4,3", "m", "[]", ItemRule::Finite, "panels and flat_patches: four vertices (P, 4, 3) per panel or patch, mesh coordinates, the normals into the fluid."), "panels flat_patches"),
            mesh_item(r("reference_length", "m", "1.0", ItemRule::Positive, "The reference length L of the normalized results."), "panels flat_patches spline_patches analytic_patches"),
            mesh_item(c("symmetry", vec!["none", "yz", "xz", "xz_and_yz"], "\"none\"", "The symmetry planes of the stored elements: none; yz the plane x = 0 (the mesh reflected x -> -x); xz the plane y = 0 (y -> -y); xz_and_yz both (one quadrant stored)."), ""),
            mesh_item(txt("label", "\"\"", "The title of the mesh (the first line of its GDF file)."), "panels flat_patches spline_patches analytic_patches"),
            mesh_item(r("file_gravity", "m/s^2", None, ItemRule::Finite, "The gravity of the GDF file it was read from; not used."), "panels flat_patches spline_patches analytic_patches"),
            mesh_item(txt("source_file", "\"\"", "The absolute path of the GDF, CSF or geometry (.hbg) file the mesh was read from, set when the engine reads it (empty for a mesh given otherwise); kept with the model, not used by the solve."), "panels flat_patches spline_patches analytic_patches"),
            mesh_item(txt("source_spl", "\"\"", "The absolute path of the SPL file read with the GDF file, empty without one; kept with the model, not used by the solve."), "flat_patches spline_patches analytic_patches"),
            mesh_item(a("discretization", T::Int, "P,6", "", "[]", ItemRule::Discretization, "Patch meshes: per patch the potential discretization nu, nv (spans), ku, kv (orders), iquo, iqui (Gauss orders)."), "flat_patches spline_patches analytic_patches trimmed_patches"),
            mesh_item(ref_("source", "meshes", "trimmed_patches: the name of the untrimmed patch mesh."), "trimmed_patches"),
            mesh_item(a("trim", T::Real, "3", "", "[0.0, 0.0, 0.0]", ItemRule::Finite, "trimmed_patches: the elevation, pitch and roll of the trim."), "trimmed_patches"),
            mesh_item(a("trim_elements", T::Int, "P,4", "", "[]", ItemRule::None, "trimmed_patches: per element the source patch, its reflection sx, sy and how the waterline cuts it."), "trimmed_patches"),
            mesh_item(a("parameter_ranges", T::Real, "P,4", "", "[]", ItemRule::Finite, "trimmed_patches: per element the part u0, u1, v0, v1 of the source patch it covers."), "trimmed_patches"),
            mesh_item(a("source_elements", T::Int, "P", "", None, ItemRule::Nonnegative, "A mesh made by a trim: per element the one-based element of the source it comes from."), "panels trimmed_patches"),
        ],
        vec![
            lo("patches", vec!["SplinePatch", "AnalyticPatch"]),
        ],
    ));
    t.push(ty(
        "SplinePatch",
        vec![
            i("u_order", None, ItemRule::IntRange, 2.0, 8.0, "The order of the geometry B-splines in u."),
            i("v_order", None, ItemRule::IntRange, 2.0, 8.0, "The order of the geometry B-splines in v."),
            a("u_knots", T::Real, "N", "", None, ItemRule::Finite, "The complete u knot vector."),
            a("v_knots", T::Real, "N", "", None, ItemRule::Finite, "The complete v knot vector."),
            a(
                "coefficients",
                T::Real,
                "Mv,Mu,3",
                "m",
                None,
                ItemRule::Finite,
                "The control coefficients (M_v, M_u, 3), the u index fastest.",
            ),
            a(
                "weights",
                T::Real,
                "Mv,Mu",
                "",
                None,
                ItemRule::Positive,
                "The weights of the control coefficients (M_v, M_u) of a rational patch (NURBS); unset: all 1, the non-rational patch.",
            ),
        ],
        vec![],
    ));
    t.push(ty(
        "AnalyticPatch",
        vec![
            c("map", vec!["line", "arc", "bilinear", "ellipses", "segment_arc"], None, "The map of the patch: line, arc, bilinear, ellipses or segment_arc."),
            c("profile", vec!["linear", "smooth", "quadratic_start", "quadratic_end"], "\"linear\"", "The profile P(t) of the map in v: linear, smooth, quadratic_start, quadratic_end."),
            r("a", "m", "1.0", ItemRule::Finite, "The horizontal scale of x."),
            r("b", "m", "1.0", ItemRule::Finite, "The horizontal scale of y."),
            r("theta0", "rad", "0.0", ItemRule::Finite, "The azimuth at u = -1."),
            r("theta1", "rad", "0.0", ItemRule::Finite, "The azimuth at u = 1."),
            a("p", T::Real, "4", "", "[0.0, 0.0, 0.0, 0.0]", ItemRule::Finite, "The four parameters of the kind (line: rho0, z0, rho1, z1; arc: c, phi0, phi1; ellipses: a0, b0, z0, z1; segment_arc: z1)."),
            a("vertices", T::Real, "N,3", "m", None, ItemRule::Finite, "bilinear: the four corners; segment_arc: the two ends of the segment."),
        ],
        vec![
        ],
    ));
    t.push(ty(
        "Tool",
        vec![txt(
            "text",
            "\"\"",
            "The data of a client of the engine (a graphical interface, a script), any UTF-8 text, kept as it is.",
        )],
        vec![],
    ));

    t
}

impl TypeDef {
    pub fn item(&self, n: &str) -> Option<&ItemDef> {
        self.items.iter().find(|d| n == d.name)
    }
    pub fn list(&self, n: &str) -> Option<&ListDef> {
        self.lists.iter().find(|l| n == l.name)
    }
}

// the descriptions of the lists of children
fn describe_lists(types: &mut [TypeDef]) {
    const DESCRIPTIONS: &[(&str, &str, &str)] = &[
        (
            "Model",
            "environment",
            "The water depth, gravity and density, the walls of the fluid domain, and the frequencies and headings of the waves.",
        ),
        ("Model", "options", "How the problem is solved: the method and its parameters."),
        (
            "Model",
            "first_order",
            "What the first-order solve computes: the problems solved, the motions, the field and body outputs, and the field points in the sea.",
        ),
        ("Model", "second_order", "The second-order loads: the mean drifts and the QTFs, by method."),
        ("Model", "bodies", "The bodies, by name."),
        ("Model", "lids", "The damping lids of the free surface, by name (global coordinates, or placed with a body)."),
        (
            "Model",
            "fixed_dampers",
            "Porous sheets fixed in space (global coordinates): the jump -(i epsilon / omega) phi_n, in no force.",
        ),
        ("Model", "meshes", "The meshes the bodies and the surfaces use, by name."),
        ("Model", "tool", "The data of other programs (any UTF-8 text) kept with the model, by program name."),
        ("Body", "dipoles", "Zero-thickness rigid sheets moving with the body (heave plates, strakes)."),
        (
            "Body",
            "dipole_dampers",
            "Porous sheets moving with the body: the jump -(i epsilon / omega)(phi_n - v_n), in the body forces.",
        ),
        ("Body", "generalized_modes", "Displacement modes after the six rigid ones, by name; always free."),
        (
            "Body",
            "couplings",
            "External stiffness and damping between this body's forces and another body's motions, one per other body.",
        ),
        ("Body", "tanks", "Internal tanks: their walls among the elements of the body's mesh, and their field points."),
        (
            "Tank",
            "dipoles",
            "Rigid sheets inside the tank (baffles), elements of the body's mesh in the tank fluid, moving with the body.",
        ),
        (
            "Tank",
            "dipole_dampers",
            "Porous sheets inside the tank, elements of the body's mesh in the tank fluid: the jump -(i epsilon / omega)(phi_n - v_n).",
        ),
        (
            "Tank",
            "free_surface_plates",
            "Massless rigid plates in the tank's free surface (a deck over part of the tank, a full cover), moving with the body: elements of the body's mesh or a mesh of their own at the tank's surface_level.",
        ),
        ("Body", "pressure_surfaces", "Surfaces where the pressure is prescribed by pressure modes."),
        (
            "Body",
            "free_surface_plates",
            "Massless rigid plates in the free surface z = 0 moving with the body (a piston, a mat, a cover).",
        ),
        ("GeneralizedMode", "pieces", "The polynomial pieces of the mode, each on elements of the body's mesh."),
        ("PressureSurface", "pressure_modes", "The pressure modes of the surface, by name."),
        ("Mesh", "patches", "The B-spline or analytic patches of a spline_patches or analytic_patches mesh."),
    ];
    for t in types.iter_mut() {
        for l in t.lists.iter_mut() {
            for d in DESCRIPTIONS.iter() {
                if t.name == d.0 && l.name == d.1 {
                    l.description = d.2;
                }
            }
        }
    }
}

// the switch items of the switched items: of a surface's items by their prefix (the method, and
// the outer of the automatic control surface where its values name one), and the either-or inputs
fn switches(types: &mut [TypeDef]) {
    const EITHER: &[(&str, &str, &str, &str)] = &[
        ("Environment", "omegas", "frequency_input", "omegas"),
        ("Environment", "periods", "frequency_input", "periods"),
        ("Body", "Ixx", "inertia_input", "components"),
        ("Body", "Ixy", "inertia_input", "components"),
        ("Body", "Ixz", "inertia_input", "components"),
        ("Body", "Iyy", "inertia_input", "components"),
        ("Body", "Iyz", "inertia_input", "components"),
        ("Body", "Izz", "inertia_input", "components"),
        ("Body", "radii_of_gyration", "inertia_input", "radii_of_gyration"),
    ];
    for t in types.iter_mut() {
        for d in t.items.iter_mut() {
            if let Some(&(_, _, s, v)) = EITHER.iter().find(|e| e.0 == t.name && e.1 == d.name) {
                (d.switch, d.switch_values) = (s, v);
            } else if !d.switch_values.is_empty() {
                let two = d.switch_values.contains('/');
                d.switch = match () {
                    _ if d.name.starts_with("control_surface_") && two => "control_surface control_surface_outer",
                    _ if d.name.starts_with("control_surface_") => "control_surface",
                    _ if d.name.starts_with("interior_surface_") => "interior_surface",
                    _ if d.name.starts_with("free_surface_") => "free_surface_method",
                    _ => panic!("the switch of {}.{}", t.name, d.name),
                };
            }
        }
    }
}

fn described() -> Vec<TypeDef> {
    let mut t = build();
    describe_lists(&mut t);
    switches(&mut t);
    t
}

static SCHEMA: LazyLock<Vec<TypeDef>> = LazyLock::new(described);

/// The schema: every object type (the model first), built once.
pub fn schema() -> &'static [TypeDef] {
    &SCHEMA
}

pub fn schema_type(n: &str) -> &'static TypeDef {
    for t in schema() {
        if n == t.name {
            return t;
        }
    }
    panic!("no object type '{n}'")
}
