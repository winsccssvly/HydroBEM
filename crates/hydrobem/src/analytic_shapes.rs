//! The named analytic shapes (reference/inputs-meshes §2): a circular or elliptic cylinder, a sphere or an
//! ellipsoid, a barge, a cylinder with a moonpool and a cylinder with annular dipole discs, as
//! their analytic patches (the quadrant x, y >= 0; the mesh symmetry gives the rest). Each
//! shape has named dimensions and named parts (its patches, in the order given).

use crate::constants::PI;
use crate::error::{Error, Result};
use crate::mesh_desc::{AnalyticKind, AnalyticPatchDesc, Profile};

const QUADRANT: f64 = 0.5 * PI;

/// A shape: its name, dimensions, parts (the default ones first) and whether it has the
/// nonuniform mapping (the profiles with zero slope at the patch ends).
struct ShapeDef {
    name: &'static str,
    dimensions: &'static [&'static str],
    optional: &'static [&'static str],
    parts: &'static [&'static str],
    default_parts: usize,
    nonuniform: bool,
}

const SHAPES: [ShapeDef; 7] = [
    ShapeDef {
        name: "circular_cylinder",
        dimensions: &["radius", "draft"],
        optional: &[],
        parts: &["side", "bottom", "waterplane"],
        default_parts: 2,
        nonuniform: true,
    },
    ShapeDef {
        name: "elliptic_cylinder",
        dimensions: &["a", "b", "draft"],
        optional: &[],
        parts: &["side", "bottom", "waterplane"],
        default_parts: 2,
        nonuniform: true,
    },
    ShapeDef {
        name: "sphere",
        dimensions: &["radius"],
        optional: &[],
        parts: &["hull", "waterplane"],
        default_parts: 1,
        nonuniform: true,
    },
    ShapeDef {
        name: "ellipsoid",
        dimensions: &["a", "b", "c"],
        optional: &[],
        parts: &["hull", "waterplane"],
        default_parts: 1,
        nonuniform: false,
    },
    ShapeDef {
        name: "barge",
        dimensions: &["half_length", "half_beam", "draft"],
        optional: &[],
        parts: &["end", "side", "bottom", "waterplane"],
        default_parts: 3,
        nonuniform: false,
    },
    ShapeDef {
        name: "moonpool_cylinder",
        dimensions: &["radius", "draft", "moonpool_radius"],
        optional: &[],
        parts: &["side", "bottom", "moonpool_wall", "moonpool_surface"],
        default_parts: 3,
        nonuniform: true,
    },
    ShapeDef {
        name: "cylinder_with_dipole_discs",
        dimensions: &["radius", "draft"],
        optional: &["moonpool_radius"],
        parts: &[],
        default_parts: 0,
        nonuniform: false,
    },
];

/// The names of the shapes.
pub fn shape_names() -> Vec<&'static str> {
    SHAPES.iter().map(|s| s.name).collect()
}

/// A named shape with its dimensions: the inputs of `named_shape`.
#[derive(Clone, Debug, Default)]
pub struct ShapeSpec {
    pub shape: String,
    /// the dimensions by name (metres)
    pub dimensions: Vec<(String, f64)>,
    /// cylinder_with_dipole_discs: per disc its outer radius, inner radius and depth
    pub discs: Vec<[f64; 3]>,
    /// the parts in the order of the patches; empty: the default parts of the shape
    pub parts: Vec<String>,
    /// "uniform" (also "") or "nonuniform"
    pub mapping: String,
}

fn line(a: f64, b: f64, rho0: f64, z0: f64, rho1: f64, z1: f64, profile: Profile) -> AnalyticPatchDesc {
    AnalyticPatchDesc {
        kind: AnalyticKind::Line,
        profile,
        a,
        b,
        theta0: 0.0,
        theta1: QUADRANT,
        p: [rho0, z0, rho1, z1],
        ..AnalyticPatchDesc::default()
    }
}

fn quad(v: [f64; 12]) -> AnalyticPatchDesc {
    AnalyticPatchDesc {
        kind: AnalyticKind::Bilinear,
        profile: Profile::Linear,
        a: 1.0,
        b: 1.0,
        vertices: v,
        ..AnalyticPatchDesc::default()
    }
}

fn pick(nonuniform: bool, profile: Profile) -> Profile {
    if nonuniform { profile } else { Profile::Linear }
}

fn names(v: &[&str]) -> String {
    v.join(", ")
}

/// The analytic patches of a named shape, in the order of its parts.
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn named_shape(spec: &ShapeSpec) -> Result<Vec<AnalyticPatchDesc>> {
    let Some(def) = SHAPES.iter().find(|s| s.name == spec.shape) else {
        return Err(Error::value(format!("shape must be one of {}, got '{}'", names(&shape_names()), spec.shape)));
    };
    let at = def.name;
    // the dimensions: each known once, the required ones given
    let mut dims: Vec<(&str, f64)> = Vec::new();
    for (k, v) in &spec.dimensions {
        let Some(&name) = def.dimensions.iter().chain(def.optional).find(|&&d| d == k) else {
            let mut all = def.dimensions.to_vec();
            all.extend(def.optional);
            return Err(Error::value(format!("{at}: unknown dimension '{k}' (the dimensions: {})", names(&all))));
        };
        if dims.iter().any(|(d, _)| *d == name) {
            return Err(Error::value(format!("{at}: the dimension {name} is given twice")));
        }
        dims.push((name, *v));
    }
    let dipoles = def.parts.is_empty();
    for &d in def.dimensions {
        if !dims.iter().any(|(n, _)| *n == d) {
            return Err(Error::value(format!("{at} needs {}", names(def.dimensions))));
        }
    }
    for (n, v) in &dims {
        // the cylinder with dipole discs takes zero dimensions as its file does
        let ok = v.is_finite() && if dipoles { *v >= 0.0 } else { *v > 0.0 };
        if !ok {
            let rule = if dipoles { "finite and not negative" } else { "positive and finite" };
            return Err(Error::value(format!("{at}: {n} must be {rule}, got {v}")));
        }
    }
    let dim = |n: &str| dims.iter().find(|(d, _)| *d == n).map(|(_, v)| *v);
    let nonuniform = match spec.mapping.as_str() {
        "" | "uniform" => false,
        "nonuniform" => true,
        m => return Err(Error::value(format!("{at}: mapping must be one of uniform, nonuniform, got '{m}'"))),
    };
    if nonuniform && !def.nonuniform {
        return Err(Error::value(format!("{at}: the nonuniform mapping is not available (uniform only)")));
    }
    if !spec.discs.is_empty() && !dipoles {
        return Err(Error::value(format!("{at}: discs are given only for cylinder_with_dipole_discs")));
    }
    if dipoles {
        if !spec.parts.is_empty() {
            return Err(Error::value(format!("{at}: the parts are fixed (the cylinder and its discs)")));
        }
        return dipole_discs(spec, dim("radius").unwrap_or(0.0), dim("draft").unwrap_or(0.0), dim("moonpool_radius"));
    }
    // the patches of every part, in the order of def.parts
    let all: Vec<AnalyticPatchDesc> = match def.name {
        "circular_cylinder" | "elliptic_cylinder" => {
            let (a, b, d) = match def.name {
                "circular_cylinder" => (dim("radius").unwrap(), dim("radius").unwrap(), dim("draft").unwrap()),
                _ => (dim("a").unwrap(), dim("b").unwrap(), dim("draft").unwrap()),
            };
            cylinder(a, b, d, nonuniform)
        }
        "sphere" | "ellipsoid" => {
            let (a, b, c) = match def.name {
                "sphere" => (dim("radius").unwrap(), dim("radius").unwrap(), dim("radius").unwrap()),
                _ => (dim("a").unwrap(), dim("b").unwrap(), dim("c").unwrap()),
            };
            let hull = AnalyticPatchDesc {
                kind: AnalyticKind::Arc,
                profile: pick(nonuniform, Profile::QuadraticStart),
                a,
                b,
                theta0: 0.0,
                theta1: QUADRANT,
                p: [c, 0.0, QUADRANT, 0.0],
                ..AnalyticPatchDesc::default()
            };
            vec![hull, line(a, b, 0.0, 0.0, 1.0, 0.0, Profile::Linear)]
        }
        "barge" => {
            let (L, B, T) = (dim("half_length").unwrap(), dim("half_beam").unwrap(), dim("draft").unwrap());
            vec![
                quad([L, 0.0, 0.0, L, B, 0.0, L, B, -T, L, 0.0, -T]),
                quad([L, B, 0.0, 0.0, B, 0.0, 0.0, B, -T, L, B, -T]),
                quad([0.0, 0.0, -T, L, 0.0, -T, L, B, -T, 0.0, B, -T]),
                quad([0.0, B, 0.0, L, B, 0.0, L, 0.0, 0.0, 0.0, 0.0, 0.0]),
            ]
        }
        _ => {
            let (R, T, Rm) = (dim("radius").unwrap(), dim("draft").unwrap(), dim("moonpool_radius").unwrap());
            if Rm >= R {
                return Err(Error::value(format!("{at}: moonpool_radius must be smaller than radius")));
            }
            moonpool(R, T, Rm, nonuniform)
        }
    };
    let parts: Vec<&str> = if spec.parts.is_empty() {
        def.parts[..def.default_parts].to_vec()
    } else {
        spec.parts.iter().map(|s| s.as_str()).collect()
    };
    let mut out = Vec::new();
    for (i, p) in parts.iter().enumerate() {
        let Some(k) = def.parts.iter().position(|q| q == p) else {
            return Err(Error::value(format!("{at}: unknown part '{p}' (the parts: {})", names(def.parts))));
        };
        if parts[..i].contains(p) {
            return Err(Error::value(format!("{at}: the part {p} is given twice")));
        }
        out.push(all[k]);
    }
    Ok(out)
}

/// The side, bottom and waterplane of an elliptic cylinder of semi-axes a, b.
fn cylinder(a: f64, b: f64, draft: f64, nonuniform: bool) -> Vec<AnalyticPatchDesc> {
    vec![
        line(a, b, 1.0, 0.0, 1.0, -draft, pick(nonuniform, Profile::Smooth)),
        line(a, b, 1.0, -draft, 0.0, -draft, pick(nonuniform, Profile::QuadraticStart)),
        line(a, b, 0.0, 0.0, 1.0, 0.0, pick(nonuniform, Profile::QuadraticEnd)),
    ]
}

/// The outer side, the bottom annulus, the moonpool wall and the moonpool's free surface.
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn moonpool(R: f64, T: f64, Rm: f64, nonuniform: bool) -> Vec<AnalyticPatchDesc> {
    vec![
        line(R, R, 1.0, 0.0, 1.0, -T, pick(nonuniform, Profile::QuadraticEnd)),
        line(1.0, 1.0, R, -T, Rm, -T, pick(nonuniform, Profile::Smooth)),
        line(Rm, Rm, 1.0, -T, 1.0, 0.0, pick(nonuniform, Profile::QuadraticStart)),
        line(Rm, Rm, 1.0, 0.0, 0.0, 0.0, Profile::Linear),
    ]
}

/// The cylinder (side and bottom, or with a moonpool its side, bottom and moonpool wall) and one
/// annular dipole disc per entry of spec.discs.
fn dipole_discs(spec: &ShapeSpec, r: f64, d: f64, rm: Option<f64>) -> Result<Vec<AnalyticPatchDesc>> {
    let at = "cylinder_with_dipole_discs";
    for (i, &[ro, ri, z]) in spec.discs.iter().enumerate() {
        if !(ro.is_finite() && ri.is_finite() && z.is_finite() && ro >= 0.0 && ri >= 0.0 && z >= 0.0) {
            return Err(Error::value(format!(
                "{at}: disc {}: outer radius, inner radius and depth must be finite and not negative",
                i + 1
            )));
        }
        if ro <= ri {
            return Err(Error::value(format!("{at}: disc {}: the outer radius must exceed the inner radius", i + 1)));
        }
        // down to the bottom (depth <= draft) the hull fills moonpool radius < rho < radius: a disc
        // there must lie in the water, outside the hull or inside the moonpool, else it cuts the
        // wall or lies in the hull or on its bottom
        let rm0 = rm.unwrap_or(0.0);
        if z <= d && !(ri >= r || ro <= rm0) {
            return Err(Error::value(format!(
                "{at}: disc {}: at a depth not below the draft a disc must lie outside the radius or \
                 inside the moonpool (it cuts the hull)",
                i + 1
            )));
        }
    }
    // as moonpool_cylinder: a moonpool as wide as the hull would turn the bottom annulus over and
    // put the moonpool wall outside the hull (a zero moonpool radius is allowed, as the file's)
    if rm.is_some_and(|rm| rm > 0.0 && rm >= r) {
        return Err(Error::value(format!("{at}: moonpool_radius must be smaller than radius")));
    }
    let mut patches = match rm {
        Some(rm) => moonpool(r, d, rm, false)[..3].to_vec(),
        None => cylinder(r, r, d, false)[..2].to_vec(),
    };
    for &[ro, ri, z] in &spec.discs {
        patches.push(line(1.0, 1.0, ro, -z, ri, -z, Profile::Linear));
    }
    Ok(patches)
}
