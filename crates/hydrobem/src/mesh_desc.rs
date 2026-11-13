//! The mesh descriptors of the engine: a low-order mesh, a
//! higher-order mesh with its patches, discretizations and trimmed elements, a trim and the
//! automatic control surface. The model fills them from its mesh objects (document_mesh.rs);
//! the readers, the trim and the geometry work on them. Clients give meshes as objects of the
//! model instead.
//!
//! A descriptor owns its arrays (`Vec`; empty where the mesh has none) and its count fields (the
//! readers and the checks compare them with the array lengths).

/* ---- meshes ------------------------------------------------------------ */

/// Low-order mesh (constant panels; reference/inputs-meshes §1).
#[derive(Clone, Debug, Default)]
pub struct LowOrderMeshDesc {
    /// P
    pub panel_count: i32,
    /// P x 4 x 3, body-local metres, raw order (normals into the fluid)
    pub vertices: Vec<f64>,
    /// the reference length L > 0
    pub reference_length: f64,
    /// gravity of the file (not used for solves)
    pub gravity: f64,
    /// symmetry flag in {0,1}: reflected about x = 0
    pub symmetry_x: i32,
    /// symmetry flag in {0,1}: reflected about y = 0
    pub symmetry_y: i32,
}

/// The kind of a mesh object (its item `kind`): low-order panels or higher-order patches.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MeshKind {
    /// constant-strength plane panels (the low-order method)
    #[default]
    Panels,
    /// bilinear patches from four coplanar vertices
    FlatPatches,
    /// B-spline (NURBS) patches
    SplinePatches,
    /// analytic patches
    AnalyticPatches,
    /// the elements of a patch mesh trimmed at the waterline
    TrimmedPatches,
}

impl MeshKind {
    pub const ALL: [MeshKind; 5] = [
        MeshKind::Panels,
        MeshKind::FlatPatches,
        MeshKind::SplinePatches,
        MeshKind::AnalyticPatches,
        MeshKind::TrimmedPatches,
    ];
    /// The name of the kind in the model, its files and the APIs.
    pub fn name(self) -> &'static str {
        match self {
            MeshKind::Panels => "panels",
            MeshKind::FlatPatches => "flat_patches",
            MeshKind::SplinePatches => "spline_patches",
            MeshKind::AnalyticPatches => "analytic_patches",
            MeshKind::TrimmedPatches => "trimmed_patches",
        }
    }
    /// The kind of a name.
    pub fn of(name: &str) -> Option<MeshKind> {
        MeshKind::ALL.into_iter().find(|k| k.name() == name)
    }
    /// Whether the mesh holds higher-order patches.
    pub fn patches(self) -> bool {
        self != MeshKind::Panels
    }
    /// The patch type of the list `patches` of a mesh of this kind, if it has one.
    pub fn patch_type(self) -> Option<&'static str> {
        match self {
            MeshKind::SplinePatches => Some("SplinePatch"),
            MeshKind::AnalyticPatches => Some("AnalyticPatch"),
            _ => None,
        }
    }
    /// The kind of a mesh as read from a GDF file (or made by the engine).
    pub fn of_geometry(higher_order: bool, geometry: PatchKind) -> MeshKind {
        match (higher_order, geometry) {
            (false, _) => MeshKind::Panels,
            (true, PatchKind::Flat) => MeshKind::FlatPatches,
            (true, PatchKind::Spline) => MeshKind::SplinePatches,
            (true, PatchKind::Analytic) => MeshKind::AnalyticPatches,
        }
    }
}

/// Patch geometry of a higher-order mesh (the geometry code of a GDF header).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PatchKind {
    /// bilinear patches from four vertices (code 0)
    #[default]
    Flat,
    /// B-spline patches (code 1)
    Spline,
    /// analytic patches (code < 0)
    Analytic,
}

/// Kind of an analytic patch: theta = theta0 + (theta1 - theta0)(u + 1)/2, t = (v + 1)/2.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AnalyticKind {
    /// revolved segment: rho, z linear in P(t) from (p0, p1) to (p2, p3); x = a rho cos, y = b rho sin
    #[default]
    Line,
    /// revolved elliptic arc: phi = p1 + (p2 - p1) P(t); x = a cos(phi) cos, y = b cos(phi) sin, z = -p0 sin(phi)
    Arc,
    /// bilinear patch of four corners (-1,-1), (1,-1), (1,1), (-1,1)
    Bilinear,
    /// from the ellipse (p0, p1) at z = p2 to the ellipse (a, b) at z = p3, linear in P(t)
    Ellipses,
    /// from the segment vertices[0..2] - vertices[3..5] to the arc (a cos, b sin, p0), linear in P(t)
    SegmentArc,
}

/// Profile P(t) of an analytic patch in v.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Profile {
    /// t
    #[default]
    Linear,
    /// 3t^2 - 2t^3 (zero slope at both ends)
    Smooth,
    /// t^2 (zero slope at t = 0)
    QuadraticStart,
    /// 2t - t^2 (zero slope at t = 1)
    QuadraticEnd,
}

/// One analytic patch; u x v points into the body.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AnalyticPatchDesc {
    pub kind: AnalyticKind,
    pub profile: Profile,
    /// horizontal scales of x and y
    pub a: f64,
    pub b: f64,
    /// azimuth at u = -1 and u = 1, radians
    pub theta0: f64,
    pub theta1: f64,
    /// line: rho0 z0 rho1 z1; arc: c phi0 phi1 (radians); ellipses: a0 b0 z0 z1; segment-arc: z1
    pub p: [f64; 4],
    /// bilinear: the four corners; segment-arc: the two segment ends
    pub vertices: [f64; 12],
}

/// Potential discretization of one patch (one SPL element).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PatchDiscretization {
    /// B-spline spans >= 1
    pub nu: i32,
    pub nv: i32,
    /// orders 2..6
    pub ku: i32,
    pub kv: i32,
    /// Gauss orders of the outer and inner integrals >= 1
    pub iquo: i32,
    pub iqui: i32,
}

/// One element of a trimmed patch mesh: a source patch,
/// reflected by (sx, sy), in the trimmed position, and how the waterline cuts it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PatchTrim {
    /// zero-based source patch (one-based in the item trim_elements of a mesh object)
    pub patch: i32,
    /// reflection of the source patch (+1 or -1; -1 needs that symmetry flag)
    pub sx: i32,
    pub sy: i32,
    /// 0 wholly submerged; 1, 2: the submerged part of every line of constant u,
    /// from v = -1 (1) or v = +1 (2) to the waterline; 3, 4: of constant v from u = -1, +1
    pub cut: i32,
    /// of the element
    pub discretization: PatchDiscretization,
    /// u0, u1, v0, v1: the part of the source patch's parameter square [-1, 1]^2 the element
    /// covers, mapped linearly onto its own square (a patch is split where its fibres change
    /// from wholly submerged to cut); all four 0: the whole square
    pub range: [f64; 4],
}

/// Higher-order mesh (B-spline or analytic patches; reference/inputs-meshes §2).
#[derive(Clone, Debug, Default)]
pub struct PatchMeshDesc {
    /// PatchKind::Flat, PatchKind::Spline or PatchKind::Analytic
    pub geometry: PatchKind,
    /// number of patches
    pub patch_count: i32,
    /// flat: patch_count x 4 x 3 vertices
    pub vertices: Vec<f64>,
    /// spline: per patch orders (ku, kv) and coefficient counts
    /// (M_u, M_v) = (nu + ku - 1, nv + kv - 1), nu/nv being the spans;
    /// knots concatenated per patch (u knots then v knots, M_u + ku and
    /// M_v + kv values); coefficients concatenated per patch, M_u * M_v x 3,
    /// u index fastest.
    /// patch_count x 2 (ku, kv)
    pub spline_orders: Vec<i32>,
    /// patch_count x 2 (M_u, M_v)
    pub spline_counts: Vec<i32>,
    /// all knot vectors
    pub knots: Vec<f64>,
    /// number of knots
    pub knot_count: i64,
    /// all control coefficients
    pub coefficients: Vec<f64>,
    /// number of 3-vectors
    pub coefficient_count: i64,
    /// the weights of the coefficients of rational patches (one per coefficient, 1 for the
    /// patches without weights); empty when no patch is rational
    pub weights: Vec<f64>,
    /// patch_count elements
    pub discretization: Vec<PatchDiscretization>,
    /// the reference length L > 0
    pub reference_length: f64,
    /// gravity of the file (not used for solves)
    pub gravity: f64,
    /// symmetry flag in {0,1}: reflected about x = 0
    pub symmetry_x: i32,
    /// symmetry flag in {0,1}: reflected about y = 0
    pub symmetry_y: i32,
    /// analytic: patch_count elements
    pub analytic_patches: Vec<AnalyticPatchDesc>,
    /// Trimmed waterline: trim_count > 0 elements replace the patches; the
    /// source patches (above) are moved by x = R(pitch, roll) xi + (0, 0, elevation).
    pub trim: Vec<PatchTrim>,
    pub trim_count: i32,
    /// m, degrees, degrees
    pub trim_elevation: f64,
    pub trim_pitch: f64,
    pub trim_roll: f64,
}

/// A trim: the mesh, in its own coordinates xi,
/// is moved to x = R(pitch, roll) xi + (0, 0, elevation) and cut at the free surface.
#[derive(Clone, Debug, Default)]
pub struct TrimDesc {
    /// m: z of the mesh origin in body coordinates
    pub elevation: f64,
    /// degrees: the pitch about the eta axis first, then the roll about the xi axis
    pub pitch: f64,
    pub roll: f64,
    /// higher order: also the interior free surface inside the trimmed waterline (0: none)
    pub interior_surface: i32,
    /// low order, None (null) or one per element: the body z where that element is cut (a tank free
    /// surface: the tank elements are cut at their own free surface, not at z = 0); NaN: z = 0
    pub record_cut: Option<Vec<f64>>,
    /// the free-surface plates generated in the trimmed waterline: per plate a point (x, y) in body
    /// coordinates inside the waterline loop it covers (the elements appended after the trimmed
    /// ones, after the interior surface of a patch mesh)
    pub plates: Vec<[f64; 2]>,
}

/// The automatic control surface (theory §11.3).
#[derive(Clone, Debug, Default)]
pub struct ControlSurfaceDesc {
    /// > 0: a circular outer surface of this radius about the body origin; <= 0: the polygon
    pub radius: f64,
    /// below the free surface, > 0
    pub depth: f64,
    /// the panel size, > 0
    pub panel_size: f64,
    /// the free surface in body axes (minus the body's z position)
    pub waterline_z: f64,
    /// also the free surface between the body waterline and the outer surface
    pub free_surface: i32,
    /// with radius <= 0: (x, y) vertices in the stored sector, counter-clockwise about the origin
    pub polygon: Vec<f64>,
    pub polygon_count: i32,
    /// further (x, y) polylines, concatenated
    pub partitions: Vec<f64>,
    /// vertices per partition
    pub partition_counts: Vec<i32>,
    pub partition_count: i32,
    /// the elements of the body's free-surface plates as (x, y) polygons in body axes: the waterline
    /// they cover (a covered moonpool) borders no free surface and is left out
    pub covers: Vec<Vec<[f64; 2]>>,
}
