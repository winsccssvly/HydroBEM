//! The result tree (reference/results §1): the views of the result quantities by group and by
//! owner. Each row of the quantity table (quantities.rs) names its view, a dot path such as
//! "drift.pressure_integration" or "lids.elevation_radiation", and the rule that selects the part of
//! one owner (`Owner`): the columns of a body on the last axis, its coordinates and radiation
//! columns, its body points, the points of one lid, of one pressure surface or of one tank. `Results::select`
//! applies the rule; the Python package generates its views from `result_views` and
//! `result_groups`, and the C ABI hands them out as tables.
//!
//! A path of one segment is the quantity itself on the results (the axes `omegas`, `headings`,
//! `radiation_mode_ids`; `added_mass`, ...) and, for a quantity read by body, its part on a body
//! (`bodies[b].added_mass`); "excitation", "body_points", "drift" and "qtf" are groups of the
//! results and of a body, "field_points", "kochin" and "hydrostatics" of the results; "lids",
//! "pressure_surfaces" and "tanks" are lists of owners, the one a list of the results, the others of
//! each body.

use std::ops::Range;

use crate::codes::QuantityId;
use crate::error::{Error, Result};
use crate::results::{QuantityData, Results};

/// How the part of an owner of a quantity is selected (the rule of its view).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Owner {
    /// The whole array, also for a body (the heading pairs, the Newman approximation of the drift by
    /// momentum conservation: of all bodies).
    Shared,
    /// The whole array of the results; none for a body (the field points, the drift by momentum
    /// conservation).
    Global,
    /// A body's n columns of the last axis (`n*body` of the axes): b n .. (b + 1) n.
    Columns(usize),
    /// A body's rows of the coordinate axes (its coordinates, by the `coordinates` quantity) and of
    /// the radiation-column axes (the columns of its coordinates, by `radiation_mode_ids`).
    Coordinates,
    /// A body's rows of the body-point axis (those of `body_points_ids` whose body is it).
    BodyPoints,
    /// The rows of the lid-point axis of one lid (`lids_ids`, one-based); no body.
    Lid,
    /// The rows of the pressure-point axis of one pressure surface of a body (`pressure_surfaces_ids`,
    /// numbered over the bodies in order, one-based).
    PressureSurface,
    /// The rows of the tank-point axis of one tank of a body (`tanks_ids`, numbered over the bodies
    /// in order, one-based).
    Tank,
}

impl Owner {
    /// Its name: "shared", "global", "columns", "coordinates", "body_points", "lid",
    /// "pressure_surface", "tank".
    pub fn name(self) -> &'static str {
        match self {
            Owner::Shared => "shared",
            Owner::Global => "global",
            Owner::Columns(_) => "columns",
            Owner::Coordinates => "coordinates",
            Owner::BodyPoints => "body_points",
            Owner::Lid => "lid",
            Owner::PressureSurface => "pressure_surface",
            Owner::Tank => "tank",
        }
    }
    /// The columns of a body of `Columns` (0 for the other rules).
    pub fn width(self) -> usize {
        match self {
            Owner::Columns(n) => n,
            _ => 0,
        }
    }
    /// Whether a body's view has the path: every rule but `Global` and `Lid` (a `Global` path of a
    /// group that a body has reads as none).
    pub fn of_body(self) -> bool {
        !matches!(self, Owner::Global | Owner::Lid)
    }
}

/// A group of the result tree: an inner node, the path of its views.
#[derive(Clone, Copy, Debug)]
pub struct ResultGroup {
    pub path: &'static str,
    pub description: &'static str,
}

/// A view of the result tree: its path, the quantity it reads and the rule of an owner's part.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResultView {
    pub path: &'static str,
    pub quantity: QuantityId,
    pub owner: Owner,
}

#[rustfmt::skip]
static GROUPS: [ResultGroup; 18] = [
    ResultGroup { path: "excitation", description: "The exciting forces per wave amplitude by method (the excitation_* quantities): diffraction (by pressure integration), froude_krylov and scattering [F, H, D], haskind [F, H, K] (by the Haskind relation)" },
    ResultGroup { path: "body_points", description: "The outputs on the bodies at the body output points (the body_points_* quantities): the points, their ids, normals and areas, the pressures and velocities" },
    ResultGroup { path: "field_points", description: "The outputs at the field points in the sea (the field_points_* quantities): the points, the pressures and velocities of the radiation, incident, scattered and total waves" },
    ResultGroup { path: "lids", description: "The outputs of one damping lid at its points (the rows of the lids_* quantities whose lids_ids is it), by name or position" },
    ResultGroup { path: "kochin", description: "The Kochin functions of the radiation and scattered potentials (theory §11.1), at the angles [A] (rad)" },
    ResultGroup { path: "drift", description: "The mean drifts by method (the drift_* quantities), of the heading pairs heading_pairs [P, 2] (deg): (beta, beta) of every heading, or every pair beta_a <= beta_b with second_order.drift_bidirectional; complex (the coefficient of a pair of two headings is complex, of equal headings real)" },
    ResultGroup { path: "qtf", description: "The QTFs by part and method (the qtf_* quantities), of the heading pairs heading_pairs [P, 2] (deg): quadratic.pressure_integration, quadratic.control_surface, potential.direct, potential.indirect, each with difference and sum [F, F, P, D] on the coordinates; newman.<drift method>, Newman's approximation difference [F, F, H, D] of the equal headings (momentum conservation [F, F, H, 3])" },
    ResultGroup { path: "qtf.quadratic", description: "The quadratic part of the QTFs: by pressure integration and by the control surface" },
    ResultGroup { path: "qtf.quadratic.pressure_integration", description: "The quadratic QTF by pressure integration at the difference and the sum frequencies" },
    ResultGroup { path: "qtf.quadratic.control_surface", description: "The quadratic QTF by the control surface at the difference and the sum frequencies" },
    ResultGroup { path: "qtf.potential", description: "The force of the second-order potential: by the direct and the indirect method" },
    ResultGroup { path: "qtf.potential.direct", description: "The force of the second-order potential by the direct method at the difference and the sum frequencies" },
    ResultGroup { path: "qtf.potential.indirect", description: "The force of the second-order potential by the indirect method at the difference and the sum frequencies" },
    ResultGroup { path: "qtf.newman", description: "Newman's approximation of the difference-frequency QTF, by the method of its mean drift: pressure_integration, control_surface, momentum_conservation" },
    ResultGroup { path: "qtf.newman.pressure_integration", description: "Newman's approximation from the mean drift by pressure integration" },
    ResultGroup { path: "qtf.newman.control_surface", description: "Newman's approximation from the mean drift by the control surface" },
    ResultGroup { path: "qtf.newman.momentum_conservation", description: "Newman's approximation from the mean drift by momentum conservation (of all bodies, global axes)" },
    ResultGroup { path: "hydrostatics", description: "The hydrostatics of every body of the solve [B, ...] (a body's: bodies[b].hydrostatics)" },
];

// the groups of the pressure surfaces and of the tanks are lists of each body: listed after those of
// the results
#[rustfmt::skip]
static BODY_GROUPS: [ResultGroup; 2] = [
    ResultGroup { path: "pressure_surfaces", description: "The outputs of one pressure surface of a body at its points (the rows of the pressure_surfaces_* quantities whose pressure_surfaces_ids is it), by name or position" },
    ResultGroup { path: "tanks", description: "The outputs of one tank of a body at its points (the rows of the tanks_* quantities whose tanks_ids is it), by name or position; radiation only (the fluid of a tank has no incident or scattered wave)" },
];

/// The groups of the result tree (its inner nodes), parents before their children.
pub fn result_groups() -> Vec<ResultGroup> {
    GROUPS.iter().chain(BODY_GROUPS.iter()).copied().collect()
}

/// The views of the result tree: every quantity with a group, in the order of the quantity table.
pub fn result_views() -> Vec<ResultView> {
    QuantityId::ALL
        .iter()
        .filter(|q| !q.group().is_empty())
        .map(|&q| ResultView { path: q.group(), quantity: q, owner: q.owner() })
        .collect()
}

/// The view of a path (`ErrorKind::Value` for a path of no view).
pub fn result_view(path: &str) -> Result<ResultView> {
    QuantityId::ALL
        .iter()
        .find(|q| !q.group().is_empty() && q.group() == path)
        .map(|&q| ResultView { path: q.group(), quantity: q, owner: q.owner() })
        .ok_or_else(|| Error::value(format!("no result view {path:?}")))
}

/// The rows of an axis an owner's part has: a range when they are contiguous (and some), else the
/// indices (zero-based).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Rows {
    Range(Range<usize>),
    Indices(Vec<usize>),
}

impl Rows {
    /// The rows of the indices (ascending): a range when they follow each other.
    pub fn of(v: Vec<usize>) -> Rows {
        match (v.first(), v.last()) {
            (Some(&a), Some(&b)) if v.windows(2).all(|w| w[1] == w[0] + 1) => Rows::Range(a..b + 1),
            _ => Rows::Indices(v),
        }
    }
    /// The zero-based indices.
    pub fn indices(&self) -> Vec<usize> {
        match self {
            Rows::Range(r) => r.clone().collect(),
            Rows::Indices(v) => v.clone(),
        }
    }
    pub fn len(&self) -> usize {
        match self {
            Rows::Range(r) => r.len(),
            Rows::Indices(v) => v.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// What an owner's part of a quantity is: the quantity and the rows of the axes selected (each
/// axis once, ascending; the other axes whole). No axes: the whole array.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Selection {
    pub quantity: QuantityId,
    pub axes: Vec<(usize, Rows)>,
}

/// An owner's part of a quantity: its dimensions and elements (C order, the quantity's type).
#[derive(Clone, Debug, PartialEq)]
pub struct Selected {
    pub quantity: QuantityId,
    pub dims: Vec<usize>,
    pub data: QuantityData,
}

// the position of a named axis of a quantity
fn axis_of(q: QuantityId, name: &str) -> Result<usize> {
    q.dims()
        .split(',')
        .position(|a| a == name)
        .ok_or_else(|| Error::internal(format!("{} has no {name} axis", q.name())))
}

impl Results {
    // the pressure surfaces and the tanks of each body of the model solved (`ErrorKind::State`
    // without a model)
    fn solved_bodies(&self) -> Result<Vec<(usize, usize)>> {
        let doc = self.solved_document().ok_or_else(|| Error::state("these results hold no model"))?;
        Ok(doc
            .root()
            .children("bodies")
            .iter()
            .map(|b| (b.children("pressure_surfaces").len(), b.children("tanks").len()))
            .collect())
    }
    // the number of lids of the model solved
    fn solved_lids(&self) -> Result<usize> {
        let doc = self.solved_document().ok_or_else(|| Error::state("these results hold no model"))?;
        Ok(doc.root().children("lids").len())
    }
    // the rows of an ids quantity equal to id (none without the quantity)
    fn rows_of_id(&self, ids: QuantityId, id: i64) -> Result<Vec<usize>> {
        if !self.has(ids) {
            return Ok(Vec::new());
        }
        Ok(self.i32s(ids)?.iter().enumerate().filter(|(_, x)| i64::from(**x) == id).map(|(k, _)| k).collect())
    }

    /// The zero-based indices of a body's coordinates on the coordinate axis (by the `coordinates`
    /// quantity) and of its radiation columns on the radiation-column axis (the columns whose
    /// `radiation_mode_ids` is one of its coordinates); none of either without those quantities.
    pub fn body_columns(&self, body: usize) -> Result<(Vec<usize>, Vec<usize>)> {
        let d: Vec<usize> = if self.has(QuantityId::Coordinates) {
            let c = self.i32s(QuantityId::Coordinates)?;
            c.chunks_exact(4).enumerate().filter(|(_, r)| r[0] as i64 == body as i64).map(|(i, _)| i).collect()
        } else {
            Vec::new()
        };
        let k: Vec<usize> = if self.has(QuantityId::RadiationModeIds) {
            let ids = self.i32s(QuantityId::RadiationModeIds)?;
            ids.iter()
                .enumerate()
                .filter(|(_, j)| **j >= 1 && d.binary_search(&((**j - 1) as usize)).is_ok())
                .map(|(k, _)| k)
                .collect()
        } else {
            Vec::new()
        };
        Ok((d, k))
    }

    /// What the view of a path selects for an owner (reference/results §1): `body` the zero-based
    /// body of the model solved (None: the results), `owner` the zero-based lid of a "lids" path,
    /// pressure surface of the body of a "pressure_surfaces" path or tank of the body of a "tanks"
    /// path (None for the others). None when the results do not have the quantity, a body has no
    /// part of it (`Owner::Global`) or have no `body_points_ids` (`Owner::BodyPoints`). `ErrorKind::Value` for a path of no view,
    /// `ErrorKind::Argument` for a body or owner out of range, missing or not taken.
    pub fn selection(&self, path: &str, body: Option<usize>, owner: Option<usize>) -> Result<Option<Selection>> {
        let v = result_view(path)?;
        let q = v.quantity;
        let surfaces = match body {
            Some(b) => {
                if v.owner == Owner::Lid {
                    return Err(Error::argument(format!("{path} is of a lid, not of a body")));
                }
                let per_body = self.solved_bodies()?;
                if b >= per_body.len() {
                    return Err(Error::argument(format!(
                        "body {b} does not exist ({} in these results)",
                        per_body.len()
                    )));
                }
                per_body
            }
            None => Vec::new(),
        };
        let takes_owner = matches!(v.owner, Owner::Lid | Owner::PressureSurface | Owner::Tank);
        match (takes_owner, owner) {
            (false, Some(_)) => return Err(Error::argument(format!("{path} takes no owner index"))),
            (true, None) => return Err(Error::argument(format!("{path} is selected by the index of its owner"))),
            _ => {}
        }
        if v.owner == Owner::PressureSurface && body.is_none() {
            return Err(Error::argument(format!("{path} is of a pressure surface of a body: give the body")));
        }
        if v.owner == Owner::Tank && body.is_none() {
            return Err(Error::argument(format!("{path} is of a tank of a body: give the body")));
        }
        // the owner within its list, checked before the availability (as an index of a list)
        let owner_rows = match (v.owner, owner) {
            (Owner::Lid, Some(l)) => {
                let n = self.solved_lids()?;
                if l >= n {
                    return Err(Error::argument(format!("lid {l} does not exist ({n} in these results)")));
                }
                Some(self.rows_of_id(QuantityId::LidsIds, l as i64 + 1)?)
            }
            (Owner::PressureSurface, Some(s)) => {
                let b = body.unwrap_or(0);
                let n = surfaces[b].0;
                if s >= n {
                    return Err(Error::argument(format!(
                        "pressure surface {s} of body {b} does not exist ({n} in these results)"
                    )));
                }
                let first: usize = surfaces[..b].iter().map(|c| c.0).sum();
                Some(self.rows_of_id(QuantityId::PressureSurfacesIds, (first + s) as i64 + 1)?)
            }
            (Owner::Tank, Some(t)) => {
                let b = body.unwrap_or(0);
                let n = surfaces[b].1;
                if t >= n {
                    return Err(Error::argument(format!("tank {t} of body {b} does not exist ({n} in these results)")));
                }
                let first: usize = surfaces[..b].iter().map(|c| c.1).sum();
                Some(self.rows_of_id(QuantityId::TanksIds, (first + t) as i64 + 1)?)
            }
            _ => None,
        };
        if !self.has(q) {
            return Ok(None);
        }
        let ndim = q.dims().split(',').count();
        let axes = match (v.owner, body) {
            (Owner::Shared, _) => Vec::new(),
            (Owner::Global, Some(_)) => return Ok(None),
            (Owner::Global | Owner::Columns(_) | Owner::Coordinates | Owner::BodyPoints, None) => Vec::new(),
            (Owner::Columns(n), Some(b)) => vec![(ndim - 1, Rows::Range(n * b..n * (b + 1)))],
            (Owner::Coordinates, Some(b)) => {
                let (d, k) = self.body_columns(b)?;
                q.dims()
                    .split(',')
                    .enumerate()
                    .filter_map(|(axis, name)| match name {
                        "coordinate" => Some((axis, Rows::of(d.clone()))),
                        "radiation_column" => Some((axis, Rows::of(k.clone()))),
                        _ => None,
                    })
                    .collect()
            }
            (Owner::BodyPoints, Some(b)) => {
                if !self.has(QuantityId::BodyPointsIds) {
                    return Ok(None);
                }
                let ids = self.i32s(QuantityId::BodyPointsIds)?;
                let rows =
                    ids.chunks_exact(4).enumerate().filter(|(_, r)| r[0] as i64 == b as i64).map(|(i, _)| i).collect();
                vec![(axis_of(q, "body_point")?, Rows::of(rows))]
            }
            (Owner::Lid, _) => vec![(axis_of(q, "lid_point")?, Rows::of(owner_rows.unwrap_or_default()))],
            (Owner::PressureSurface, _) => {
                vec![(axis_of(q, "pressure_point")?, Rows::of(owner_rows.unwrap_or_default()))]
            }
            (Owner::Tank, _) => vec![(axis_of(q, "tank_point")?, Rows::of(owner_rows.unwrap_or_default()))],
        };
        Ok(Some(Selection { quantity: q, axes }))
    }

    /// The part of a quantity the view of a path selects for an owner, as `selection` says, copied:
    /// its dimensions and elements (C order, the quantity's element type).
    pub fn select(&self, path: &str, body: Option<usize>, owner: Option<usize>) -> Result<Option<Selected>> {
        let Some(s) = self.selection(path, body, owner)? else {
            return Ok(None);
        };
        let q = s.quantity;
        let shape = self.shape(q).ok_or_else(|| Error::internal("a quantity selected is not there"))?;
        let size = q.element_type().size();
        let (dims, bytes) = gather(self.bytes(q)?, size, &shape.dims, &s.axes);
        let data = QuantityData::of_bytes(q.element_type(), &bytes)?;
        Ok(Some(Selected { quantity: q, dims, data }))
    }
}

/// The elements (of `size` bytes, C order, dimensions dims) at the rows of the axes given (the
/// others whole): the dimensions and bytes of the part.
fn gather(src: &[u8], size: usize, dims: &[usize], axes: &[(usize, Rows)]) -> (Vec<usize>, Vec<u8>) {
    let n = dims.len();
    let idx: Vec<Vec<usize>> = (0..n)
        .map(|k| match axes.iter().find(|(a, _)| *a == k) {
            Some((_, r)) => r.indices(),
            None => (0..dims[k]).collect(),
        })
        .collect();
    let out_dims: Vec<usize> = idx.iter().map(|v| v.len()).collect();
    let count: usize = out_dims.iter().product();
    let mut out = Vec::with_capacity(count * size);
    if count == 0 {
        return (out_dims, out);
    }
    if n == 0 {
        out.extend_from_slice(&src[..size]);
        return (out_dims, out);
    }
    // the strides of the source (elements)
    let mut stride = vec![1usize; n];
    for k in (0..n - 1).rev() {
        stride[k] = stride[k + 1] * dims[k + 1];
    }
    let last = &idx[n - 1];
    let whole_last = last.len() == dims[n - 1];
    let mut at = vec![0usize; n - 1]; // the position among the selected rows of the outer axes
    loop {
        let base: usize = (0..n - 1).map(|k| idx[k][at[k]] * stride[k]).sum();
        if whole_last {
            out.extend_from_slice(&src[base * size..(base + dims[n - 1]) * size]);
        } else {
            for &j in last {
                out.extend_from_slice(&src[(base + j) * size..(base + j + 1) * size]);
            }
        }
        // the next outer position (odometer)
        let mut k = n - 1;
        loop {
            if k == 0 {
                return (out_dims, out);
            }
            k -= 1;
            at[k] += 1;
            if at[k] < idx[k].len() {
                break;
            }
            at[k] = 0;
        }
    }
}
