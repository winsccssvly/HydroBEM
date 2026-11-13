//! The geometry file `.hbg` (reference/inputs-meshes §5): one mesh as a JSON text, its members the items of the
//! mesh object and of its patches (by their names in the schema), with the format name and
//! version, the patch names and the discretization per patch, as rows or from a panel size.
//! Read into a mesh object through the item checks of the model; written from one.

use crate::document::{DType, ItemDef, ItemKind, ItemValue, Obj, ObjMut};
use crate::document_schema::schema_type;
use crate::error::{Error, Result};
use crate::gdf::{PatchOptions, panel_size_discretization};
use crate::json::{self, Json};
use crate::mesh::Mesh;
use crate::mesh_desc::{MeshKind, PatchDiscretization};

/// The format name and the version this engine reads and writes.
pub const HBG_FORMAT: &str = "hbg";
pub const HBG_VERSION: i64 = 3;

// the items of a mesh the engine sets (where a mesh was read from, a trim), never given in a file
const ENGINE_ITEMS: [&str; 8] = [
    "source_file",
    "source_spl",
    "file_gravity",
    "source",
    "trim",
    "trim_elements",
    "parameter_ranges",
    "source_elements",
];

// the members of the file that are not items
const TOP: [&str; 10] = [
    "format",
    "version",
    "kind",
    "info",
    "patches",
    "discretization",
    "panel_size",
    "spline_order",
    "outer_quadrature",
    "inner_quadrature",
];

fn integral(x: f64) -> Option<i64> {
    if x.fract() == 0.0 && x.abs() < 9.007_199_254_740_992e15 { Some(x as i64) } else { None }
}

// the shape and the leaves of a rectangular nested array (C order)
fn nested<'a>(
    j: &'a Json,
    shape: &mut Vec<i64>,
    leaves: &mut Vec<&'a Json>,
    depth: usize,
) -> std::result::Result<(), String> {
    match j {
        Json::Array(a) => {
            if shape.len() == depth {
                if !leaves.is_empty() {
                    // values came first at a shallower level
                    return Err("an array mixes values and arrays at one level".into());
                }
                shape.push(a.len() as i64);
            } else if shape[depth] != a.len() as i64 {
                return Err("the rows of an array must have equal lengths".into());
            }
            for x in a {
                nested(x, shape, leaves, depth + 1)?;
            }
            Ok(())
        }
        _ => {
            if shape.len() != depth {
                return Err("an array mixes values and arrays at one level".into());
            }
            leaves.push(j);
            Ok(())
        }
    }
}

/// The item value of a member for the item definition d.
fn item_value(d: &ItemDef, j: &Json) -> std::result::Result<ItemValue, String> {
    let want = |what: &str| format!("must be {what}, got {}", j.kind());
    Ok(match d.kind {
        ItemKind::Bool => match j {
            Json::Bool(b) => ItemValue::Bool(*b),
            _ => return Err(want("a boolean")),
        },
        ItemKind::Int | ItemKind::Real => match j {
            Json::Number(x, _) => {
                if d.kind == ItemKind::Int {
                    ItemValue::Int(integral(*x).ok_or_else(|| format!("must be an integer, got {}", py(*x)))?)
                } else {
                    ItemValue::Real(*x)
                }
            }
            Json::String(s) if !d.choices.is_empty() => ItemValue::Name(s.clone()),
            _ => return Err(want("a number")),
        },
        ItemKind::Text | ItemKind::Choice => match j {
            Json::String(s) => ItemValue::Text(s.clone()),
            _ => return Err(want("a string")),
        },
        ItemKind::Array => {
            let mut leaves = Vec::new();
            let mut shape = Vec::new();
            nested(j, &mut shape, &mut leaves, 0)?;
            if !matches!(j, Json::Array(_)) {
                return Err(want("an array"));
            }
            let (mut reals, mut ints, mut texts) = (Vec::new(), Vec::new(), Vec::new());
            for x in leaves {
                match (d.dtype, x) {
                    (DType::Real, Json::Number(y, _)) => reals.push(*y),
                    (DType::Int, Json::Number(y, _)) => {
                        ints.push(integral(*y).ok_or_else(|| format!("must hold integers, got {}", py(*y)))?)
                    }
                    (DType::Text, Json::String(s)) => texts.push(s.clone()),
                    (DType::Text, _) => return Err("must hold strings".into()),
                    _ => return Err("must hold numbers".into()),
                }
            }
            match d.dtype {
                DType::Real => ItemValue::Reals { shape, data: reals },
                DType::Int => ItemValue::Ints { shape, data: ints },
                DType::Text => ItemValue::Texts { shape, data: texts },
            }
        }
        _ => return Err("is not given in a geometry file".into()),
    })
}

fn py(x: f64) -> String {
    crate::text::py_repr(x)
}

/// The JSON value of an item value.
fn json_of(v: &ItemValue, d: &ItemDef) -> Json {
    if v.is_array() {
        // rebuild the nesting from the shape
        fn build(v: &ItemValue, dims: &[i64], at: &mut usize) -> Json {
            if dims.is_empty() {
                let k = *at;
                *at += 1;
                return match v.dtype() {
                    DType::Real => Json::Number(v.reals()[k], false),
                    DType::Int => Json::Number(v.ints()[k] as f64, true),
                    DType::Text => Json::String(v.texts()[k].clone()),
                };
            }
            Json::Array((0..dims[0]).map(|_| build(v, &dims[1..], at)).collect())
        }
        let mut at = 0;
        return build(v, v.shape(), &mut at);
    }
    if v.is_name() {
        return Json::String(v.text().to_string());
    }
    match d.kind {
        ItemKind::Bool => Json::Bool(v.boolean()),
        ItemKind::Int => Json::Number(v.integer() as f64, true),
        ItemKind::Real => Json::Number(v.real(), false),
        _ => Json::String(v.text().to_string()),
    }
}

// a member as a discretization row (nu, nv, ku, kv, iquo, iqui)
fn row(j: &Json, at: &str) -> Result<PatchDiscretization> {
    let bad = || Error::value(format!("{at}discretization: six integers (nu, nv, ku, kv, iquo, iqui)"));
    let Json::Array(a) = j else { return Err(bad()) };
    let mut r = [0i32; 6];
    if a.len() != 6 {
        return Err(bad());
    }
    for (k, x) in a.iter().enumerate() {
        let Json::Number(y, _) = x else { return Err(bad()) };
        r[k] = integral(*y).filter(|v| v.abs() < i32::MAX as i64).ok_or_else(bad)? as i32;
    }
    Ok(PatchDiscretization { nu: r[0], nv: r[1], ku: r[2], kv: r[3], iquo: r[4], iqui: r[5] })
}

fn rows_value(rows: &[PatchDiscretization]) -> ItemValue {
    let mut ints = Vec::new();
    for q in rows {
        ints.extend_from_slice(&[q.nu as i64, q.nv as i64, q.ku as i64, q.kv as i64, q.iquo as i64, q.iqui as i64]);
    }
    ItemValue::Ints { shape: vec![rows.len() as i64, 6], data: ints }
}

// an integer of lo..=hi (the rules of the discretization of a patch: the potential order 2..6, the
// Gauss orders >= 1, here at most 64)
fn order(j: &Json, name: &str, at: &str, lo: i64, hi: i64) -> Result<i32> {
    match j {
        Json::Number(x, _) if integral(*x).is_some_and(|v| (lo..=hi).contains(&v)) => Ok(*x as i32),
        _ => Err(Error::value(format!("{at}{name} must be an integer from {lo} to {hi}"))),
    }
}

/// Fills an empty mesh object from the text of a geometry file. at: the prefix of the messages
/// (the path); opt: a discretization or panel size given with the reader (only when the file has
/// none); mesh: the engine's mesh of the object (for a panel size).
pub fn read_hbg(
    m: &mut ObjMut,
    text: &str,
    at: &str,
    opt: &PatchOptions,
    mesh: impl Fn(Obj) -> Result<Mesh>,
) -> Result<()> {
    let fail = |e: Error| Error::value(format!("{at}{}", e.message()));
    let v = json::parse(text).map_err(fail)?;
    let Json::Object(top) = &v else {
        return Err(Error::value(format!("{at}a geometry file is a JSON object, got {}", v.kind())));
    };
    let get = |k: &str| top.iter().find(|(n, _)| n == k).map(|(_, x)| x);
    match get("format") {
        Some(Json::String(s)) if s == HBG_FORMAT => {}
        _ => return Err(Error::value(format!("{at}the member \"format\" must be \"{HBG_FORMAT}\""))),
    }
    match get("version") {
        Some(Json::Number(x, _)) if integral(*x) == Some(HBG_VERSION) => {}
        Some(Json::Number(x, _)) => {
            let v = integral(*x).map_or_else(|| py(*x), |i| i.to_string());
            return Err(Error::value(format!("{at}version {v}; this HydroBEM reads version {HBG_VERSION}")));
        }
        _ => return Err(Error::value(format!("{at}the member \"version\" must be a number"))),
    }
    let Some(Json::String(given)) = get("kind") else {
        let names = MeshKind::ALL
            .iter()
            .filter(|k| **k != MeshKind::TrimmedPatches)
            .map(|k| k.name())
            .collect::<Vec<_>>()
            .join(", ");
        return Err(Error::value(format!("{at}the member \"kind\" must be a string ({names})")));
    };
    // a text that names no kind is left to the item check
    let kind_text = given.as_str();
    if MeshKind::of(kind_text) == Some(MeshKind::TrimmedPatches) {
        return Err(Error::value(format!("{at}a trimmed mesh is made by the engine from a body's trim, not read")));
    }
    let mesh_type = schema_type("Mesh");
    let set = |m: &mut ObjMut, d: &ItemDef, j: &Json, where_: &str| -> Result<()> {
        let value = item_value(d, j).map_err(|e| Error::value(format!("{at}{where_}{}: {e}", d.name)))?;
        m.set(d.name, value).map_err(|e| Error::value(format!("{at}{where_}{}", e.message())))
    };
    set(m, mesh_type.item("kind").expect("a Mesh has kind"), &Json::String(kind_text.into()), "")?;
    let kind = m.as_obj().mesh_kind();
    let kind_name = kind.name();
    for (k, j) in top {
        if TOP.contains(&k.as_str()) {
            continue;
        }
        let Some(d) = mesh_type.item(k) else {
            return Err(Error::value(format!("{at}unknown member \"{k}\"")));
        };
        if ENGINE_ITEMS.contains(&k.as_str()) {
            return Err(Error::value(format!("{at}\"{k}\" is set by the engine, not given in a geometry file")));
        }
        if !m.as_obj().applies(d) {
            return Err(Error::value(format!("{at}\"{k}\" is not an item of a {kind_name} mesh")));
        }
        set(m, d, j, "")?;
    }
    let patch_type = kind.patch_type();
    let low_order = kind == MeshKind::Panels;
    // the patches: the items of their type, a name, a discretization row
    let mut patch_rows: Vec<Option<PatchDiscretization>> = Vec::new();
    match get("patches") {
        Some(_) if patch_type.is_none() => {
            return Err(Error::value(format!("{at}a {kind_name} mesh has no \"patches\"")));
        }
        None if patch_type.is_some() => {
            return Err(Error::value(format!("{at}a {kind_name} mesh needs \"patches\"")));
        }
        Some(Json::Array(ps)) => {
            let type_ = patch_type.expect("a kind with patches");
            let pt = schema_type(type_);
            for (i, p) in ps.iter().enumerate() {
                let where_ = format!("patch {}: ", i + 1);
                let Json::Object(members) = p else {
                    return Err(Error::value(format!("{at}{where_}a patch is an object, got {}", p.kind())));
                };
                let name = match members.iter().find(|(n, _)| n == "name") {
                    Some((_, Json::String(s))) => s.as_str(),
                    Some(_) => return Err(Error::value(format!("{at}{where_}name must be a string"))),
                    None => "",
                };
                let mut c = m
                    .add("patches", type_, name, -1)
                    .map_err(|e| Error::value(format!("{at}{where_}{}", e.message())))?;
                let mut r = None;
                for (k, j) in members {
                    match k.as_str() {
                        "name" | "info" => {}
                        "type" => {
                            if *j != Json::String(type_.into()) {
                                return Err(Error::value(format!(
                                    "{at}{where_}a {kind_name} mesh has patches of type {type_}"
                                )));
                            }
                        }
                        "discretization" => r = Some(row(j, &format!("{at}{where_}"))?),
                        _ => {
                            let Some(d) = pt.item(k) else {
                                return Err(Error::value(format!("{at}{where_}unknown member \"{k}\" of a {type_}")));
                            };
                            set(&mut c, d, j, &where_)?;
                        }
                    }
                }
                patch_rows.push(r);
            }
        }
        Some(x) => return Err(Error::value(format!("{at}\"patches\" must be an array, got {}", x.kind()))),
        None => {}
    }
    // the discretization: from exactly one of the patch rows, the rows of the file, a panel size
    // of the file, or what the reader was given
    let given_rows = patch_rows.iter().any(|r| r.is_some());
    if given_rows && patch_rows.iter().any(|r| r.is_none()) {
        return Err(Error::value(format!("{at}a discretization for every patch or for none")));
    }
    let top_rows = get("discretization");
    let size = get("panel_size");
    let orders = ["spline_order", "outer_quadrature", "inner_quadrature"];
    let file_ways = given_rows as i32 + top_rows.is_some() as i32 + size.is_some() as i32;
    let reader_ways = opt.has_discretization as i32 + opt.has_panel_size as i32;
    if low_order {
        if file_ways > 0 || reader_ways > 0 || orders.iter().any(|o| get(o).is_some()) {
            return Err(Error::value(format!("{at}a panels mesh has no discretization")));
        }
        return mesh(m.as_obj()).map(|_| ()).map_err(|e| Error::value(format!("{at}{}", e.message())));
    }
    if file_ways + reader_ways != 1 {
        return Err(Error::value(format!(
            "{at}the discretization once: per patch, \"discretization\", \"panel_size\" or with the reader \
             ({} given)",
            file_ways + reader_ways
        )));
    }
    if size.is_none() && orders.iter().any(|o| get(o).is_some()) {
        return Err(Error::value(format!("{at}spline_order and the quadrature orders go with \"panel_size\"")));
    }
    let count = if kind == MeshKind::FlatPatches {
        (m.as_obj().get("vertices").reals().len() / 12) as i64
    } else {
        patch_rows.len() as i64
    };
    let rows: Vec<PatchDiscretization> = if given_rows {
        patch_rows.into_iter().flatten().collect()
    } else if let Some(j) = top_rows {
        // one row for every patch, or one per patch
        match j {
            Json::Array(a) if a.iter().all(|x| matches!(x, Json::Array(_))) => {
                a.iter().map(|x| row(x, at)).collect::<Result<Vec<_>>>()?
            }
            _ => vec![row(j, at)?; count as usize],
        }
    } else if opt.has_discretization {
        if opt.discretization.len() == 1 {
            vec![opt.discretization[0]; count as usize]
        } else {
            opt.discretization.clone()
        }
    } else {
        // a panel size: the edges of the patches measured on the mesh (a placeholder
        // discretization first, so that the mesh can be made)
        let mut o = opt.clone();
        if let Some(s) = size {
            let Json::Number(x, _) = s else {
                return Err(Error::value(format!("{at}panel_size must be a number")));
            };
            o.panel_size = *x;
            o.spline_order = 0;
            o.outer_quadrature = 0;
            o.inner_quadrature = 0;
            if let Some(j) = get("spline_order") {
                o.spline_order = order(j, "spline_order", at, 2, 6)?;
            }
            if let Some(j) = get("outer_quadrature") {
                o.outer_quadrature = order(j, "outer_quadrature", at, 1, 64)?;
            }
            if let Some(j) = get("inner_quadrature") {
                o.inner_quadrature = order(j, "inner_quadrature", at, 1, 64)?;
            }
        }
        if !(o.panel_size.is_finite() && o.panel_size > 0.0) {
            return Err(Error::value(format!("{at}panel_size must be a positive number")));
        }
        let one = PatchDiscretization { nu: 1, nv: 1, ku: 2, kv: 2, iquo: 1, iqui: 1 };
        m.set("discretization", rows_value(&vec![one; count as usize]))?;
        let made = mesh(m.as_obj()).map_err(|e| Error::value(format!("{at}{}", e.message())))?;
        panel_size_discretization(&made, &o).map_err(|e| Error::value(format!("{at}{}", e.message())))?
    };
    if rows.len() as i64 != count {
        return Err(Error::value(format!("{at}{} discretization rows for {count} patches", rows.len())));
    }
    m.set("discretization", rows_value(&rows)).map_err(|e| Error::value(format!("{at}{}", e.message())))?;
    // the mesh made here, so that its messages name the file
    mesh(m.as_obj()).map(|_| ()).map_err(|e| Error::value(format!("{at}{}", e.message())))
}

/// The text of the geometry file of a mesh object (every kind but trimmed_patches): the items it
/// has set, in the order of the schema; per patch its name, items and discretization.
pub fn hbg_text(m: Obj) -> Result<String> {
    let kind = m.mesh_kind();
    if kind == MeshKind::TrimmedPatches {
        return Err(Error::value(format!(
            "{}a trimmed mesh is made by the engine from a body's trim; its source mesh is written instead",
            m.prefix()
        )));
    }
    let mut top: Vec<(String, Json)> = vec![
        ("format".into(), Json::String(HBG_FORMAT.into())),
        ("version".into(), Json::Number(HBG_VERSION as f64, true)),
        ("kind".into(), Json::String(kind.name().into())),
    ];
    let mt = schema_type("Mesh");
    for d in &mt.items {
        if d.name == "kind" || d.name == "discretization" || ENGINE_ITEMS.contains(&d.name) {
            continue;
        }
        if m.applies(d) && m.is_set(d.name) {
            top.push((d.name.into(), json_of(&m.get(d.name), d)));
        }
    }
    let disc = m.get("discretization");
    let disc_row = |p: usize| -> Json {
        Json::Array(disc.ints()[6 * p..6 * p + 6].iter().map(|&x| Json::Number(x as f64, true)).collect())
    };
    if kind == MeshKind::FlatPatches {
        let n = disc.ints().len() / 6;
        top.push(("discretization".into(), Json::Array((0..n).map(disc_row).collect())));
    }
    if kind.patch_type().is_some() {
        let mut ps = Vec::new();
        for (p, c) in m.children("patches").iter().enumerate() {
            let mut members: Vec<(String, Json)> = Vec::new();
            if !c.name().is_empty() {
                members.push(("name".into(), Json::String(c.name().into())));
            }
            for d in &c.type_().items {
                if c.is_set(d.name) {
                    members.push((d.name.into(), json_of(&c.get(d.name), d)));
                }
            }
            if 6 * p + 6 <= disc.ints().len() {
                members.push(("discretization".into(), disc_row(p)));
            }
            ps.push(Json::Object(members));
        }
        top.push(("patches".into(), Json::Array(ps)));
    }
    Ok(json::to_text(&Json::Object(top)))
}
