//! The public Rust API (model.rs, results.rs, codes.rs) as a client such as a graphical interface
//! uses it, the crate root only: a floating vertical cylinder built from items, its checks and
//! hydrostatics, a solve with progress reports and a cancel token, the added mass, the state of the
//! model, and a solve job on another thread while the model is read; the model by ownership: the
//! roles of the elements by mesh references, a trimmed body against its trimmed surfaces given as one
//! mesh, the rules of trimmed bodies, the formulation auto, the fixed dampers, a body of
//! sheets only and the memory estimate; the result tree and the selection of an owner's part.

use std::f64::consts::PI;
use std::sync::atomic::{AtomicUsize, Ordering};

use hydrobem_core::{
    CancelToken, Domain, ErrorKind, FrequencyState, Limit, Method, Model, ModelState, ObjectId, Owner, PanelRole,
    Phase, Progress, ProgressAt, QuantityData, QuantityId, Result, Results, Route, Rows, Selection, SolveProgress,
    Stage, Value, WorkerState, result_groups, result_view, result_views,
};

// a quarter of a floating vertical cylinder (radius, draft): n_phi panels round the side in n_z rows,
// and the bottom in n_r rings; four vertices per panel, the normals (q2 - q0) x (q3 - q1) into the fluid
fn quarter_cylinder(radius: f64, draft: f64, n_phi: usize, n_z: usize, n_r: usize) -> Vec<[[f64; 3]; 4]> {
    let phi = |j: usize| 0.5 * PI * j as f64 / n_phi as f64;
    let orient = |q: [[f64; 3]; 4], n: [f64; 3]| {
        let a = [q[2][0] - q[0][0], q[2][1] - q[0][1], q[2][2] - q[0][2]];
        let b = [q[3][0] - q[1][0], q[3][1] - q[1][1], q[3][2] - q[1][2]];
        let c = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
        if c[0] * n[0] + c[1] * n[1] + c[2] * n[2] > 0.0 { q } else { [q[0], q[3], q[2], q[1]] }
    };
    let mut out = Vec::new();
    for i in 0..n_z {
        let (z0, z1) = (-draft * i as f64 / n_z as f64, -draft * (i + 1) as f64 / n_z as f64);
        for j in 0..n_phi {
            let (p0, p1) = (phi(j), phi(j + 1));
            let q = [
                [radius * p0.cos(), radius * p0.sin(), z0],
                [radius * p1.cos(), radius * p1.sin(), z0],
                [radius * p1.cos(), radius * p1.sin(), z1],
                [radius * p0.cos(), radius * p0.sin(), z1],
            ];
            let m = 0.5 * (p0 + p1);
            out.push(orient(q, [m.cos(), m.sin(), 0.0]));
        }
    }
    for i in 0..n_r {
        let (r0, r1) = (radius * (1.0 - i as f64 / n_r as f64), radius * (1.0 - (i + 1) as f64 / n_r as f64));
        for j in 0..n_phi {
            let (p0, p1) = (phi(j), phi(j + 1));
            let mut q = [
                [r0 * p0.cos(), r0 * p0.sin(), -draft],
                [r0 * p1.cos(), r0 * p1.sin(), -draft],
                [r1 * p1.cos(), r1 * p1.sin(), -draft],
                [r1 * p0.cos(), r1 * p0.sin(), -draft],
            ];
            if i + 1 == n_r {
                q[3] = q[2]; // a triangle at the axis
            }
            out.push(orient(q, [0.0, 0.0, -1.0]));
        }
    }
    out
}

// the model: one floating cylinder (displaced mass) at two frequencies and one heading
fn cylinder_model() -> Result<(Model, ObjectId)> {
    let mut m = Model::new();
    let root = m.root();
    let panels = quarter_cylinder(1.0, 0.5, 8, 2, 2);
    let vertices: Vec<f64> = panels.iter().flatten().flatten().copied().collect();
    let mesh = m.add(root, "meshes", None, "cylinder", None)?;
    m.set(mesh, "kind", "panels")?;
    m.set(mesh, "symmetry", "xz_and_yz")?;
    m.set(mesh, "vertices", Value::reals(&[panels.len(), 4, 3], vertices))?;
    let body = m.add(root, "bodies", None, "cylinder", None)?;
    m.set(body, "mesh", "cylinder")?;
    m.set(body, "mass", "displaced")?; // a real item given as one of its choices
    m.set(body, "center_of_gravity", [0.0, 0.0, -0.1])?;
    m.set(body, "radii_of_gyration", [0.6, 0.6, 0.7])?;
    let f = m.object("environment")?;
    m.set(f, "omegas", vec![1.0, 2.0])?;
    m.set(f, "headings", vec![0.0])?;
    Ok((m, body))
}

#[test]
fn a_floating_cylinder_from_items() -> Result<()> {
    let (mut m, body) = cylinder_model()?;
    // the objects and items as a client reads them back
    assert_eq!(m.path(body)?, "bodies/cylinder");
    assert_eq!(m.get(body, "mass")?, Some(Value::Text("displaced".into())));
    assert_eq!(m.get(body, "mesh")?.and_then(|v| v.as_text().map(String::from)), Some("cylinder".into()));
    let info = m.object_info(body)?;
    assert_eq!((info.type_name, info.list.as_str(), info.index), ("Body", "bodies", 0));
    let e = m.set(body, "mass", -1.0).unwrap_err();
    assert_eq!(
        (e.kind(), e.message()),
        (ErrorKind::Value, "bodies/cylinder: mass must be a finite number >= 0, got -1.0")
    );
    assert_eq!(m.set(body, "mass", true).unwrap_err().kind(), ErrorKind::Type);

    m.check()?;
    let c = m.counts()?;
    assert_eq!((c.bodies, c.ordinary_frequencies, c.headings, c.coordinates), (1, 2, 1, 6));
    assert_eq!(m.route()?, "low_order_infinite_open_sea");
    let route = Route::Solve { method: Method::LowOrder, finite_depth: false, domain: Domain::OpenSea };
    assert!(m.route()? == route);
    assert_eq!(format!("{route}"), route.name());
    // the displaced volume of the polygonal cylinder: a little less than pi r^2 T
    let h = m.body_hydrostatics(body)?;
    assert!(h.volumes[2] > 0.95 * PI * 0.5 && h.volumes[2] < PI * 0.5, "{:?}", h.volumes);
    assert_eq!(m.state(), ModelState::Reset);

    // the solve with progress reports
    let mut reports: Vec<Progress> = Vec::new();
    let r = m.solve(|p: &Progress| reports.push(*p))?;
    let shape = r.shape(QuantityId::AddedMass).expect("added mass");
    assert_eq!(shape.dims, vec![2, 6, 6]);
    let a = r.f32s(QuantityId::AddedMass)?;
    let (a33, a11) = (a[2 * 6 + 2], a[0]); // heave, surge at the first frequency
    assert!(a33 > 0.0 && a11 > 0.0, "a33 {a33}, a11 {a11}");
    assert!(r.complex32s(QuantityId::ExcitationDiffraction)?.len() == 2 * 6);
    assert!(r.f32s(QuantityId::ExcitationDiffraction).is_err()); // a complex quantity
    let done: Vec<&Progress> = reports.iter().filter(|p| p.stage == Stage::FrequencyDone).collect();
    assert_eq!(done.len(), 2);
    assert_eq!(done[1].frequencies_done, 2);
    assert!(matches!(done[0].at, ProgressAt::Frequency(_)));
    assert_eq!(m.state(), ModelState::Calculated);

    // the results stay until an input changes
    let f = m.object("environment")?;
    m.set(f, "headings", vec![0.0, 90.0])?;
    assert_eq!(m.state(), ModelState::Reset);
    assert_eq!(m.results().unwrap_err().kind(), ErrorKind::State);
    Ok(())
}

#[test]
fn a_solve_is_cancelled_by_its_token() -> Result<()> {
    let (mut m, _) = cylinder_model()?;
    // a request sent before the solve applies to it, and is cleared when it ends
    let token: CancelToken = m.cancel_token();
    token.cancel();
    let e = m.solve(|_: &Progress| {}).expect_err("cancelled");
    assert_eq!(e.kind(), ErrorKind::Cancelled);
    assert!(!token.is_cancelled());
    assert_eq!(m.state(), ModelState::Reset);
    // a cancel from the progress callback (as from another thread)
    let reports = AtomicUsize::new(0);
    let t = m.cancel_token();
    let e = m
        .solve(|_: &Progress| {
            reports.fetch_add(1, Ordering::SeqCst);
            t.cancel();
        })
        .expect_err("cancelled");
    assert_eq!(e.kind(), ErrorKind::Cancelled);
    assert!(reports.load(Ordering::SeqCst) >= 1);
    // and the model solves again
    m.solve(|_: &Progress| {})?;
    assert_eq!(m.state(), ModelState::Calculated);
    Ok(())
}

#[test]
fn a_solve_job_runs_on_another_thread_while_the_model_is_read() -> Result<()> {
    let (mut m, body) = cylinder_model()?;
    let mut job = m.begin_solve()?;
    assert_eq!(m.state(), ModelState::Calculating);
    let outcome = std::thread::scope(|s| {
        let worker = s.spawn(move || {
            let r = job.run(None);
            (job, r)
        });
        // the model is read meanwhile; its inputs cannot change, and the queries wait
        assert_eq!(m.get(body, "mesh").unwrap(), Some(Value::Text("cylinder".into())));
        let e = m.set(body, "mass", 100.0).unwrap_err();
        assert_eq!(e.message(), "the model is calculating: its inputs cannot change until the solve ends");
        assert_eq!(
            m.counts().unwrap_err().message(),
            "the model is calculating: it can be queried when the solve ends"
        );
        worker.join().expect("the solve thread")
    });
    let (job, r) = outcome;
    let results = m.finish_solve(job, r)?;
    assert!(results.has(QuantityId::AddedMass) && results.has(QuantityId::HydrostaticsVolume));
    assert_eq!(results.document().map(|d| d.root().children("bodies").len()), Some(1));
    assert_eq!(m.state(), ModelState::Calculated);
    Ok(())
}

// ---------------------------------------------------------------- the model by ownership

// a quarter disc (radius) at depth z of n_r rings and n_phi panels each, the normals up
fn quarter_disc(radius: f64, z: f64, n_phi: usize, n_r: usize) -> Vec<[[f64; 3]; 4]> {
    let phi = |j: usize| 0.5 * PI * j as f64 / n_phi as f64;
    let mut out = Vec::new();
    for i in 0..n_r {
        let (r0, r1) = (radius * (1.0 - i as f64 / n_r as f64), radius * (1.0 - (i + 1) as f64 / n_r as f64));
        for j in 0..n_phi {
            let (p0, p1) = (phi(j), phi(j + 1));
            let mut q = [
                [r0 * p0.cos(), r0 * p0.sin(), z],
                [r1 * p0.cos(), r1 * p0.sin(), z],
                [r1 * p1.cos(), r1 * p1.sin(), z],
                [r0 * p1.cos(), r0 * p1.sin(), z],
            ];
            if i + 1 == n_r {
                q[2] = q[1]; // a triangle at the axis
            }
            out.push(q);
        }
    }
    out
}

// a quarter annulus at depth z between the radii r0 < r1, the normals up
fn quarter_annulus(r0: f64, r1: f64, z: f64, n_phi: usize) -> Vec<[[f64; 3]; 4]> {
    let phi = |j: usize| 0.5 * PI * j as f64 / n_phi as f64;
    (0..n_phi)
        .map(|j| {
            let (p0, p1) = (phi(j), phi(j + 1));
            [
                [r1 * p0.cos(), r1 * p0.sin(), z],
                [r0 * p0.cos(), r0 * p0.sin(), z],
                [r0 * p1.cos(), r0 * p1.sin(), z],
                [r1 * p1.cos(), r1 * p1.sin(), z],
            ]
        })
        .collect()
}

// a low-order mesh of the model with the symmetry planes x = 0 and y = 0
fn add_mesh(m: &mut Model, name: &str, panels: &[[[f64; 3]; 4]]) -> Result<ObjectId> {
    let root = m.root();
    let mesh = m.add(root, "meshes", None, name, None)?;
    m.set(mesh, "symmetry", "xz_and_yz")?;
    let v: Vec<f64> = panels.iter().flatten().flatten().copied().collect();
    m.set(mesh, "vertices", Value::reals(&[panels.len(), 4, 3], v))?;
    Ok(mesh)
}

// the mesh of a surface and the ranges of its stored elements
fn set_mesh(m: &mut Model, o: ObjectId, mesh: &str, ranges: &[[i64; 2]]) -> Result<()> {
    m.set(o, "mesh", mesh)?;
    m.set(o, "ranges", Value::ranges(ranges))
}

fn waves(m: &mut Model, omegas: Vec<f64>, headings: Vec<f64>) -> Result<()> {
    let f = m.object("environment")?;
    m.set(f, "omegas", omegas)?;
    m.set(f, "headings", headings)
}

fn message<T: std::fmt::Debug>(r: Result<T>) -> String {
    r.expect_err("an error").message().to_string()
}

#[test]
fn mesh_references_give_the_roles_of_the_records() -> Result<()> {
    let hull = quarter_cylinder(1.0, 0.5, 8, 2, 2);
    let plate = quarter_annulus(1.2, 1.8, -0.3, 4);
    let (nh, np) = (hull.len() as i64, plate.len() as i64);
    let mut m = Model::new();
    add_mesh(&mut m, "spar", &[hull.clone(), plate.clone()].concat())?;
    let root = m.root();
    let b = m.add(root, "bodies", None, "spar", None)?;
    waves(&mut m, vec![1.0], vec![0.0])?;
    // the whole mesh is the wetted surface: a dipole of its elements has two roles
    m.set(b, "mesh", "spar")?;
    let d = m.add(b, "dipoles", None, "", None)?;
    set_mesh(&mut m, d, "spar", &[[nh + 1, nh + np]])?;
    assert_eq!(
        message(m.check()),
        format!(
            "bodies/spar: element {} of the mesh 'spar' is taken by both the wetted surface (mesh) and dipoles 1 (an \
             element has one role in a body)",
            nh + 1
        )
    );
    set_mesh(&mut m, b, "spar", &[[1, nh]])?;
    assert_eq!(m.get(b, "ranges")?, Some(Value::ranges(&[[1, nh]])));
    m.check()?;
    let c = m.body_counts(b)?;
    assert_eq!((c.stored, c.wetted, c.dipole), ((nh + np) as usize, nh as usize, np as usize));
    let sheets = m.body_panels(b, &["dipole"], false)?;
    assert_eq!(sheets.len(), np as usize);
    // a range beyond the stored elements
    set_mesh(&mut m, d, "spar", &[[nh + 1, nh + np + 1]])?;
    assert_eq!(
        message(m.check()),
        format!(
            "bodies/spar: dipoles 1: the element range ({}, {}) is beyond the mesh 'spar' ({} elements)",
            nh + 1,
            nh + np + 1,
            nh + np
        )
    );
    // elements no surface takes are not the body's: its mesh holds the hull elements only
    m.remove(d)?;
    m.check()?;
    // the stored elements of the mesh, the elements the body uses by their roles
    let c = m.body_counts(b)?;
    assert_eq!((c.stored, c.wetted, c.dipole), ((nh + np) as usize, nh as usize, 0));
    // the plate as a dipole damper of the body (porous, in its forces)
    let dd = m.add(b, "dipole_dampers", None, "", None)?;
    set_mesh(&mut m, dd, "spar", &[[nh + 1, nh + np]])?;
    m.set(dd, "damping_factor", 0.5)?;
    m.check()?;
    let roles = m.body_panels(b, &["dipole_damper"], false)?;
    assert_eq!(roles.len(), np as usize);
    // several roles at once; the words of the list names and the old role words are refused
    let both = m.body_panels(b, &["wetted", "dipole_damper"], false)?;
    assert_eq!(both.len(), nh as usize + np as usize);
    for old in ["hull", "dipole_dampers", "damper", "body", "wet"] {
        let e = m.body_panels(b, &[old], false).unwrap_err();
        assert_eq!(e.kind(), ErrorKind::Value);
        assert!(e.message().starts_with("role must be one of wetted, interior_surface, dipole,"), "{}", e.message());
    }
    // the elements of no body are not a body's
    let e = m.body_panels(b, &["fixed_damper"], false).unwrap_err();
    assert!(e.message().contains("not of a body"), "{}", e.message());
    Ok(())
}

// a hull above the free surface (a cylinder of draft 0.6 and freeboard 0.4) with a heave plate below
fn trimmed_spar(m: &mut Model) -> Result<(ObjectId, i64, i64)> {
    let mut hull = quarter_cylinder(1.0, 1.0, 8, 5, 2);
    for q in hull.iter_mut() {
        for v in q.iter_mut() {
            v[2] += 0.4;
        }
    }
    let plate = quarter_disc(1.5, -0.8, 4, 2);
    let mesh = add_mesh(m, "hull", &[hull.clone(), plate.clone()].concat())?;
    waves(m, vec![1.0, 1.6], vec![0.0, 45.0])?;
    Ok((mesh, hull.len() as i64, plate.len() as i64))
}

#[test]
fn a_trimmed_body_is_its_trimmed_surfaces_given_as_one_mesh() -> Result<()> {
    let trim = [-0.05, 2.0, 1.0];
    // the trim on the body: the engine trims the mesh and maps the ranges
    let mut a = Model::new();
    let (_, nh, np) = trimmed_spar(&mut a)?;
    let root = a.root();
    let b = a.add(root, "bodies", None, "spar", None)?;
    set_mesh(&mut a, b, "hull", &[[1, nh]])?;
    a.set(b, "trim", trim)?;
    let d = a.add(b, "dipoles", None, "", None)?;
    set_mesh(&mut a, d, "hull", &[[nh + 1, nh + np]])?;
    // the surface of each role as the engine uses it, added to the meshes
    let wet = a.body_trimmed_mesh(b, "spar_wet", "wetted")?;
    let sheet = a.body_trimmed_mesh(b, "spar_sheet", "dipole")?;
    let vertices = |m: &Model, o: ObjectId| -> Result<Vec<f64>> {
        Ok(m.get(o, "vertices")?.expect("vertices").as_reals().expect("reals").to_vec())
    };
    let (vw, vs) = (vertices(&a, wet)?, vertices(&a, sheet)?);
    assert_eq!(vw.len() / 12, a.body_counts(b)?.wetted);
    // the same surfaces as one mesh of a model without trim: the elements in the order of the body's
    // (the panels of the body by element, each of its role), each role by its ranges
    let stored = a.body_panels(b, &[], false)?;
    let (mut all, mut ranges) = (Vec::new(), [Vec::<[i64; 2]>::new(), Vec::new()]);
    let (mut iw, mut is) = (0, 0);
    for (k, ro) in stored.role.iter().enumerate() {
        let (side, src, i) =
            if *ro == hydrobem_core::PanelRole::Dipole { (1, &vs, &mut is) } else { (0, &vw, &mut iw) };
        all.extend_from_slice(&src[12 * *i..12 * *i + 12]);
        *i += 1;
        let r = k as i64 + 1;
        match ranges[side].last_mut() {
            Some(last) if last[1] + 1 == r => last[1] = r,
            _ => ranges[side].push([r, r]),
        }
    }
    let mut t = Model::new();
    waves(&mut t, vec![1.0, 1.6], vec![0.0, 45.0])?;
    let root = t.root();
    let tm = t.add(root, "meshes", None, "hull_trimmed", None)?;
    t.set(tm, "symmetry", a.get(wet, "symmetry")?.expect("symmetry"))?;
    t.set(tm, "vertices", Value::reals(&[all.len() / 12, 4, 3], all))?;
    let tb = t.add(root, "bodies", None, "spar", None)?;
    set_mesh(&mut t, tb, "hull_trimmed", &ranges[0])?;
    let d = t.add(tb, "dipoles", None, "", None)?;
    set_mesh(&mut t, d, "hull_trimmed", &ranges[1])?;
    // element for element: the same panels, the same results bit for bit
    let pa = a.body_panels(b, &[], true)?;
    let pt = t.body_panels(tb, &[], true)?;
    assert_eq!(pa.vertices, pt.vertices);
    assert_eq!(pa.role, pt.role);
    assert_eq!(pa.areas, pt.areas);
    let ra = a.solve(|_: &Progress| {})?.clone();
    let rt = t.solve(|_: &Progress| {})?;
    for q in [QuantityId::AddedMass, QuantityId::RadiationDamping, QuantityId::ExcitationDiffraction] {
        assert_eq!(ra.bytes(q)?, rt.bytes(q)?, "{q:?}");
    }
    Ok(())
}

#[test]
fn the_rules_of_trimmed_bodies_lids_and_surfaces() -> Result<()> {
    let mut m = Model::new();
    let (_, nh, _) = trimmed_spar(&mut m)?;
    let root = m.root();
    let b = m.add(root, "bodies", None, "spar", None)?;
    set_mesh(&mut m, b, "hull", &[[1, nh]])?;
    m.set(b, "trim", [0.0, 0.0, 0.0])?;
    // a given interior surface lies on the waterline of the untrimmed hull
    m.set(b, "interior_surface", "mesh")?;
    assert_eq!(message(m.check()), "bodies/spar: interior_surface = 'mesh' needs interior_surface_mesh");
    m.set(b, "interior_surface_mesh", "hull")?;
    assert_eq!(
        message(m.check()),
        "bodies/spar: a trimmed body has a generated interior surface (given elements lie on the waterline of the \
         untrimmed hull): set interior_surface to a generated one"
    );
    // the items of a method not chosen are kept unused, not checked
    m.set(b, "interior_surface", "graded")?;
    assert!(m.item_info(b, "interior_surface_mesh")?.is_set);
    m.check()?;
    // a lid from the elements of a trimmed body
    let l = m.add(root, "lids", None, "moon", None)?;
    set_mesh(&mut m, l, "hull", &[[1, 1]])?;
    m.set(l, "placed_with", "spar")?;
    assert_eq!(
        message(m.check()),
        "lids/moon: a lid from the elements of the trimmed body 'spar' is not available (give the lid as a mesh \
         of its own)"
    );
    m.remove(l)?;
    // a pressure surface of a trimmed body
    let p = m.add(b, "pressure_surfaces", None, "", None)?;
    set_mesh(&mut m, p, "hull", &[[nh + 1, nh + 1]])?;
    m.add(p, "pressure_modes", None, "p", None)?;
    assert_eq!(
        message(m.check()),
        "bodies/spar/pressure_surfaces/1: a pressure surface of a trimmed body is not available"
    );
    Ok(())
}

#[test]
fn auto_formulation_is_total_with_a_lid() -> Result<()> {
    // a floating cylinder with a damping lid around it
    let mut m = Model::new();
    add_mesh(&mut m, "hull", &quarter_cylinder(1.0, 0.5, 8, 2, 2))?;
    add_mesh(&mut m, "ring", &quarter_annulus(1.5, 2.5, 0.0, 8))?;
    let root = m.root();
    let b = m.add(root, "bodies", None, "cyl", None)?;
    m.set(b, "mesh", "hull")?;
    waves(&mut m, vec![1.5], vec![0.0])?;
    m.check()?; // auto: scattered without a lid
    let l = m.add(root, "lids", None, "ring", None)?;
    m.set(l, "mesh", "ring")?;
    m.set(l, "damping_factor", 0.1)?;
    m.check()?; // auto: total with a lid
    let auto = m.solve(|_: &Progress| {})?.clone();
    let first = m.object("first_order")?;
    m.set(first, "diffraction_formulation", "scattered")?;
    m.check()?; // the low order also solves the scattered potential with a lid
    m.set(first, "diffraction_formulation", "total")?;
    let total = m.solve(|_: &Progress| {})?;
    assert_eq!(auto.bytes(QuantityId::ExcitationDiffraction)?, total.bytes(QuantityId::ExcitationDiffraction)?);
    Ok(())
}

// the added mass and damping of a floating cylinder with a porous ring around it, at rest (a damper
// of the water); the ring as elements of the body's mesh (the body at the origin) or as a mesh of its
// own in global coordinates
fn cylinder_with_a_ring(own_mesh: bool) -> Result<Results> {
    let hull = quarter_cylinder(1.0, 0.5, 8, 3, 3);
    let ring = quarter_annulus(1.3, 2.0, -0.25, 8);
    let (nh, nr) = (hull.len() as i64, ring.len() as i64);
    let mut m = Model::new();
    if own_mesh {
        add_mesh(&mut m, "hull", &hull)?;
        add_mesh(&mut m, "ring", &ring)?;
    } else {
        add_mesh(&mut m, "hull", &[hull, ring.clone()].concat())?;
    }
    let root = m.root();
    let b = m.add(root, "bodies", None, "cyl", None)?;
    if own_mesh {
        m.set(b, "mesh", "hull")?;
    } else {
        set_mesh(&mut m, b, "hull", &[[1, nh]])?;
    }
    let d = m.add(root, "fixed_dampers", None, "", None)?;
    if own_mesh {
        m.set(d, "mesh", "ring")?;
    } else {
        set_mesh(&mut m, d, "hull", &[[nh + 1, nh + nr]])?;
    }
    m.set(d, "damping_factor", 0.4)?;
    waves(&mut m, vec![0.8, 1.4, 2.0], vec![0.0])?;
    let r = m.solve(|_: &Progress| {})?.clone();
    let roles = m.panels(&["fixed_damper"], false)?;
    assert_eq!(roles.len(), ring.len());
    Ok(r)
}

#[test]
fn dampers_of_the_water_are_at_rest() -> Result<()> {
    let attached = cylinder_with_a_ring(false)?;
    let own = cylinder_with_a_ring(true)?;
    let (a, b) = (attached.f32s(QuantityId::AddedMass)?, attached.f32s(QuantityId::RadiationDamping)?);
    let peak = |v: &[f32]| v.iter().fold(0.0f32, |s, x| s.max(x.abs()));
    let (pa, pb) = (peak(a), peak(b));
    for f in 0..3 {
        for i in 0..6 {
            // the damping absorbs energy
            assert!(b[36 * f + 7 * i] >= -1e-4 * pb, "B[{i}{i}] at frequency {f}: {}", b[36 * f + 7 * i]);
            // symmetric to the discretization of the sharp-edged cylinder (about 1e-3 of the peak)
            for j in 0..6 {
                let (x, y) = (36 * f + 6 * i + j, 36 * f + 6 * j + i);
                assert!((a[x] - a[y]).abs() <= 3e-3 * pa, "A not symmetric at {i}, {j}: {} {} of {pa}", a[x], a[y]);
                assert!((b[x] - b[y]).abs() <= 3e-3 * pb, "B not symmetric at {i}, {j}: {} {} of {pb}", b[x], b[y]);
            }
        }
    }
    // the ring in the body's frame and as a sheet of its own: the same problem
    let (a2, b2) = (own.f32s(QuantityId::AddedMass)?, own.f32s(QuantityId::RadiationDamping)?);
    for k in 0..a.len() {
        assert!((a[k] - a2[k]).abs() <= 1e-4 * pa, "A[{k}]: {} vs {}", a[k], a2[k]);
        assert!((b[k] - b2[k]).abs() <= 1e-4 * pb, "B[{k}]: {} vs {}", b[k], b2[k]);
    }
    // without the damper the damping differs: the ring is in the problem
    let mut bare = Model::new();
    add_mesh(&mut bare, "hull", &quarter_cylinder(1.0, 0.5, 8, 3, 3))?;
    let root = bare.root();
    let body = bare.add(root, "bodies", None, "cyl", None)?;
    bare.set(body, "mesh", "hull")?;
    waves(&mut bare, vec![0.8, 1.4, 2.0], vec![0.0])?;
    let r = bare.solve(|_: &Progress| {})?;
    let b0 = r.f32s(QuantityId::RadiationDamping)?;
    assert!((b0[0] - b[0]).abs() > 1e-3 * pb);
    Ok(())
}

#[test]
fn the_memory_estimate_has_its_resident_peak_and_commit_charge() -> Result<()> {
    let (mut m, _) = cylinder_model()?;
    let e = m.memory_estimate()?;
    assert_eq!(e.peak, e.statics + e.per_frequency + e.outputs);
    // the commit charge holds the whole packing buffers of the threads, the peak the pages written
    assert!(e.commit >= e.peak && e.peak > 0, "{e:?}");
    let o = m.object("options")?;
    m.set(o, "inner_threads", 4i64)?;
    let four = m.memory_estimate()?;
    assert!(four.per_frequency > e.per_frequency && four.commit - four.peak >= e.commit - e.peak, "{four:?}");
    Ok(())
}

#[test]
fn a_body_of_sheets_only_has_no_wetted_surface() -> Result<()> {
    let mut m = Model::new();
    let root = m.root();
    add_mesh(&mut m, "disc", &quarter_disc(1.0, -2.0, 6, 3))?;
    let b = m.add(root, "bodies", None, "disc", None)?;
    m.set(b, "modes", Value::from(&["heave"][..]))?;
    let out = m.object("first_order")?;
    m.set(out, "diffraction", false)?;
    waves(&mut m, vec![1.0], vec![])?;
    assert_eq!(
        message(m.check()),
        "bodies/disc: the body has no wetted surface, no dipoles and no free-surface plates"
    );
    let d = m.add(b, "dipoles", None, "", None)?;
    m.set(d, "mesh", "disc")?;
    m.check()?;
    let c = m.body_counts(b)?;
    assert_eq!((c.wetted, c.dipole), (0, 18));
    assert!(m.solve(|_: &Progress| {})?.f32s(QuantityId::AddedMass)?[2] > 0.0); // heave, the one column
    Ok(())
}

#[test]
fn the_free_surface_queries_check_the_free_surface_alone() -> Result<()> {
    // the potential QTF's free surface of a fixed cylinder, its items given; the model's rules
    // broken by a frequency rule and by a rule of the orders: the free-surface queries answer as
    // for the valid model, the model's check refuses
    let (mut m, body) = cylinder_model()?;
    m.set(body, "modes", Value::from(Vec::<String>::new()))?;
    let second = m.object("second_order")?;
    m.set(second, "qtf_potential_indirect", true)?;
    m.set(second, "free_surface_inner_radius", 3.0)?;
    m.set(second, "free_surface_panel_size", 0.5)?;
    m.set(second, "outer_circle_segments", 8)?;
    m.check()?;
    // the areas and elements of the panels, the points and weights of the outer circle
    type Answers = (Vec<f64>, Vec<i32>, Vec<[f64; 3]>, Vec<f64>);
    let answers = |m: &mut Model| -> Result<Answers> {
        let p = m.panels(&["free_surface"], true)?;
        let fs = m.free_surface_points("outer_circle")?;
        Ok((p.areas, p.element, fs.points, fs.weights))
    };
    let valid = answers(&mut m)?;
    let full = m.panels(&["wetted", "free_surface"], true)?;
    let rows: Vec<usize> = (0..full.len()).filter(|&i| full.role[i] == PanelRole::FreeSurface).collect();
    assert_eq!(valid.0, rows.iter().map(|&i| full.areas[i]).collect::<Vec<_>>());
    assert_eq!(valid.1, rows.iter().map(|&i| full.element[i]).collect::<Vec<_>>());
    let f = m.object("environment")?;
    m.set(f, "omegas", Vec::<f64>::new())?;
    assert!(message(m.check()).contains("omegas is empty"));
    assert_eq!(answers(&mut m)?, valid);
    m.set(f, "omegas", vec![1.0, 2.0])?;
    let first = m.object("first_order")?;
    m.set(first, "diffraction", false)?;
    assert!(message(m.check()).contains("require the diffraction solution"));
    assert_eq!(answers(&mut m)?, valid);
    // the auto size of the generated panels takes the frequencies: refused without them
    m.set(f, "omegas", Vec::<f64>::new())?;
    m.reset_item(second, "free_surface_panel_size")?;
    let e = message(m.panels(&["free_surface"], true));
    assert!(e.contains("no frequency between 0 and inf") && e.contains("free_surface_panel_size"), "{e}");
    Ok(())
}

#[test]
fn the_queries_of_a_mesh_a_body_and_the_water() -> Result<()> {
    let hull = quarter_cylinder(1.0, 0.5, 8, 2, 2);
    let plate = quarter_annulus(1.2, 1.8, -0.3, 4);
    let (nh, np) = (hull.len() as i64, plate.len() as i64);
    let mut m = Model::new();
    let mesh = add_mesh(&mut m, "spar", &[hull.clone(), plate].concat())?;
    waves(&mut m, vec![1.0], vec![0.0])?;
    // the mesh alone: its geometry and elements in its own coordinates
    let g = m.mesh_geometry(mesh)?;
    // the open hull closed by its waterplane; the plate (a horizontal sheet) adds to the z estimate alone
    assert!(g.volumes[0] > 0.95 * PI * 0.5 && g.volumes[0] < PI * 0.5 && g.volumes[2] > g.volumes[0], "{g:?}");
    assert!((g.volume - g.volumes.iter().sum::<f64>() / 3.0).abs() < 1e-15);
    let rec = m.mesh_elements(mesh)?;
    assert_eq!(rec.len(), (nh + np) as usize);
    let q = m.mesh_geometry_quality(mesh)?;
    assert_eq!(q.volumes, vec![g.volumes]);
    let t = m.mesh_tessellation(mesh, [2, 2], false)?;
    assert_eq!(t.len(), (nh + np) as usize);
    // the body: the hull of the mesh, a named dipole
    let root = m.root();
    let b = m.add(root, "bodies", None, "spar", None)?;
    set_mesh(&mut m, b, "spar", &[[1, nh]])?;
    let d = m.add(b, "dipoles", None, "plate", None)?;
    set_mesh(&mut m, d, "spar", &[[nh + 1, nh + np]])?;
    assert_eq!(m.object("bodies/spar/dipoles/plate")?, d);
    assert_eq!(m.object("bodies/spar/dipoles/1")?, d);
    assert_eq!(m.path(d)?, "bodies/spar/dipoles/plate");
    assert_eq!(message(m.add(b, "dipoles", None, "plate", None)), "bodies/spar: duplicate dipole name 'plate'");
    let h = m.body_hydrostatics(b)?;
    assert!(h.metacentric_height.iter().all(|x| x.is_nan()) && h.displaced_mass > 0.0);
    let p = m.body_panels(b, &["dipole"], false)?;
    assert_eq!(p.len(), np as usize);
    assert_eq!(m.panels(&[], true)?.len(), 4 * (nh + np) as usize);
    assert!(m.panels(&["lid"], true)?.is_empty() && m.panels(&["fixed_damper"], true)?.is_empty());
    assert_eq!(m.body_mesh_quality(b)?.volumes.len(), 1);
    assert_eq!(m.body_tessellation(b, &[], [1, 1], true)?.len(), 4 * (nh + np) as usize);
    Ok(())
}

#[test]
fn the_progress_snapshot_of_a_solve() -> Result<()> {
    let (mut m, _) = cylinder_model()?;
    let f = m.object("environment")?;
    m.set(f, "omegas", vec![0.0, 1.0, 2.0, f64::INFINITY])?;
    let monitor = m.progress_monitor();
    assert_eq!(monitor.snapshot(), SolveProgress::default()); // before the first solve
    // read from the callback (as from another thread): the frequency of a stage report is running
    // with its stage, the others waiting or done; fractions never decrease
    let mut seen: Vec<SolveProgress> = Vec::new();
    let mut stage_of_report = Vec::new();
    let mon = monitor.clone();
    m.solve(|p: &Progress| {
        let s = mon.snapshot();
        if let (ProgressAt::Frequency(fi), Stage::Assembly | Stage::Lu | Stage::Outputs) = (p.at, p.stage) {
            stage_of_report.push((s.frequencies[fi].state, s.frequencies[fi].stage.is_some()));
        }
        seen.push(s);
    })?;
    assert!(stage_of_report.iter().all(|&(st, staged)| st == FrequencyState::Running && staged));
    for (a, b) in seen.iter().zip(seen.iter().skip(1)) {
        assert!(a.running && b.seconds >= a.seconds);
        for (fa, fb) in a.frequencies.iter().zip(&b.frequencies) {
            assert!(fb.fraction >= fa.fraction);
        }
    }
    let end = monitor.snapshot();
    assert!(!end.running && end.done == 4 && end.total == 4 && end.preparation == 1.0);
    assert_eq!(end.limits.iter().map(|l| l.limit).collect::<Vec<_>>(), vec![Limit::Zero, Limit::Infinite]);
    assert!(
        end.frequencies
            .iter()
            .chain(end.limits.iter().map(|l| &l.progress))
            .all(|f| { f.state == FrequencyState::Done && f.fraction == 1.0 && f.stage.is_none() })
    );
    // one frequency worker: every frequency and limit on worker 0, the worker done at the end; the
    // phases in their order, reported once each; no second order
    assert!(end.frequencies.iter().chain(end.limits.iter().map(|l| &l.progress)).all(|f| f.worker == Some(0)));
    assert_eq!(end.phase, Some(Phase::Done));
    assert_eq!(end.workers.len(), 1);
    assert!(end.workers.iter().all(|w| w.state == WorkerState::Done && w.task.is_none()));
    assert!(end.second_order.is_empty());
    let mut phases: Vec<Phase> = Vec::new();
    m.solve(|p: &Progress| {
        if p.stage == Stage::Phase {
            assert_eq!(p.at, ProgressAt::Solve);
            phases.push(p.phase);
        }
    })?;
    assert_eq!(phases, vec![Phase::Limits, Phase::FirstOrder, Phase::Finishing, Phase::Done]);
    let end = monitor.snapshot();
    assert_eq!(m.progress(), end);
    // a solve cancelled in the statics: every frequency and limit cancelled
    let t = m.cancel_token();
    let e = m.solve(|p: &Progress| {
        if p.at == ProgressAt::Statics {
            t.cancel();
        }
    });
    assert_eq!(e.expect_err("cancelled").kind(), ErrorKind::Cancelled);
    let c = monitor.snapshot();
    assert!(!c.running && c.done == 0 && c.total == 4);
    assert!(
        c.frequencies.iter().chain(c.limits.iter().map(|l| &l.progress)).all(|f| f.state == FrequencyState::Cancelled)
    );
    // a job dropped without finish_solve: the solve ended, its frequencies cancelled
    let job = m.begin_solve()?;
    assert!(monitor.snapshot().running);
    drop(job);
    let d = monitor.snapshot();
    assert!(!d.running && d.frequencies.iter().all(|f| f.state == FrequencyState::Cancelled));
    // the hydrostatics alone (two ordinary frequencies given): no frequency work, nothing to report
    let w = m.object("environment")?;
    m.set(w, "omegas", vec![1.0, 2.0])?;
    m.set(w, "headings", Vec::<f64>::new())?;
    let o = m.object("first_order")?;
    m.set(o, "radiation", false)?;
    m.set(o, "diffraction", false)?;
    m.set(o, "motion", false)?;
    m.solve(|_: &Progress| {})?;
    let h = monitor.snapshot();
    assert!(!h.running && h.total == 0 && h.frequencies.is_empty() && h.limits.is_empty());
    Ok(())
}

#[test]
fn a_list_is_cleared_all_or_nothing() -> Result<()> {
    // the meshes are not cleared while a body refers to one of them; the bodies are, and then the
    // meshes
    let (mut m, _) = cylinder_model()?;
    let root = m.root();
    assert!(message(m.clear_list(root, "meshes")).contains("refers to meshes/cylinder"));
    assert_eq!(m.children(root, "meshes")?.len(), 1);
    m.clear_list(root, "bodies")?;
    m.clear_list(root, "meshes")?;
    assert!(m.children(root, "bodies")?.is_empty() && m.children(root, "meshes")?.is_empty());
    // a list that always holds its object is not cleared
    assert!(message(m.clear_list(root, "second_order")).contains("always holds its object"));
    Ok(())
}

// ---------------------------------------------------------------- the result tree

// the four quarters of a quarter mesh (the mirror images reordered to keep the normals)
fn whole(quarter: &[[[f64; 3]; 4]]) -> Vec<[[f64; 3]; 4]> {
    let mut out = Vec::new();
    for (sx, sy) in [(1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)] {
        for q in quarter {
            let mut p = q.map(|v| [sx * v[0], sy * v[1], v[2]]);
            if sx * sy < 0.0 {
                p.reverse();
            }
            out.push(p);
        }
    }
    out
}

// two floating cylinders without symmetry, the second at x = 5 free in heave and pitch only, with
// the body pressure and the motions; with the pressure and momentum drifts, or with a damping lid
// round the first (the drift is not available with a lid)
fn two_cylinders(lid: bool) -> Result<Model> {
    let mut m = Model::new();
    let root = m.root();
    let hull = whole(&quarter_cylinder(1.0, 0.5, 4, 1, 1));
    let ring = whole(&quarter_annulus(1.5, 2.0, 0.0, 4));
    for (name, panels) in [("hull", hull), ("ring", ring)] {
        let mesh = m.add(root, "meshes", None, name, None)?;
        let v: Vec<f64> = panels.iter().flatten().flatten().copied().collect();
        m.set(mesh, "vertices", Value::reals(&[panels.len(), 4, 3], v))?;
    }
    for (name, x) in [("a", 0.0), ("b", 5.0)] {
        let b = m.add(root, "bodies", None, name, None)?;
        m.set(b, "mesh", "hull")?;
        m.set(b, "position", [x, 0.0, 0.0])?;
        m.set(b, "mass", "displaced")?;
        m.set(b, "center_of_gravity", [0.0, 0.0, -0.1])?;
        m.set(b, "radii_of_gyration", [0.6, 0.6, 0.7])?;
        if name == "b" {
            m.set(b, "modes", &["heave", "pitch"][..])?;
        }
    }
    let out = m.object("first_order")?;
    m.set(out, "body_pressure", true)?;
    m.set(out, "motion", true)?;
    if lid {
        let l = m.add(root, "lids", None, "ring", None)?;
        m.set(l, "mesh", "ring")?;
        m.set(l, "damping_factor", 0.1)?;
    } else {
        let second = m.object("second_order")?;
        m.set(second, "drift_pressure_integration", true)?;
        m.set(second, "drift_momentum_conservation", true)?;
    }
    waves(&mut m, vec![1.0, 1.5], vec![0.0, 45.0])?;
    Ok(m)
}

// the bytes of a quantity at the rows of the axes of a selection, the others whole (element by
// element, in C order)
fn part_of(r: &Results, s: &Selection) -> Result<(Vec<usize>, Vec<u8>)> {
    let q = s.quantity;
    let dims = r.shape(q).unwrap().dims;
    let size = q.element_type().size();
    let rows: Vec<Vec<usize>> = (0..dims.len())
        .map(|k| s.axes.iter().find(|(a, _)| *a == k).map_or((0..dims[k]).collect(), |(_, r)| r.indices()))
        .collect();
    let bytes = r.bytes(q)?;
    let mut out = Vec::new();
    let total: usize = rows.iter().map(|v| v.len()).product();
    for flat in 0..total {
        let (mut rest, mut at, mut stride) = (flat, 0usize, 1usize);
        for k in (0..dims.len()).rev() {
            at += rows[k][rest % rows[k].len()] * stride;
            rest /= rows[k].len();
            stride *= dims[k];
        }
        out.extend_from_slice(&bytes[at * size..(at + 1) * size]);
    }
    Ok((rows.iter().map(|v| v.len()).collect(), out))
}

// the bytes of the elements of a part
fn bytes_of(d: &QuantityData) -> Vec<u8> {
    match d {
        QuantityData::Float64(v) => v.iter().flat_map(|x| x.to_ne_bytes()).collect(),
        QuantityData::Float32(v) => v.iter().flat_map(|x| x.to_ne_bytes()).collect(),
        QuantityData::Int32(v) => v.iter().flat_map(|x| x.to_ne_bytes()).collect(),
        QuantityData::Int64(v) => v.iter().flat_map(|x| x.to_ne_bytes()).collect(),
        QuantityData::ComplexF32(v) => {
            v.iter().flat_map(|x| [x.re.to_ne_bytes(), x.im.to_ne_bytes()].concat()).collect()
        }
        QuantityData::ComplexF64(v) => {
            v.iter().flat_map(|x| [x.re.to_ne_bytes(), x.im.to_ne_bytes()].concat()).collect()
        }
    }
}

#[test]
fn the_result_tree_selects_the_part_of_an_owner() -> Result<()> {
    // the tree: every quantity with a group has one view, under a group of the tree
    let views = result_views();
    let groups: Vec<&str> = result_groups().iter().map(|g| g.path).collect();
    assert!(views.len() > 40);
    assert!(views.iter().all(|v| v.quantity.group() == v.path && v.quantity.owner() == v.owner));
    for v in &views {
        match v.path.rsplit_once('.') {
            Some((parent, _)) => assert!(groups.contains(&parent), "{}", v.path),
            None => assert_eq!(v.path, v.quantity.name()), // a quantity read by body
        }
    }
    assert_eq!(result_view("drift.pressure_integration")?.owner, Owner::Coordinates);
    assert_eq!(result_view("drift.pressure_integration_body_axes")?.owner, Owner::Columns(3));
    assert_eq!(result_view("lids.points")?.quantity, QuantityId::Lids);
    assert_eq!(result_view("drift.pressure").unwrap_err().kind(), ErrorKind::Value);

    let mut m = two_cylinders(false)?;
    let r = m.solve(|_: &Progress| {})?.clone();
    // the coordinates of each body, the radiation columns of its coordinates
    let coords = r.i32s(QuantityId::Coordinates)?;
    let ids = r.i32s(QuantityId::RadiationModeIds)?;
    for b in 0..2 {
        let (d, k) = r.body_columns(b)?;
        let want_d: Vec<usize> = (0..coords.len() / 4).filter(|&i| coords[4 * i] as usize == b).collect();
        let want_k: Vec<usize> = (0..ids.len()).filter(|&j| want_d.contains(&(ids[j] as usize - 1))).collect();
        assert_eq!((&d, &k), (&want_d, &want_k));
        assert_eq!((d.len(), k.len()), if b == 0 { (6, 6) } else { (6, 2) });
        let s = r.selection("added_mass", Some(b), None)?.unwrap();
        assert_eq!(s.axes, vec![(1, Rows::of(d.clone())), (2, Rows::of(k.clone()))]);
        assert_eq!(r.select("added_mass", Some(b), None)?.unwrap().dims, vec![2, d.len(), k.len()]);
        // the coordinates of a body on the last axis: 6 b .. 6 b + 6 (six each)
        let s = r.selection("drift.pressure_integration", Some(b), None)?.unwrap();
        assert_eq!(s.axes, vec![(2, Rows::Range(6 * b..6 * b + 6))]);
        // every view of a body is the part of its selection
        let of_body = |o: Owner| matches!(o, Owner::Columns(_) | Owner::Coordinates | Owner::BodyPoints);
        for v in views.iter().filter(|v| of_body(v.owner)) {
            let (Some(s), Some(x)) = (r.selection(v.path, Some(b), None)?, r.select(v.path, Some(b), None)?) else {
                assert!(!r.has(v.quantity), "{}", v.path);
                continue;
            };
            assert_eq!((x.dims, bytes_of(&x.data)), part_of(&r, &s)?, "{}", v.path);
            assert_eq!(x.data.element_type(), v.quantity.element_type());
        }
    }
    // the whole array for the results; none of the momentum-conservation drift for a body
    let all = r.select("drift.pressure_integration", None, None)?.unwrap();
    assert_eq!(all.dims, r.shape(QuantityId::DriftPressureIntegration).unwrap().dims);
    assert_eq!(bytes_of(&all.data), r.bytes(QuantityId::DriftPressureIntegration)?);
    assert!(r.selection("drift.momentum_conservation", None, None)?.unwrap().axes.is_empty());
    assert!(r.selection("drift.momentum_conservation", Some(1), None)?.is_none());
    assert!(r.selection("drift.heading_pairs", Some(1), None)?.unwrap().axes.is_empty());
    assert!(r.selection("field_points.pressure_radiation", None, None)?.is_none()); // not computed
    // the body points of a body: the rows of body_points_ids whose body is it
    let bp = r.i32s(QuantityId::BodyPointsIds)?;
    let rows: Vec<usize> = (0..bp.len() / 4).filter(|&i| bp[4 * i] == 1).collect();
    assert!(!rows.is_empty());
    let s = r.selection("body_points.pressure_diffraction", Some(1), None)?.unwrap();
    assert_eq!(s.axes, vec![(1, Rows::of(rows))]);
    // the arguments: a body or lid out of range, an owner missing or not taken (each refusal by its
    // message: they are all ErrorKind::Argument)
    let refused = |e: Result<Option<Selection>>, message: &str| {
        let e = e.unwrap_err();
        assert_eq!((e.kind(), e.message()), (ErrorKind::Argument, message));
    };
    refused(r.selection("added_mass", Some(2), None), "body 2 does not exist (2 in these results)");
    refused(r.selection("lids.points", None, Some(0)), "lid 0 does not exist (0 in these results)");
    refused(r.selection("lids.points", None, None), "lids.points is selected by the index of its owner");
    refused(r.selection("lids.points", Some(0), Some(0)), "lids.points is of a lid, not of a body");
    refused(r.selection("added_mass", Some(0), Some(0)), "added_mass takes no owner index");
    refused(
        r.selection("pressure_surfaces.points", None, Some(0)),
        "pressure_surfaces.points is of a pressure surface of a body: give the body",
    );
    refused(
        r.selection("pressure_surfaces.points", Some(0), Some(0)),
        "pressure surface 0 of body 0 does not exist (0 in these results)",
    );
    refused(r.selection("tanks.points", None, Some(0)), "tanks.points is of a tank of a body: give the body");
    refused(
        r.selection("tanks.pressure_radiation", Some(1), Some(0)),
        "tank 0 of body 1 does not exist (0 in these results)",
    );
    refused(r.selection("tanks.points", Some(0), None), "tanks.points is selected by the index of its owner");
    // the points of the lid: all of them (one lid), its elevation by the lid-point axis
    let lr = two_cylinders(true)?.solve(|_: &Progress| {})?.clone();
    let lid = lr.select("lids.points", None, Some(0))?.unwrap();
    assert_eq!(lid.dims, lr.shape(QuantityId::Lids).unwrap().dims);
    let s = lr.selection("lids.elevation_radiation", None, Some(0))?.unwrap();
    assert_eq!(s.axes, vec![(1, Rows::Range(0..lid.dims[0]))]);
    refused(lr.selection("lids.points", None, Some(1)), "lid 1 does not exist (1 in these results)");

    // rows that are not contiguous: the body points of two bodies interleaved, in results made
    // from arrays; the copy is the rows in their order
    let mut made = m.new_results()?;
    let n = 4;
    let ids: Vec<i32> = (0..n).flat_map(|i| [(i % 2) as i32, i as i32 + 1, 1, 1]).collect();
    made.put(QuantityId::BodyPointsIds, &[n, 4], QuantityData::Int32(ids))?;
    made.put(QuantityId::BodyPoints, &[n, 3], QuantityData::Float64((0..3 * n).map(|k| k as f64).collect()))?;
    let s = made.selection("body_points.points", Some(0), None)?.unwrap();
    assert_eq!(s.axes, vec![(0, Rows::Indices(vec![0, 2]))]);
    let x = made.select("body_points.points", Some(0), None)?.unwrap();
    assert_eq!((x.dims, x.data), (vec![2, 3], QuantityData::Float64(vec![0.0, 1.0, 2.0, 6.0, 7.0, 8.0])));
    Ok(())
}

#[test]
fn the_document_refuses_the_ids_of_another_model() -> Result<()> {
    // a document changed directly (Model::with_document takes it): the id of a removed object is
    // an error of the call, never a panic
    let mut doc = hydrobem_core::Document::new();
    let root = doc.root_id();
    let gone = doc.add_child(root, "bodies", "", "b", -1)?;
    doc.remove(gone)?;
    let e = doc.rename(gone, "c").unwrap_err();
    assert_eq!(
        (e.kind(), e.message()),
        (ErrorKind::Argument, "the object is not one of this model (removed, or of another)")
    );
    assert_eq!(doc.remove(gone).unwrap_err().kind(), ErrorKind::Argument);
    assert_eq!(doc.move_(gone, 0).unwrap_err().kind(), ErrorKind::Argument);
    assert_eq!(doc.reset(gone, "yaw").unwrap_err().kind(), ErrorKind::Argument);
    assert_eq!(doc.add_child(gone, "tanks", "", "", -1).unwrap_err().kind(), ErrorKind::Argument);
    assert_eq!(doc.clear_list(gone, "tanks").unwrap_err().kind(), ErrorKind::Argument);
    assert_eq!(doc.copy_items(gone, root).unwrap_err().kind(), ErrorKind::Argument);
    assert!(doc.get_obj(gone).is_none());
    // a document of a model, changed and given to a new model
    let (m, body) = cylinder_model()?;
    let mut doc = m.document().clone();
    doc.rename(body, "hull")?;
    let mut copy = Model::with_document(doc);
    assert_eq!(copy.path(body)?, "bodies/hull");
    assert_eq!(copy.solved_body_counts()?.len(), 1);
    // a scalar set in the document holds its item's kind: an integer yaw is a real, a text for a
    // bool refused, so that the model gives the value of the item's kind
    let mut doc = m.document().clone();
    doc.set(body, "yaw", hydrobem_core::ItemValue::Int(30))?;
    let e = doc.set(body, "enabled", hydrobem_core::ItemValue::Text("yes".into())).unwrap_err();
    assert_eq!((e.kind(), e.message()), (ErrorKind::Type, "bodies/cylinder: enabled must be a bool"));
    let copy = Model::with_document(doc);
    assert_eq!(copy.get(body, "yaw")?, Some(Value::Real(30.0)));
    Ok(())
}

#[test]
fn the_values_of_items_by_their_kind() -> Result<()> {
    let (mut m, body) = cylinder_model()?;
    let modes = m.get(body, "modes")?.unwrap();
    assert_eq!(modes.as_texts().unwrap(), ["surge", "sway", "heave", "roll", "pitch", "yaw"]);
    assert!(modes.as_reals().is_none() && Value::Real(1.0).as_texts().is_none());
    // the counts of every body solved, in order: a disabled body is not solved
    let root = m.root();
    let other = m.add(root, "bodies", None, "other", None)?;
    m.set(other, "mesh", "cylinder")?;
    m.set(other, "position", [10.0, 0.0, 0.0])?;
    assert_eq!(m.solved_body_counts()?.len(), 2);
    m.set(body, "enabled", false)?;
    let per = m.solved_body_counts()?;
    assert_eq!(per, vec![m.body_counts(other)?]);
    // an array of more than eight dimensions, refused with the item named
    let e = m.set(body, "position", Value::reals(&[1; 9], vec![0.0])).unwrap_err();
    assert_eq!(
        (e.kind(), e.message()),
        (ErrorKind::Value, "bodies/cylinder: position: an array has 1 to 8 dimensions, got 9")
    );
    Ok(())
}
