//! The C ABI called as a C program calls it (through its exported functions): a model built and
//! solved through it gives the bits of the same model through the Rust API; the tables, the
//! errors, the progress callback and its cancellation, the refusals of a calculating model, the
//! files and results made from arrays; the result tree and the part of an owner; and every other
//! function of the header, its answer compared with the Rust API on the same model.

use std::ffi::{CStr, CString, c_char, c_void};
use std::ptr;

use hydrobem::*;
use hydrobem_core::{AnalyticShape, Model, QuantityData, QuantityId, Value};

fn c(s: &str) -> CString {
    CString::new(s).unwrap()
}

fn ok(s: HbStatus) {
    assert_eq!(s, HB_OK, "{}", last_error());
}

fn last_error() -> String {
    unsafe { CStr::from_ptr(hb_last_error_message()) }.to_str().unwrap().to_string()
}

const ROWS: [[i32; 6]; 2] = [[4, 4, 3, 3, 3, 4], [4, 2, 3, 3, 3, 4]];
const OMEGAS: [f64; 2] = [1.0, 2.0];

/// The analytic cylinder through the ABI: a model with one body, higher order, two frequencies.
unsafe fn c_model() -> *mut HbModel {
    let mut m = ptr::null_mut();
    unsafe {
        ok(hb_model_new(&mut m));
        let (dn, mapping, sym, shape) =
            ([c("radius"), c("draft")], c("uniform"), c("xz_and_yz"), c("circular_cylinder"));
        let names: Vec<*const c_char> = dn.iter().map(|s| s.as_ptr()).collect();
        let s = HbAnalyticShape {
            shape: shape.as_ptr(),
            dimension_names: names.as_ptr(),
            dimension_values: [1.0, 0.5].as_ptr(),
            dimensions: 2,
            discs: ptr::null(),
            disc_count: 0,
            parts: ptr::null(),
            part_count: 0,
            mapping: mapping.as_ptr(),
            discretization: ROWS.as_flattened().as_ptr(),
            discretization_rows: 2,
            reference_length: 1.0,
            symmetry: sym.as_ptr(),
            label: ptr::null(),
        };
        let mut mesh = 0;
        ok(hb_model_analytic_mesh(m, c("cyl").as_ptr(), &s, &mut mesh));
        let mut root = 0;
        ok(hb_object(m, c("").as_ptr(), &mut root));
        let mut body = 0;
        ok(hb_add(m, root, c("bodies").as_ptr(), ptr::null(), c("b").as_ptr(), -1, &mut body));
        ok(hb_set_text(m, body, c("mesh").as_ptr(), c("cyl").as_ptr()));
        let mut opt = 0;
        ok(hb_object(m, c("options").as_ptr(), &mut opt));
        ok(hb_set_bool(m, opt, c("higher_order").as_ptr(), 1));
        let mut w = 0;
        ok(hb_object(m, c("environment").as_ptr(), &mut w));
        ok(hb_set_reals(m, w, c("omegas").as_ptr(), OMEGAS.as_ptr(), [2usize].as_ptr(), 1));
        ok(hb_set_reals(m, w, c("headings").as_ptr(), [0.0].as_ptr(), [1usize].as_ptr(), 1));
    }
    m
}

/// The same model through the Rust API.
fn rust_model() -> Model {
    let mut m = Model::new();
    let shape = AnalyticShape {
        shape: "circular_cylinder".into(),
        dimensions: vec![("radius".into(), 1.0), ("draft".into(), 0.5)],
        discs: vec![],
        parts: vec![],
        mapping: "uniform".into(),
        discretization: ROWS.to_vec(),
        reference_length: 1.0,
        symmetry: "xz_and_yz".into(),
        label: String::new(),
    };
    m.analytic_mesh("cyl", &shape).unwrap();
    let root = m.root();
    let b = m.add(root, "bodies", None, "b", None).unwrap();
    m.set(b, "mesh", "cyl").unwrap();
    let o = m.object("options").unwrap();
    m.set(o, "higher_order", true).unwrap();
    let w = m.object("environment").unwrap();
    m.set(w, "omegas", OMEGAS.to_vec()).unwrap();
    m.set(w, "headings", vec![0.0]).unwrap();
    m
}

unsafe fn column(t: *const HbTable, name: &str) -> HbColumn {
    let mut col = HbColumn { name: ptr::null(), dtype: 0, ndim: 0, shape: ptr::null(), count: 0, data: ptr::null() };
    ok(unsafe { hb_table_find(t, c(name).as_ptr(), &mut col) });
    col
}

unsafe fn shape(col: &HbColumn) -> Vec<usize> {
    unsafe { std::slice::from_raw_parts(col.shape, col.ndim) }.to_vec()
}

unsafe fn f64s(col: &HbColumn) -> Vec<f64> {
    assert_eq!(col.dtype, HB_F64);
    unsafe { std::slice::from_raw_parts(col.data.cast::<f64>(), col.count) }.to_vec()
}

unsafe fn text_at(col: &HbColumn, k: usize) -> String {
    assert_eq!(col.dtype, HB_TEXT);
    let p = unsafe { *col.data.cast::<*const c_char>().add(k) };
    unsafe { CStr::from_ptr(p) }.to_str().unwrap().to_string()
}

#[test]
fn a_model_through_the_abi_solves_to_the_bits_of_the_rust_api() {
    unsafe {
        let m = c_model();
        let mut r = ptr::null_mut();
        ok(hb_model_solve(m, None, ptr::null_mut(), &mut r));
        let mut col =
            HbColumn { name: ptr::null(), dtype: 0, ndim: 0, shape: ptr::null(), count: 0, data: ptr::null() };
        let mut reference = rust_model();
        let rr = reference.solve(|_| {}).unwrap().clone();
        for q in [QuantityId::AddedMass, QuantityId::RadiationDamping, QuantityId::ExcitationDiffraction] {
            ok(hb_results_quantity(r, c(q.name()).as_ptr(), &mut col));
            assert_eq!(CStr::from_ptr(col.name).to_str().unwrap(), q.name());
            assert_eq!(shape(&col), rr.shape(q).unwrap().dims);
            let bytes = std::slice::from_raw_parts(col.data.cast::<u8>(), rr.bytes(q).unwrap().len());
            assert_eq!(bytes, rr.bytes(q).unwrap(), "{}", q.name());
        }
        let mut state = -1;
        ok(hb_model_state(m, &mut state));
        assert_eq!(state, HB_STATE_CALCULATED);
        // the results of the model are those handed out
        let mut again = ptr::null_mut();
        ok(hb_model_results(m, &mut again));
        let mut a2 = col;
        ok(hb_results_quantity(again, c("added_mass").as_ptr(), &mut a2));
        ok(hb_results_quantity(r, c("added_mass").as_ptr(), &mut col));
        assert_eq!(
            std::slice::from_raw_parts(a2.data.cast::<f32>(), a2.count),
            std::slice::from_raw_parts(col.data.cast::<f32>(), col.count)
        );
        hb_results_free(again);
        hb_results_free(r);
        ok(hb_model_free(m));
    }
}

#[test]
fn items_objects_and_tables() {
    unsafe {
        let m = c_model();
        let mut w = 0;
        ok(hb_object(m, c("environment").as_ptr(), &mut w));
        let mut t = ptr::null_mut();
        ok(hb_get(m, w, c("omegas").as_ptr(), &mut t));
        assert_eq!(text_at(&column(t, "kind"), 0), "reals");
        let v = column(t, "value");
        assert_eq!((shape(&v), f64s(&v)), (vec![2], OMEGAS.to_vec()));
        hb_table_free(t);
        // the elements of a surface: its mesh (a name) and its ranges (an (n, 2) integer array)
        let mut b = 0;
        ok(hb_object(m, c("bodies/b").as_ptr(), &mut b));
        ok(hb_set_ints(m, b, c("ranges").as_ptr(), [1i64, 2].as_ptr(), [1usize, 2].as_ptr(), 2));
        ok(hb_get(m, b, c("mesh").as_ptr(), &mut t));
        assert_eq!(text_at(&column(t, "kind"), 0), "text");
        assert_eq!(text_at(&column(t, "value"), 0), "cyl");
        hb_table_free(t);
        ok(hb_get(m, b, c("ranges").as_ptr(), &mut t));
        assert_eq!(text_at(&column(t, "kind"), 0), "ints");
        let r = column(t, "value");
        assert_eq!((r.dtype, shape(&r)), (HB_I64, vec![1, 2]));
        assert_eq!(std::slice::from_raw_parts(r.data.cast::<i64>(), 2), [1, 2]);
        hb_table_free(t);
        ok(hb_reset(m, b, c("ranges").as_ptr()));
        // the info of an item and an object, the children of a list
        ok(hb_item_info(m, b, c("mass").as_ptr(), &mut t));
        assert_eq!(text_at(&column(t, "kind"), 0), "real");
        hb_table_free(t);
        ok(hb_object_info(m, b, &mut t));
        assert_eq!(
            (text_at(&column(t, "type"), 0), text_at(&column(t, "path"), 0)),
            ("Body".into(), "bodies/b".into())
        );
        hb_table_free(t);
        let mut root = 0;
        ok(hb_object(m, c("").as_ptr(), &mut root));
        ok(hb_children(m, root, c("bodies").as_ptr(), &mut t));
        let ids = column(t, "ids");
        assert_eq!((ids.dtype, ids.count, *ids.data.cast::<u64>()), (HB_U64, 1, b));
        let mut n = 0;
        ok(hb_table_count(t, &mut n));
        assert_eq!(n, 1);
        hb_table_free(t);
        // the hydrostatics of the body: the volume of the whole cylinder (its symmetry expanded)
        ok(hb_body_hydrostatics(m, b, &mut t));
        let vol = f64s(&column(t, "volumes"));
        assert!((vol[2] - std::f64::consts::PI * 0.5).abs() < 1e-6, "{vol:?}");
        hb_table_free(t);
        // the geometry file of the mesh read back as another mesh: the same geometry
        let mut mesh = 0;
        ok(hb_object(m, c("meshes/cyl").as_ptr(), &mut mesh));
        let mut text = ptr::null_mut();
        ok(hb_mesh_hbg(m, mesh, &mut text));
        let mut copy = 0;
        ok(hb_model_mesh_from_hbg(m, c("copy").as_ptr(), text, &mut copy));
        hb_string_free(text);
        let (mut g1, mut g2) = (ptr::null_mut(), ptr::null_mut());
        ok(hb_mesh_geometry(m, mesh, &mut g1));
        ok(hb_mesh_geometry(m, copy, &mut g2));
        assert_eq!(f64s(&column(g1, "volumes")), f64s(&column(g2, "volumes")));
        hb_table_free(g1);
        hb_table_free(g2);
        // the schema and the quantities
        ok(hb_schema_items(&mut t));
        let items = column(t, "item");
        assert!((0..items.count).any(|k| text_at(&items, k) == "omegas"));
        hb_table_free(t);
        ok(hb_quantities(&mut t));
        assert_eq!(text_at(&column(t, "name"), 0), QuantityId::ALL[0].name());
        hb_table_free(t);
        assert_eq!(hb_abi_version(), HB_ABI_VERSION);
        ok(hb_model_free(m));
    }
}

#[test]
fn errors_are_statuses_with_their_messages() {
    unsafe {
        let m = c_model();
        let mut w = 0;
        ok(hb_object(m, c("environment").as_ptr(), &mut w));
        // a value the engine refuses, an unknown item, a wrong kind
        assert_eq!(hb_set_reals(m, w, c("omegas").as_ptr(), [-1.0].as_ptr(), [1usize].as_ptr(), 1), HB_ERR_VALUE);
        assert!(last_error().contains("omegas"), "{}", last_error());
        assert_ne!(hb_set_real(m, w, c("no_such_item").as_ptr(), 1.0), HB_OK);
        assert!(last_error().contains("no_such_item"), "{}", last_error());
        assert_eq!(hb_set_bool(m, w, c("omegas").as_ptr(), 1), HB_ERR_TYPE, "{}", last_error());
        // null pointers and text that is not UTF-8
        assert_eq!(hb_object(m, ptr::null(), &mut w), HB_ERR_ARGUMENT);
        assert!(last_error().contains("path is a null pointer"));
        assert_eq!(hb_object(ptr::null_mut(), c("").as_ptr(), &mut w), HB_ERR_ARGUMENT);
        let bad = [0xffu8, 0];
        assert_eq!(hb_object(m, bad.as_ptr().cast(), &mut w), HB_ERR_ARGUMENT);
        assert!(last_error().contains("not UTF-8"));
        // an object removed: its id is refused
        let mut b = 0;
        ok(hb_object(m, c("bodies/b").as_ptr(), &mut b));
        ok(hb_remove(m, b));
        assert_ne!(hb_set_real(m, b, c("mass").as_ptr(), 1.0), HB_OK);
        // an unknown quantity, a file that does not exist
        let mut t = ptr::null_mut();
        assert_eq!(hb_results_normalized(ptr::null_mut(), c("added_mass").as_ptr(), &mut t), HB_ERR_ARGUMENT);
        let mut other = ptr::null_mut();
        assert_eq!(hb_model_load(c("no/such/file.hbm").as_ptr(), &mut other), HB_ERR_IO, "{}", last_error());
        // the location of the last failure (a file names no body)
        let mut l = HbLocation { body: 0, mesh: 0, element: 0, field_point: 0, lid: 0, frequency: 0 };
        ok(hb_last_error_location(&mut l));
        assert_eq!(l.body, -1);
        ok(hb_model_free(m));
    }
}

struct Seen {
    model: *mut HbModel,
    calls: usize,
    stages: Vec<i32>,
    // (stage, frequency, worker, phase, step, item) of every report
    fields: Vec<(i32, i64, i32, i32, i32, i64)>,
    free_refused: bool,
    running_seen: bool,
    stop_after: usize,
}

unsafe extern "C" fn progress(p: *const HbProgress, user: *mut c_void) -> i32 {
    let s = unsafe { &mut *user.cast::<Seen>() };
    s.calls += 1;
    s.stages.push(unsafe { (*p).stage });
    let q = unsafe { &*p };
    s.fields.push((q.stage, q.frequency, q.worker, q.phase, q.step, q.item));
    // the model refuses to be freed while it calculates; its progress is read meanwhile
    if s.calls == 1 {
        s.free_refused = unsafe { hb_model_free(s.model) } == HB_ERR_STATE;
        let mut t = ptr::null_mut();
        if unsafe { hb_model_progress(s.model, &mut t) } == HB_OK {
            s.running_seen = unsafe { *column(t, "running").data.cast::<i32>() } == 1;
            unsafe { hb_table_free(t) };
        }
    }
    i32::from(s.calls < s.stop_after)
}

#[test]
fn progress_cancel_and_a_calculating_model() {
    unsafe {
        let m = c_model();
        let mut seen = Seen {
            model: m,
            calls: 0,
            stages: vec![],
            fields: vec![],
            free_refused: false,
            running_seen: false,
            stop_after: usize::MAX,
        };
        let mut r = ptr::null_mut();
        ok(hb_model_solve(m, Some(progress), (&raw mut seen).cast(), &mut r));
        assert!(seen.free_refused && seen.running_seen && seen.calls > 2);
        // after the solve: both frequencies done, in the order of the results
        let mut t = ptr::null_mut();
        ok(hb_model_progress(m, &mut t));
        let (state, fraction) = (column(t, "state"), column(t, "fraction"));
        assert_eq!((state.count, text_at(&state, 0), text_at(&state, 1)), (2, "done".into(), "done".into()));
        assert_eq!(f64s(&fraction), vec![1.0, 1.0]);
        assert_eq!(*column(t, "done").data.cast::<i64>(), 2);
        assert_eq!(column(t, "limit").count, 0);
        // one frequency worker (the default): both frequencies on worker 0; the worker done, the
        // phase done, no second order
        assert_eq!(i64s(&column(t, "worker")), vec![0, 0]);
        let ws = column(t, "worker_state");
        assert_eq!((ws.count, text_at(&ws, 0)), (1, "done".into()));
        assert_eq!(text_at(&column(t, "worker_task"), 0), "");
        assert_eq!(i64s(&column(t, "worker_index")), vec![-1]);
        assert_eq!(text_at(&column(t, "phase"), 0), "done");
        assert_eq!((column(t, "step").count, column(t, "item_state").count), (0, 0));
        hb_table_free(t);
        assert!(seen.stages.contains(&HB_STAGE_STATICS) && seen.stages.contains(&HB_STAGE_FREQUENCY_DONE));
        // the reports: the frequencies' on worker 0, the statics' on none; the phases in their
        // order (always reported), ending done; no second-order step
        for &(stage, f, w, phase, step, item) in &seen.fields {
            assert!((HB_PHASE_PREPARATION..=HB_PHASE_FAILED).contains(&phase));
            assert_eq!((step, item), (-1, -1));
            if f >= 0 {
                assert_eq!(w, 0, "stage {stage}");
            }
            if stage == HB_STAGE_STATICS {
                assert_eq!((f, w), (-2, -1));
            }
        }
        let phases: Vec<i32> = seen
            .fields
            .iter()
            .filter(|x| x.0 == HB_STAGE_PHASE)
            .map(|x| (x.3, x.1))
            .map(|(p, f)| {
                assert_eq!(f, -4);
                p
            })
            .collect();
        assert_eq!(phases, vec![HB_PHASE_FIRST_ORDER, HB_PHASE_FINISHING, HB_PHASE_DONE]);
        hb_results_free(r);
        // returning 0 cancels the solve; the model is reset and solves again
        let mut seen = Seen {
            model: m,
            calls: 0,
            stages: vec![],
            fields: vec![],
            free_refused: false,
            running_seen: false,
            stop_after: 1,
        };
        assert_eq!(hb_model_solve(m, Some(progress), (&raw mut seen).cast(), ptr::null_mut()), HB_ERR_CANCELLED);
        let mut state = -1;
        ok(hb_model_state(m, &mut state));
        assert_eq!(state, HB_STATE_RESET);
        let mut t = ptr::null_mut();
        ok(hb_model_progress(m, &mut t));
        let st = column(t, "state");
        assert!((0..st.count).all(|k| text_at(&st, k) == "cancelled"));
        assert_eq!(text_at(&column(t, "phase"), 0), "cancelled");
        assert_eq!(text_at(&column(t, "worker_state"), 0), "done");
        hb_table_free(t);
        ok(hb_model_solve(m, None, ptr::null_mut(), ptr::null_mut()));
        ok(hb_model_free(m));
    }
}

#[test]
fn files_and_results_of_arrays() {
    let dir = std::env::temp_dir().join(format!("hydrobem-c-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    unsafe {
        let m = c_model();
        let mut r = ptr::null_mut();
        ok(hb_model_solve(m, None, ptr::null_mut(), &mut r));
        let (hbm, hbr) = (dir.join("m.hbm"), dir.join("m.hbr"));
        let (hbm, hbr) = (c(hbm.to_str().unwrap()), c(hbr.to_str().unwrap()));
        ok(hb_model_save(m, hbm.as_ptr(), 0));
        assert_eq!(hb_model_save(m, hbm.as_ptr(), 0), HB_ERR_EXISTS);
        ok(hb_results_save(r, hbr.as_ptr(), 0));
        let stem = c(dir.join("wamit/cyl").to_str().unwrap());
        ok(hb_results_write_wamit(r, stem.as_ptr(), 0, 0, c("2026-01-01").as_ptr()));
        assert!(dir.join("wamit/cyl.1").exists());
        // the model file read back solves to the same added mass; the results file holds it
        let mut back = ptr::null_mut();
        ok(hb_model_load(hbm.as_ptr(), &mut back));
        let mut loaded = ptr::null_mut();
        ok(hb_model_load_results(hbr.as_ptr(), &mut loaded));
        let mut lr = ptr::null_mut();
        ok(hb_model_results(loaded, &mut lr));
        let mut a = HbColumn { name: ptr::null(), dtype: 0, ndim: 0, shape: ptr::null(), count: 0, data: ptr::null() };
        let mut b = a;
        ok(hb_results_quantity(r, c("added_mass").as_ptr(), &mut a));
        ok(hb_results_quantity(lr, c("added_mass").as_ptr(), &mut b));
        assert_eq!((a.dtype, a.count), (HB_F32, b.count));
        assert_eq!(
            std::slice::from_raw_parts(a.data.cast::<f32>(), a.count),
            std::slice::from_raw_parts(b.data.cast::<f32>(), b.count)
        );
        // results made from arrays: put a quantity and read it back
        let mut made = ptr::null_mut();
        ok(hb_model_new_results(back, &mut made));
        let v: Vec<f32> = (0..a.count).map(|k| k as f32).collect();
        let dims = shape(&a);
        ok(hb_results_put(made, c("added_mass").as_ptr(), v.as_ptr().cast(), dims.as_ptr(), dims.len()));
        let mut has = 0;
        ok(hb_results_has(made, c("added_mass").as_ptr(), &mut has));
        assert_eq!(has, 1);
        ok(hb_results_quantity(made, c("added_mass").as_ptr(), &mut b));
        assert_eq!(std::slice::from_raw_parts(b.data.cast::<f32>(), b.count), v.as_slice());
        // a refused put changes nothing: the view taken before it stays valid
        let wrong = [1usize, 2, 3];
        assert_eq!(
            hb_results_put(made, c("added_mass").as_ptr(), v.as_ptr().cast(), wrong.as_ptr(), wrong.len()),
            HB_ERR_VALUE
        );
        let name = std::ffi::CStr::from_ptr(b.name).to_str().unwrap();
        assert_eq!((name, std::slice::from_raw_parts(b.shape, b.ndim)), ("added_mass", dims.as_slice()));
        // the quantities given together, checked once all are put: one hydrostatics quantity of the
        // eleven is refused, naming one missing
        ok(hb_results_check(made));
        let one = [1usize];
        let vol = [1.0f64];
        ok(hb_results_put(made, c("hydrostatics_volume").as_ptr(), vol.as_ptr().cast(), one.as_ptr(), 1));
        assert_eq!(hb_results_check(made), HB_ERR_VALUE);
        assert!(last_error().contains("hydrostatics_"), "{}", last_error());
        assert_eq!(hb_results_check(ptr::null_mut()), HB_ERR_ARGUMENT);
        for p in [made, lr, r] {
            hb_results_free(p);
        }
        for p in [back, loaded, m] {
            ok(hb_model_free(p));
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_result_tree_and_the_part_of_an_owner() {
    unsafe {
        // the tree: the views of the engine, one row each
        let mut t = ptr::null_mut();
        ok(hb_result_views(&mut t));
        let views = hydrobem_core::result_views();
        let (path, owner, width) = (column(t, "path"), column(t, "owner"), column(t, "width"));
        assert_eq!(path.count, views.len());
        for (k, v) in views.iter().enumerate() {
            assert_eq!((text_at(&path, k), text_at(&owner, k)), (v.path.to_string(), v.owner.name().to_string()));
            assert_eq!(*width.data.cast::<i32>().add(k) as usize, v.owner.width());
        }
        hb_table_free(t);
        ok(hb_result_groups(&mut t));
        assert_eq!(column(t, "path").count, hydrobem_core::result_groups().len());
        hb_table_free(t);
        // the part of the body: its coordinates, as the Rust API selects them
        let m = c_model();
        let mut r = ptr::null_mut();
        ok(hb_model_solve(m, None, ptr::null_mut(), &mut r));
        let mut reference = rust_model();
        let rr = reference.solve(|_| {}).unwrap().clone();
        ok(hb_results_select(r, c("excitation.diffraction").as_ptr(), 0, -1, &mut t));
        let want = rr.select("excitation.diffraction", Some(0), None).unwrap().unwrap();
        let col = column(t, "values");
        assert_eq!((col.dtype, shape(&col)), (HB_C64, want.dims.clone()));
        let QuantityData::ComplexF32(v) = want.data else { panic!("complex64") };
        assert_eq!(std::slice::from_raw_parts(col.data.cast::<hydrobem_core::Complex32>(), col.count), v.as_slice());
        hb_table_free(t);
        // where that part lies: the axes and indices of the Rust API's selection, one row per index
        ok(hb_results_selection(r, c("excitation.diffraction").as_ptr(), 0, -1, &mut t));
        let sel = rr.selection("excitation.diffraction", Some(0), None).unwrap().unwrap();
        let (mut axes, mut rows) = (Vec::new(), Vec::new());
        for (axis, taken) in &sel.axes {
            for i in taken.indices() {
                axes.push(*axis as i64);
                rows.push(i as i64);
            }
        }
        assert!(!axes.is_empty());
        assert_eq!(text_at(&column(t, "quantity"), 0), "excitation_diffraction");
        assert_eq!(i64s(&column(t, "axis")), axes);
        assert_eq!(i64s(&column(t, "index")), rows);
        hb_table_free(t);
        ok(hb_results_selection(r, c("drift.pressure_integration").as_ptr(), 0, -1, &mut t));
        let mut n = 1;
        ok(hb_table_count(t, &mut n));
        assert_eq!(n, 0);
        hb_table_free(t);
        assert_eq!(hb_results_selection(r, c("drift.pressure").as_ptr(), -1, -1, &mut t), HB_ERR_VALUE);
        assert_eq!(hb_results_selection(r, c("added_mass").as_ptr(), 1, -1, &mut t), HB_ERR_ARGUMENT);
        // not computed: a table without a column; a path of no view, a body out of range
        ok(hb_results_select(r, c("drift.pressure_integration").as_ptr(), 0, -1, &mut t));
        let mut n = 1;
        ok(hb_table_count(t, &mut n));
        assert_eq!(n, 0);
        hb_table_free(t);
        assert_eq!(hb_results_select(r, c("drift.pressure").as_ptr(), -1, -1, &mut t), HB_ERR_VALUE);
        assert_eq!(hb_results_select(r, c("added_mass").as_ptr(), 1, -1, &mut t), HB_ERR_ARGUMENT);
        assert_eq!(hb_results_select(r, c("added_mass").as_ptr(), -2, -1, &mut t), HB_ERR_ARGUMENT);
        hb_results_free(r);
        ok(hb_model_free(m));
    }
}

#[test]
fn the_constants_are_those_of_the_engine() {
    use hydrobem_core::QualityFlags;
    assert_eq!(HB_QUALITY_ORIENTATION as u32, QualityFlags::ORIENTATION.0);
    assert_eq!(HB_QUALITY_FREE_EDGE as u32, QualityFlags::FREE_EDGE.0);
    assert_eq!(HB_QUALITY_NON_MANIFOLD as u32, QualityFlags::NON_MANIFOLD.0);
    assert_eq!(HB_QUALITY_COARSE as u32, QualityFlags::COARSE.0);
    // an id round trips through its number
    let m = Model::new();
    let root = m.root();
    assert_eq!(hydrobem_core::ObjectId::from_bits(root.to_bits()), root);
    let _ = (Value::Bool(true), QuantityData::Int32(vec![]));
}

// ------------------------------------------------------------------ every function of the ABI
// (the queries, the objects and the results compared with the Rust API on the same model: the
// C model saved and read back by Model::load)

fn quad(p: [f64; 3], a: [f64; 3], b: [f64; 3], normal: [f64; 3]) -> [[f64; 3]; 4] {
    let add = |x: [f64; 3], y: [f64; 3]| [x[0] + y[0], x[1] + y[1], x[2] + y[2]];
    let q = [p, add(p, a), add(add(p, a), b), add(p, b)];
    let (d1, d2) = (
        [q[2][0] - q[0][0], q[2][1] - q[0][1], q[2][2] - q[0][2]],
        [q[3][0] - q[1][0], q[3][1] - q[1][1], q[3][2] - q[1][2]],
    );
    let n = [d1[1] * d2[2] - d1[2] * d2[1], d1[2] * d2[0] - d1[0] * d2[2], d1[0] * d2[1] - d1[1] * d2[0]];
    if n[0] * normal[0] + n[1] * normal[1] + n[2] * normal[2] > 0.0 { q } else { [q[0], q[3], q[2], q[1]] }
}

/// The box |x| <= 1, |y| <= 0.5, -0.5 <= z <= 0 as panels, 2 x 2 per face (normals into the fluid):
/// the 20 elements of the hull, then the 4 of its waterplane (upward normals).
fn box_vertices() -> Vec<[[f64; 3]; 4]> {
    let face = |o: [f64; 3], du: [f64; 3], dv: [f64; 3], normal: [f64; 3]| {
        let mut out = Vec::new();
        for i in 0..2 {
            for j in 0..2 {
                let (a, b) = (du.map(|x| x / 2.0), dv.map(|x| x / 2.0));
                let p = [0, 1, 2].map(|k| o[k] + a[k] * i as f64 + b[k] * j as f64);
                out.push(quad(p, a, b, normal));
            }
        }
        out
    };
    let mut v = Vec::new();
    v.extend(face([-1.0, -0.5, -0.5], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]));
    v.extend(face([1.0, -0.5, -0.5], [0.0, 1.0, 0.0], [0.0, 0.0, 0.5], [1.0, 0.0, 0.0]));
    v.extend(face([-1.0, 0.5, -0.5], [2.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 1.0, 0.0]));
    v.extend(face([-1.0, -0.5, -0.5], [0.0, 1.0, 0.0], [0.0, 0.0, 0.5], [-1.0, 0.0, 0.0]));
    v.extend(face([-1.0, -0.5, -0.5], [2.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, -1.0, 0.0]));
    v.extend(face([-1.0, -0.5, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]));
    v
}

/// A low-order floating box through the ABI: its mesh of panels, its waterplane the interior
/// surface, displaced mass, two frequencies, the motions and the potential QTF (indirect) with a
/// free surface of panels to R = 3.
unsafe fn c_box() -> *mut HbModel {
    let mut m = ptr::null_mut();
    unsafe {
        ok(hb_model_new(&mut m));
        let mut root = 0;
        ok(hb_object(m, c("").as_ptr(), &mut root));
        let mut mesh = 0;
        ok(hb_add(m, root, c("meshes").as_ptr(), ptr::null(), c("box").as_ptr(), -1, &mut mesh));
        let v = box_vertices();
        ok(hb_set_reals(
            m,
            mesh,
            c("vertices").as_ptr(),
            v.as_flattened().as_flattened().as_ptr(),
            [24usize, 4, 3].as_ptr(),
            3,
        ));
        let mut b = 0;
        ok(hb_add(m, root, c("bodies").as_ptr(), ptr::null(), c("b").as_ptr(), -1, &mut b));
        ok(hb_set_text(m, b, c("mesh").as_ptr(), c("box").as_ptr()));
        ok(hb_set_ints(m, b, c("ranges").as_ptr(), [1i64, 20].as_ptr(), [1usize, 2].as_ptr(), 2));
        // the interior surface: the given elements of its mesh, items of the body (an item of a method
        // not chosen is kept unused, not refused)
        ok(hb_set_text(m, b, c("interior_surface_mesh").as_ptr(), c("box").as_ptr()));
        ok(hb_set_text(m, b, c("interior_surface").as_ptr(), c("mesh").as_ptr()));
        ok(hb_set_ints(m, b, c("interior_surface_ranges").as_ptr(), [21i64, 24].as_ptr(), [1usize, 2].as_ptr(), 2));
        ok(hb_set_real(m, b, c("control_surface_radius").as_ptr(), 2.0)); // control_surface "none": unused
        // an item the body does not have
        assert_eq!(hb_set_real(m, b, c("radius").as_ptr(), 1.0), HB_ERR_ARGUMENT);
        ok(hb_set_text(m, b, c("mass").as_ptr(), c("displaced").as_ptr()));
        ok(hb_set_reals(m, b, c("center_of_gravity").as_ptr(), [0.0, 0.0, -0.1].as_ptr(), [3usize].as_ptr(), 1));
        ok(hb_set_reals(m, b, c("radii_of_gyration").as_ptr(), [0.4, 0.5, 0.6].as_ptr(), [3usize].as_ptr(), 1));
        let mut env = 0;
        ok(hb_object(m, c("environment").as_ptr(), &mut env));
        ok(hb_set_reals(m, env, c("omegas").as_ptr(), [1.0, 1.4].as_ptr(), [2usize].as_ptr(), 1));
        ok(hb_set_reals(m, env, c("headings").as_ptr(), [0.0].as_ptr(), [1usize].as_ptr(), 1));
        let mut out = 0;
        ok(hb_object(m, c("first_order").as_ptr(), &mut out));
        ok(hb_set_bool(m, out, c("motion").as_ptr(), 1));
        let mut second = 0;
        ok(hb_object(m, c("second_order").as_ptr(), &mut second));
        ok(hb_set_bool(m, second, c("qtf_potential_indirect").as_ptr(), 1));
        ok(hb_set_real(m, second, c("free_surface_inner_radius").as_ptr(), 3.0));
        ok(hb_set_real(m, second, c("free_surface_panel_size").as_ptr(), 0.5));
        let mut opt = 0;
        ok(hb_object(m, c("options").as_ptr(), &mut opt));
        ok(hb_set_int(m, opt, c("frequency_threads").as_ptr(), 2));
    }
    m
}

fn scratch(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("hydrobem-c-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// The model of a C model as the Rust API reads it from the model file the ABI writes.
unsafe fn rust_of(m: *mut HbModel, dir: &std::path::Path) -> Model {
    let p = dir.join("copy.hbm");
    ok(unsafe { hb_model_save(m, c(p.to_str().unwrap()).as_ptr(), 1) });
    Model::load(&p).unwrap()
}

unsafe fn i32s(col: &HbColumn) -> Vec<i32> {
    assert_eq!(col.dtype, HB_I32);
    unsafe { std::slice::from_raw_parts(col.data.cast::<i32>(), col.count) }.to_vec()
}

unsafe fn i64s(col: &HbColumn) -> Vec<i64> {
    assert_eq!(col.dtype, HB_I64);
    unsafe { std::slice::from_raw_parts(col.data.cast::<i64>(), col.count) }.to_vec()
}

unsafe fn texts_of(col: &HbColumn) -> Vec<String> {
    (0..col.count).map(|k| unsafe { text_at(col, k) }).collect()
}

/// A table of a call, its column read and the table freed.
unsafe fn take<T>(t: *mut HbTable, name: &str, read: unsafe fn(&HbColumn) -> T) -> T {
    let v = unsafe { read(&column(t, name)) };
    unsafe { hb_table_free(t) };
    v
}

fn flat3(v: &[[f64; 3]]) -> Vec<f64> {
    v.as_flattened().to_vec()
}

fn flat43(v: &[[[f64; 3]; 4]]) -> Vec<f64> {
    v.as_flattened().as_flattened().to_vec()
}

#[test]
fn every_query_of_the_model_gives_the_answer_of_the_rust_api() {
    let dir = scratch("queries");
    unsafe {
        let m = c_box();
        let mut rm = rust_of(m, &dir);
        let mut b = 0;
        ok(hb_object(m, c("bodies/b").as_ptr(), &mut b));
        let rb = rm.object("bodies/b").unwrap();
        let mut t = ptr::null_mut();
        // the rules, their warnings, the route
        ok(hb_model_check(m, ptr::null_mut()));
        rm.check().unwrap();
        ok(hb_model_diagnostics(m, &mut t));
        assert_eq!(column(t, "code").count, rm.diagnostics().len());
        hb_table_free(t);
        let mut route = ptr::null_mut();
        ok(hb_model_route(m, &mut route));
        assert_eq!(CStr::from_ptr(route).to_str().unwrap(), rm.route().unwrap());
        hb_string_free(route);
        // the counts, of the model and per body solved
        let rc = rm.counts().unwrap();
        ok(hb_model_counts(m, &mut t));
        assert_eq!(i64s(&column(t, "coordinates")), vec![rc.coordinates as i64]);
        assert_eq!(i64s(&column(t, "ordinary_frequencies")), vec![2]);
        let per = rm.solved_body_counts().unwrap();
        assert_eq!(i64s(&column(t, "wetted")), vec![per[0].wetted as i64]);
        assert_eq!(i64s(&column(t, "stored")), vec![24]);
        assert_eq!(i32s(&column(t, "radiation_mode_ids")), rm.radiation_mode_ids().unwrap());
        hb_table_free(t);
        // the motion equation: ready with the displaced mass and the radii of gyration
        let me = rm.motion_equation().unwrap();
        ok(hb_model_motion_equation(m, &mut t));
        assert_eq!(shape(&column(t, "inertia")), vec![6, 6]);
        assert_eq!(f64s(&column(t, "inertia")), me.inertia);
        assert_eq!(f64s(&column(t, "stiffness")), me.stiffness);
        assert_eq!(*column(t, "ready").data.cast::<i32>(), 1);
        hb_table_free(t);
        // the panels, of every role and of some (the names of hb_panel_roles separated by commas)
        let rp = rm.panels(&[], true).unwrap();
        ok(hb_model_panels(m, ptr::null(), 1, &mut t));
        assert_eq!(f64s(&column(t, "centroids")), flat3(&rp.centroids));
        assert_eq!(i32s(&column(t, "role")), rp.role.iter().map(|r| r.value()).collect::<Vec<_>>());
        assert_eq!(i32s(&column(t, "element")), rp.element);
        assert_eq!(i32s(&column(t, "part")), rp.part);
        assert_eq!(i32s(&column(t, "distinct_vertices")), rp.distinct_vertices);
        hb_table_free(t);
        ok(hb_model_panels(m, c("wetted").as_ptr(), 0, &mut t));
        assert_eq!(take(t, "areas", f64s), rm.panels(&["wetted"], false).unwrap().areas);
        ok(hb_model_panels(m, c("wetted,lid").as_ptr(), 1, &mut t));
        assert_eq!(take(t, "areas", f64s), rm.panels(&["wetted", "lid"], true).unwrap().areas);
        for old in ["hull", "body", "wet"] {
            assert_eq!(hb_model_panels(m, c(old).as_ptr(), 0, &mut t), HB_ERR_VALUE);
            assert!(last_error().contains("role must be one of wetted, interior_surface"), "{}", last_error());
        }
        ok(hb_panel_roles(&mut t));
        let names = take(t, "names", texts_of);
        assert_eq!(names[rp.role[0].value() as usize], "wetted");
        ok(hb_model_panels(m, c("lid").as_ptr(), 1, &mut t));
        assert_eq!(take(t, "areas", f64s).len(), rm.panels(&["lid"], true).unwrap().areas.len());
        ok(hb_model_tessellation(m, ptr::null(), 2, 2, 1, &mut t));
        let rt = rm.tessellation(&[], [2, 2], true).unwrap();
        assert_eq!(f64s(&column(t, "vertices")), flat43(&rt.vertices));
        assert_eq!(i32s(&column(t, "element")), rt.element);
        assert_eq!(i32s(&column(t, "role")), rt.role.iter().map(|r| r.map_or(-1, |r| r.value())).collect::<Vec<_>>());
        hb_table_free(t);
        ok(hb_model_tessellation(m, c("wetted").as_ptr(), 0, 0, 0, &mut t));
        let spans = rm.tessellation(&["wetted"], [0, 0], false).unwrap();
        assert_eq!(take(t, "vertices", f64s), flat43(&spans.vertices));
        assert_eq!(hb_model_tessellation(m, ptr::null(), 0, 3, 1, &mut t), HB_ERR_VALUE);
        ok(hb_model_mesh_quality(m, &mut t));
        assert_eq!(take(t, "per_wavelength", f64s), rm.mesh_quality().unwrap().per_wavelength);
        // the displacements of every coordinate at the centroids of the elements
        let n = rp.centroids.len().min(6);
        let pts: Vec<[f64; 3]> = rp.centroids[..n].to_vec();
        let (bodies, elements) = (vec![0i32; n], rp.element[..n].to_vec());
        ok(hb_model_mode_displacements(m, pts.as_flattened().as_ptr(), bodies.as_ptr(), elements.as_ptr(), n, &mut t));
        assert_eq!(shape(&column(t, "values")), vec![n, rc.coordinates, 3]);
        assert_eq!(take(t, "values", f64s), rm.mode_displacements(&pts, &bodies, &elements).unwrap());
        ok(hb_model_memory_estimate(m, &mut t));
        assert_eq!(take(t, "peak", i64s), vec![rm.memory_estimate().unwrap().peak]);
        // the status of points: in the fluid, inside the body
        let pts = [[3.0, 0.0, -0.2], [0.0, 0.0, -0.2]];
        ok(hb_model_classify_points(m, pts.as_flattened().as_ptr(), 2, 0, &mut t));
        let want = rm.classify_points(&pts, None).unwrap();
        assert_eq!(
            texts_of(&column(t, "status")),
            want.iter().map(|p| p.status.name().to_string()).collect::<Vec<_>>()
        );
        assert_eq!(text_at(&column(t, "status"), 0), "fluid");
        assert_eq!(i64s(&column(t, "body")), want.iter().map(|p| p.body.map_or(-1, |i| i as i64)).collect::<Vec<_>>());
        assert_eq!(i64s(&column(t, "body")), vec![-1, 0]);
        assert_eq!(i64s(&column(t, "lid")), vec![-1, -1]);
        assert_eq!(take(t, "tank", i64s), vec![-1, -1]);
        // HB_NO_OBJECT names the sea as 0 does
        ok(hb_model_classify_points(m, pts.as_flattened().as_ptr(), 2, HB_NO_OBJECT, &mut t));
        assert_eq!(take(t, "body", i64s), vec![-1, 0]);
        // the queries of a body
        ok(hb_body_panels(m, b, ptr::null(), 1, &mut t));
        assert_eq!(take(t, "centroids", f64s), flat3(&rm.body_panels(rb, &[], true).unwrap().centroids));
        ok(hb_body_mesh_quality(m, b, &mut t));
        assert_eq!(take(t, "aspect", f64s), rm.body_mesh_quality(rb).unwrap().aspect);
        ok(hb_body_tessellation(m, b, ptr::null(), 1, 1, 0, &mut t));
        assert_eq!(take(t, "normals", f64s), flat43(&rm.body_tessellation(rb, &[], [1, 1], false).unwrap().normals));
        // the free surface of the potential QTF: its panels, and the points of its zones
        let fp = rm.panels(&["free_surface"], true).unwrap();
        ok(hb_model_panels(m, c("free_surface").as_ptr(), 1, &mut t));
        let areas = f64s(&column(t, "areas"));
        assert_eq!(areas, fp.areas);
        assert!(i32s(&column(t, "body")).iter().all(|&x| x == -1));
        hb_table_free(t);
        for zone in ["panelled", "quadrature", "outer_circle"] {
            let p = rm.free_surface_points(zone).unwrap();
            ok(hb_model_free_surface_points(m, c(zone).as_ptr(), &mut t));
            assert_eq!(shape(&column(t, "points")), vec![p.points.len(), 3]);
            assert_eq!(f64s(&column(t, "points")), flat3(&p.points));
            assert_eq!(take(t, "weights", f64s), p.weights);
        }
        assert_eq!(hb_model_free_surface_points(m, c("far").as_ptr(), &mut t), HB_ERR_VALUE);
        assert!(last_error().contains("zone must be"));
        ok(hb_model_free(m));
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn every_query_of_a_mesh_gives_the_answer_of_the_rust_api() {
    let dir = scratch("meshes");
    unsafe {
        let m = c_box();
        let rm = rust_of(m, &dir);
        let mut mesh = 0;
        ok(hb_object(m, c("meshes/box").as_ptr(), &mut mesh));
        let rmesh = rm.object("meshes/box").unwrap();
        let mut t = ptr::null_mut();
        ok(hb_mesh_elements(m, mesh, &mut t));
        let rr = rm.mesh_elements(rmesh).unwrap();
        assert_eq!(i32s(&column(t, "element")), (1..=24).collect::<Vec<_>>());
        assert_eq!(take(t, "vertices", f64s), flat43(&rr.vertices));
        ok(hb_mesh_tessellation(m, mesh, 1, 1, 0, &mut t));
        assert_eq!(take(t, "vertices", f64s), flat43(&rm.mesh_tessellation(rmesh, [1, 1], false).unwrap().vertices));
        ok(hb_mesh_geometry_quality(m, mesh, &mut t));
        assert_eq!(take(t, "size", f64s), rm.mesh_geometry_quality(rmesh).unwrap().size);
        // a mesh written and read back: the geometry file and GDF
        for name in ["box.hbg", "box.gdf"] {
            let p = c(dir.join(name).to_str().unwrap());
            ok(hb_mesh_write(m, mesh, p.as_ptr(), ptr::null(), 0));
            assert_eq!(hb_mesh_write(m, mesh, p.as_ptr(), ptr::null(), 0), HB_ERR_EXISTS);
            let mut back = 0;
            ok(hb_model_load_mesh(
                m,
                c(&format!("from_{name}").replace('.', "_")).as_ptr(),
                p.as_ptr(),
                ptr::null(),
                &mut back,
            ));
            ok(hb_mesh_elements(m, back, &mut t));
            let v = take(t, "vertices", f64s);
            let want = flat43(&rr.vertices);
            assert!(v.iter().zip(&want).all(|(a, b)| (a - b).abs() <= 1e-12), "{name}");
        }
        // the options of a mesh file: a GDF of flat patches read with its discretization
        let mut root = 0;
        ok(hb_object(m, c("").as_ptr(), &mut root));
        let mut fp = 0;
        ok(hb_add(m, root, c("meshes").as_ptr(), ptr::null(), c("fp").as_ptr(), -1, &mut fp));
        ok(hb_set_text(m, fp, c("kind").as_ptr(), c("flat_patches").as_ptr()));
        let v = [[[-1.0, -1.0, -2.0], [1.0, -1.0, -2.0], [1.0, 1.0, -2.0], [-1.0, 1.0, -2.0]]];
        let (v, rows) = (v.as_flattened().as_flattened(), [2i64, 2, 2, 2, 2, 2]);
        ok(hb_set_reals(m, fp, c("vertices").as_ptr(), v.as_ptr(), [1usize, 4, 3].as_ptr(), 3));
        ok(hb_set_ints(m, fp, c("discretization").as_ptr(), rows.as_ptr(), [1usize, 6].as_ptr(), 2));
        let gdf = c(dir.join("fp.gdf").to_str().unwrap());
        ok(hb_mesh_write(m, fp, gdf.as_ptr(), ptr::null(), 0));
        let opts = HbMeshOptions {
            higher_order: 1,
            zero_trailer: 0,
            spl_path: ptr::null(),
            discretization: [3i32, 3, 2, 2, 2, 2].as_ptr(),
            discretization_rows: 1,
            panel_size: 0.0,
            spline_order: 0,
            outer_quadrature: 0,
            inner_quadrature: 0,
        };
        let mut patches = 0;
        ok(hb_model_load_mesh(m, c("patches").as_ptr(), gdf.as_ptr(), &opts, &mut patches));
        ok(hb_get(m, patches, c("kind").as_ptr(), &mut t));
        assert_eq!(take(t, "value", |col| text_at(col, 0)), "flat_patches");
        ok(hb_get(m, patches, c("discretization").as_ptr(), &mut t));
        assert_eq!(take(t, "value", i64s), vec![3, 3, 2, 2, 2, 2]);
        ok(hb_mesh_elements(m, patches, &mut t));
        assert_eq!(take(t, "vertices", f64s), v.to_vec());
        // the points of a patch (of the analytic cylinder)
        let cm = c_model();
        let rcm = rust_of(cm, &dir);
        let mut cyl = 0;
        ok(hb_object(cm, c("meshes/cyl").as_ptr(), &mut cyl));
        let uv = [[0.0, 0.0], [0.5, -0.5]];
        ok(hb_mesh_patch_points(cm, cyl, 1, uv.as_flattened().as_ptr(), 2, &mut t));
        let want = rcm.mesh_patch_points(rcm.object("meshes/cyl").unwrap(), 1, &uv).unwrap();
        assert_eq!(f64s(&column(t, "points")), flat3(&want.points));
        assert_eq!(take(t, "normals", f64s), flat3(&want.normals));
        ok(hb_model_free(cm));
        // a control surface (CSF, panels) and a free surface (FDF) read as meshes
        let csf = dir.join("s.csf");
        std::fs::write(&csf, "control surface\n0\n1 1\n1\n2 0 0 2 0 -1 2 1 -1 2 1 0\n").unwrap();
        let mut cs = 0;
        ok(hb_model_load_csf(m, c("cs").as_ptr(), c(csf.to_str().unwrap()).as_ptr(), 0, HB_NO_OBJECT, &mut cs));
        ok(hb_mesh_elements(m, cs, &mut t));
        assert_eq!(take(t, "element", i32s), vec![1]);
        let fdf = dir.join("fs.fdf");
        std::fs::write(
            &fdf,
            "a free surface\n3.0 RINNER\n2 4 NPF NTCL\n0 0.0 0 0 NAL\n1 2 2 1 0 0 1 1\n2 3 3 2 0 0 1 1\n",
        )
        .unwrap();
        let mut fs = 0;
        ok(hb_model_load_fdf(m, c("fs").as_ptr(), c(fdf.to_str().unwrap()).as_ptr(), c("xz_and_yz").as_ptr(), &mut fs));
        ok(hb_mesh_elements(m, fs, &mut t));
        let v = take(t, "vertices", f64s);
        assert_eq!((v.len(), v[3 * 4], v[2]), (24, 2.0, 0.0)); // panel 2 begins at x = 2; z = 0
        ok(hb_get(m, fs, c("symmetry").as_ptr(), &mut t));
        assert_eq!(take(t, "value", |col| text_at(col, 0)), "xz_and_yz");
        let mut other = 0;
        ok(hb_model_load_fdf(m, c("fs2").as_ptr(), c(fdf.to_str().unwrap()).as_ptr(), ptr::null(), &mut other));
        ok(hb_get(m, other, c("symmetry").as_ptr(), &mut t));
        assert_eq!(take(t, "value", |col| text_at(col, 0)), "none");
        assert_eq!(
            hb_model_load_fdf(m, c("fs3").as_ptr(), c(fdf.to_str().unwrap()).as_ptr(), c("z").as_ptr(), &mut other),
            HB_ERR_VALUE
        );
        // the surface of one role of the body as a mesh of its own
        let mut b = 0;
        ok(hb_object(m, c("bodies/b").as_ptr(), &mut b));
        let mut wet = 0;
        ok(hb_body_trimmed_mesh(m, b, c("wet").as_ptr(), c("wetted").as_ptr(), &mut wet));
        ok(hb_mesh_elements(m, wet, &mut t));
        assert_eq!(take(t, "vertices", f64s), flat43(&rr.vertices[..20]));
        ok(hb_model_free(m));
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_objects_and_items_of_a_model() {
    unsafe {
        let m = c_box();
        let mut root = 0;
        ok(hb_object(m, c("").as_ptr(), &mut root));
        let mut t = ptr::null_mut();
        let ids = |m: *mut HbModel, list: &str| {
            let mut t = ptr::null_mut();
            ok(hb_children(m, root, c(list).as_ptr(), &mut t));
            let col = column(t, "ids");
            let v = std::slice::from_raw_parts(col.data.cast::<u64>(), col.count).to_vec();
            hb_table_free(t);
            v
        };
        // a mesh added at the start, moved to the end; refused places
        let mut first = 0;
        ok(hb_add(m, root, c("meshes").as_ptr(), ptr::null(), c("first").as_ptr(), 0, &mut first));
        let box_id = ids(m, "meshes")[1];
        assert_eq!(ids(m, "meshes"), vec![first, box_id]);
        ok(hb_move(m, first, 1));
        assert_eq!(ids(m, "meshes"), vec![box_id, first]);
        assert_eq!(hb_move(m, first, -1), HB_ERR_ARGUMENT);
        assert_eq!(hb_move(m, first, 2), HB_ERR_ARGUMENT, "{}", last_error());
        let mut x = 0;
        assert_eq!(hb_add(m, root, c("meshes").as_ptr(), ptr::null(), c("x").as_ptr(), -2, &mut x), HB_ERR_ARGUMENT);
        assert!(last_error().contains("index must be -1"), "{}", last_error());
        // integer arrays and items: the discretization of a patch mesh, the threads
        ok(hb_set_text(m, first, c("kind").as_ptr(), c("flat_patches").as_ptr()));
        let v = [[[-1.0, -1.0, -2.0], [1.0, -1.0, -2.0], [1.0, 1.0, -2.0], [-1.0, 1.0, -2.0]]];
        ok(hb_set_reals(
            m,
            first,
            c("vertices").as_ptr(),
            v.as_flattened().as_flattened().as_ptr(),
            [1usize, 4, 3].as_ptr(),
            3,
        ));
        let rows = [2i64, 2, 2, 2, 2, 2];
        ok(hb_set_ints(m, first, c("discretization").as_ptr(), rows.as_ptr(), [1usize, 6].as_ptr(), 2));
        ok(hb_get(m, first, c("discretization").as_ptr(), &mut t));
        assert_eq!((shape(&column(t, "value")), take(t, "value", i64s)), (vec![1, 6], rows.to_vec()));
        assert_eq!(
            hb_set_ints(m, first, c("discretization").as_ptr(), rows.as_ptr(), [1usize, 5].as_ptr(), 2),
            HB_ERR_VALUE
        );
        let mut opt = 0;
        ok(hb_object(m, c("options").as_ptr(), &mut opt));
        ok(hb_get(m, opt, c("frequency_threads").as_ptr(), &mut t));
        assert_eq!(take(t, "value", i64s), vec![2]);
        // an element count beyond 2^40 is refused before any element is read
        let huge = [1usize << 21, 1 << 21];
        assert_eq!(hb_set_reals(m, first, c("vertices").as_ptr(), ptr::null(), huge.as_ptr(), 2), HB_ERR_ARGUMENT);
        assert!(last_error().contains("more than 2^40 elements"), "{}", last_error());
        let overflow = [usize::MAX, 2];
        assert_eq!(hb_set_reals(m, first, c("vertices").as_ptr(), ptr::null(), overflow.as_ptr(), 2), HB_ERR_ARGUMENT);
        // text arrays, a reset item, a rename that the references follow
        let (mut out, mut second) = (0, 0);
        ok(hb_object(m, c("first_order").as_ptr(), &mut out));
        ok(hb_object(m, c("second_order").as_ptr(), &mut second));
        ok(hb_get(m, second, c("qtf_potential_indirect").as_ptr(), &mut t));
        assert_eq!(take(t, "value", |col| *col.data.cast::<i32>()), 1);
        ok(hb_get(m, out, c("diffraction").as_ptr(), &mut t));
        assert_eq!(take(t, "kind", |col| text_at(col, 0)), "bool");
        ok(hb_reset(m, out, c("motion").as_ptr()));
        ok(hb_item_info(m, out, c("motion").as_ptr(), &mut t));
        assert_eq!(take(t, "is_set", |col| *col.data.cast::<i32>()), 0);
        ok(hb_rename(m, box_id, c("hull").as_ptr()));
        let mut b = 0;
        ok(hb_object(m, c("bodies/b").as_ptr(), &mut b));
        ok(hb_get(m, b, c("mesh").as_ptr(), &mut t));
        assert_eq!(take(t, "value", |col| text_at(col, 0)), "hull");
        // a list emptied, all or nothing: not the meshes while a body refers to one of them
        assert_eq!(hb_clear(m, root, c("meshes").as_ptr()), HB_ERR_VALUE);
        assert!(last_error().contains("refers to meshes/hull"), "{}", last_error());
        ok(hb_clear(m, root, c("bodies").as_ptr()));
        assert!(ids(m, "bodies").is_empty());
        let mut nb = 0;
        ok(hb_add(m, root, c("bodies").as_ptr(), ptr::null(), c("b").as_ptr(), -1, &mut nb));
        ok(hb_set_text(m, nb, c("mesh").as_ptr(), c("hull").as_ptr()));
        assert_eq!(ids(m, "bodies"), vec![nb]);
        assert_ne!(hb_set_real(m, b, c("yaw").as_ptr(), 1.0), HB_OK); // the old body is gone
        // every column of a table, by its place
        ok(hb_object_info(m, nb, &mut t));
        let mut count = 0;
        ok(hb_table_count(t, &mut count));
        let mut names = Vec::new();
        for k in 0..count {
            let mut col =
                HbColumn { name: ptr::null(), dtype: 0, ndim: 0, shape: ptr::null(), count: 0, data: ptr::null() };
            ok(hb_table_column(t, k, &mut col));
            names.push(CStr::from_ptr(col.name).to_str().unwrap().to_string());
        }
        let mut col =
            HbColumn { name: ptr::null(), dtype: 0, ndim: 0, shape: ptr::null(), count: 0, data: ptr::null() };
        assert_eq!(hb_table_column(t, count, &mut col), HB_ERR_ARGUMENT);
        hb_table_free(t);
        assert_eq!(names, ["type", "name", "list", "index", "path", "parent"]);
        // the location of a failure: kept by a call that has no place to write it
        assert_eq!(hb_object(m, c("bodies/none").as_ptr(), &mut x), HB_ERR_ARGUMENT);
        let before = last_error();
        assert_eq!(hb_last_error_location(ptr::null_mut()), HB_ERR_ARGUMENT);
        assert_eq!(last_error(), before);
        ok(hb_model_free(m));
    }
}

#[test]
fn the_life_of_a_model_and_its_results() {
    let dir = scratch("life");
    unsafe {
        let m = c_box();
        let mut t = ptr::null_mut();
        // a copy holds the inputs; cleared, nothing
        let mut copy = ptr::null_mut();
        ok(hb_model_copy(m, &mut copy));
        ok(hb_model_counts(copy, &mut t));
        assert_eq!(take(t, "bodies", i64s), vec![1]);
        ok(hb_model_clear(copy));
        let mut root = 0;
        ok(hb_object(copy, c("").as_ptr(), &mut root));
        ok(hb_children(copy, root, c("bodies").as_ptr(), &mut t));
        assert_eq!(column(t, "ids").count, 0);
        hb_table_free(t);
        ok(hb_model_free(copy));
        // a cancel request before a solve applies to it, and is cleared at its end
        ok(hb_model_cancel(m));
        assert_eq!(hb_model_solve(m, None, ptr::null_mut(), ptr::null_mut()), HB_ERR_CANCELLED);
        let mut r = ptr::null_mut();
        ok(hb_model_solve(m, None, ptr::null_mut(), &mut r));
        let mut state = -1;
        ok(hb_model_state(m, &mut state));
        assert_eq!(state, HB_STATE_CALCULATED);
        // what the results have, their diagnostics and meta data
        let rr = Model::load_results(&{
            let p = dir.join("r.hbr");
            ok(hb_model_save_results(m, c(p.to_str().unwrap()).as_ptr(), 0));
            p
        })
        .unwrap();
        let rr = rr.results().unwrap();
        ok(hb_results_available(r, &mut t));
        let want: Vec<String> = rr.available().iter().map(|q| q.name().to_string()).collect();
        assert_eq!(take(t, "names", texts_of), want);
        ok(hb_results_diagnostics(r, &mut t));
        assert_eq!(column(t, "message").count, rr.diagnostics().len());
        hb_table_free(t);
        ok(hb_results_meta(r, &mut t));
        assert_eq!(text_at(&column(t, "route"), 0), rr.route());
        assert_eq!(*column(t, "api_version").data.cast::<i32>(), hydrobem_core::build_info().api_version);
        assert_eq!(*column(t, "frequency_threads").data.cast::<i32>(), 2);
        assert_eq!(texts_of(&column(t, "environment_hooks")), rr.meta().environment_hooks);
        hb_table_free(t);
        ok(hb_results_body_columns(r, 0, &mut t));
        let (d, k) = rr.body_columns(0).unwrap();
        assert_eq!(i64s(&column(t, "coordinates")), d.iter().map(|&i| i as i64).collect::<Vec<_>>());
        assert_eq!(take(t, "radiation_columns", i64s), k.iter().map(|&i| i as i64).collect::<Vec<_>>());
        assert_eq!(d, (0..6).collect::<Vec<_>>());
        // a body the results do not have, as hb_results_select
        for body in [1, -1] {
            assert_eq!(hb_results_body_columns(r, body, &mut t), HB_ERR_ARGUMENT);
            let want = format!("body {body} does not exist (1 in these results)");
            assert!(last_error().contains(&want), "{}", last_error());
        }
        // the model of the results and the model solved
        for f in [hb_results_model, hb_results_solved_model] {
            let mut x = ptr::null_mut();
            ok(f(r, &mut x));
            ok(hb_model_counts(x, &mut t));
            assert_eq!(take(t, "bodies", i64s), vec![1]);
            ok(hb_model_free(x));
        }
        let mut hbr = ptr::null_mut();
        ok(hb_model_load_results(c(dir.join("r.hbr").to_str().unwrap()).as_ptr(), &mut hbr));
        ok(hb_model_state(hbr, &mut state));
        assert_eq!(state, HB_STATE_CALCULATED);
        ok(hb_model_free(hbr));
        // a reset discards the results; the inputs stay
        ok(hb_model_reset(m));
        ok(hb_model_state(m, &mut state));
        assert_eq!(state, HB_STATE_RESET);
        let mut none = ptr::null_mut();
        assert_eq!(hb_model_results(m, &mut none), HB_ERR_STATE);
        assert_eq!(hb_model_save_results(m, c(dir.join("x.hbr").to_str().unwrap()).as_ptr(), 0), HB_ERR_STATE);
        hb_results_free(r);
        ok(hb_model_free(m));
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_functions_without_a_model() {
    unsafe {
        let mut t = ptr::null_mut();
        ok(hb_build_info(&mut t));
        let b = hydrobem_core::build_info();
        assert_eq!(text_at(&column(t, "version"), 0), b.version);
        assert_eq!(text_at(&column(t, "version"), 0), env!("CARGO_PKG_VERSION"));
        assert_eq!(*column(t, "api_version").data.cast::<i32>(), b.api_version);
        hb_table_free(t);
        ok(hb_schema_lists(&mut t));
        let (ty, list) = (texts_of(&column(t, "type")), texts_of(&column(t, "list")));
        let always = i32s(&column(t, "always"));
        hb_table_free(t);
        let schema = hydrobem_core::schema();
        assert_eq!(ty.len(), schema.iter().map(|x| x.lists.len()).sum::<usize>());
        let k = (0..ty.len()).find(|&k| ty[k] == "Model" && list[k] == "second_order").unwrap();
        assert_eq!(always[k], 1);
        let mut k_ = 0.0;
        ok(hb_wavenumber(1.0, f64::INFINITY, 9.80665, &mut k_));
        assert_eq!(k_, hydrobem_core::wavenumber(1.0, f64::INFINITY, 9.80665).unwrap());
        assert_eq!(hb_wavenumber(-1.0, 10.0, 9.80665, &mut k_), HB_ERR_VALUE);
    }
}

/// The entries of a model or results file but "meta" (which holds the time of the save, to the
/// second), by name: the layout of results_file.rs (a 64-byte header with the directory's offset
/// and size, the directory a count and per entry its name, type, shape, offset, size and CRC).
fn entries_but_meta(bytes: &[u8]) -> Vec<(String, Vec<u8>)> {
    let u = |at: usize, n: usize| (0..n).fold(0u64, |v, k| v | (bytes[at + k] as u64) << (8 * k)) as usize;
    let (mut at, mut out) = (u(16, 8), Vec::new());
    let text = |at: &mut usize| {
        let n = u(*at, 4);
        *at += 4 + n;
        String::from_utf8(bytes[*at - n..*at].to_vec()).unwrap()
    };
    let count = u(at, 4);
    at += 4;
    for _ in 0..count {
        let name = text(&mut at);
        let rank = u(at + 1, 1);
        at += 2 + 8 * rank;
        let (offset, size) = (u(at, 8), u(at + 8, 8));
        at += 20;
        text(&mut at); // dims
        text(&mut at); // unit
        if name != "meta" {
            out.push((name, bytes[offset..offset + size].to_vec()));
        }
    }
    out
}

/// Calls f with a null out-pointer: `HB_ERR_ARGUMENT` naming it, and the model unchanged (its model
/// file the same entries; its meta, the time of the save, aside).
unsafe fn refused(m: *mut HbModel, file: &std::path::Path, what: &str, f: impl FnOnce() -> HbStatus) {
    let before = entries_but_meta(&std::fs::read(file).unwrap());
    assert_eq!(f(), HB_ERR_ARGUMENT, "{what}: {}", last_error());
    assert!(last_error().ends_with("is a null pointer"), "{what}: {}", last_error());
    ok(unsafe { hb_model_save(m, c(file.to_str().unwrap()).as_ptr(), 1) });
    let after = entries_but_meta(&std::fs::read(file).unwrap());
    assert!(before.iter().any(|(n, _)| n == "model"), "{what}: no model entry read");
    assert_eq!(after, before, "{what} changed the model");
}

#[test]
fn a_null_out_pointer_is_refused_before_the_call_changes_anything() {
    let dir = scratch("null-out");
    unsafe {
        let m = c_box();
        let f = dir.join("state.hbm");
        ok(hb_model_save(m, c(f.to_str().unwrap()).as_ptr(), 1));
        let (mut root, mut b, mut mesh) = (0, 0, 0);
        ok(hb_object(m, c("").as_ptr(), &mut root));
        ok(hb_object(m, c("bodies/b").as_ptr(), &mut b));
        ok(hb_object(m, c("meshes/box").as_ptr(), &mut mesh));
        let gdf = dir.join("box.gdf");
        ok(hb_mesh_write(m, mesh, c(gdf.to_str().unwrap()).as_ptr(), ptr::null(), 0));
        let mut r = ptr::null_mut();
        ok(hb_model_new_results(m, &mut r));
        let mut t = ptr::null_mut();
        ok(hb_model_counts(m, &mut t));
        let (u64n, i32n, tn) = (ptr::null_mut::<u64>(), ptr::null_mut::<i32>(), ptr::null_mut::<*mut HbTable>());
        let (sn, mn, rn) =
            (ptr::null_mut::<*mut c_char>(), ptr::null_mut::<*mut HbModel>(), ptr::null_mut::<*mut HbResults>());
        let path = c(gdf.to_str().unwrap());
        let (names, sym, shape, mapping) =
            ([c("radius"), c("draft")], c("xz_and_yz"), c("circular_cylinder"), c("uniform"));
        let dn: Vec<*const c_char> = names.iter().map(|s| s.as_ptr()).collect();
        let cylinder = HbAnalyticShape {
            shape: shape.as_ptr(),
            dimension_names: dn.as_ptr(),
            dimension_values: [1.0, 0.5].as_ptr(),
            dimensions: 2,
            discs: ptr::null(),
            disc_count: 0,
            parts: ptr::null(),
            part_count: 0,
            mapping: mapping.as_ptr(),
            discretization: ROWS.as_flattened().as_ptr(),
            discretization_rows: 2,
            reference_length: 1.0,
            symmetry: sym.as_ptr(),
            label: ptr::null(),
        };
        let (p3, uv) = ([0.0, 0.0, -0.5], [0.0, 0.0]);
        let g = c("g");
        // the functions that change the model when they succeed (the inputs valid)
        refused(m, &f, "hb_add", || hb_add(m, root, c("bodies").as_ptr(), ptr::null(), g.as_ptr(), -1, u64n));
        refused(m, &f, "hb_model_load_mesh", || hb_model_load_mesh(m, g.as_ptr(), path.as_ptr(), ptr::null(), u64n));
        refused(m, &f, "hb_model_load_fdf", || hb_model_load_fdf(m, g.as_ptr(), path.as_ptr(), ptr::null(), u64n));
        refused(m, &f, "hb_model_load_csf", || hb_model_load_csf(m, g.as_ptr(), path.as_ptr(), 0, HB_NO_OBJECT, u64n));
        refused(m, &f, "hb_model_mesh_from_hbg", || hb_model_mesh_from_hbg(m, g.as_ptr(), c("").as_ptr(), u64n));
        refused(m, &f, "hb_model_analytic_mesh", || hb_model_analytic_mesh(m, g.as_ptr(), &cylinder, u64n));
        refused(m, &f, "hb_body_trimmed_mesh", || hb_body_trimmed_mesh(m, b, g.as_ptr(), c("body").as_ptr(), u64n));
        // the functions that hand out a new model, results or text
        let hbm = c(f.to_str().unwrap());
        refused(m, &f, "hb_model_new", || hb_model_new(mn));
        refused(m, &f, "hb_model_load", || hb_model_load(hbm.as_ptr(), mn));
        refused(m, &f, "hb_model_load_results", || hb_model_load_results(c("none.hbr").as_ptr(), mn));
        refused(m, &f, "hb_model_copy", || hb_model_copy(m, mn));
        refused(m, &f, "hb_model_results", || hb_model_results(m, rn));
        refused(m, &f, "hb_model_new_results", || hb_model_new_results(m, rn));
        refused(m, &f, "hb_results_model", || hb_results_model(r, mn));
        refused(m, &f, "hb_results_solved_model", || hb_results_solved_model(r, mn));
        refused(m, &f, "hb_model_route", || hb_model_route(m, sn));
        refused(m, &f, "hb_mesh_hbg", || hb_mesh_hbg(m, mesh, sn));
        refused(m, &f, "hb_results_route", || hb_results_route(r, sn));
        // the queries of the model
        refused(m, &f, "hb_object", || hb_object(m, c("bodies/b").as_ptr(), u64n));
        refused(m, &f, "hb_object_info", || hb_object_info(m, b, tn));
        refused(m, &f, "hb_children", || hb_children(m, root, c("bodies").as_ptr(), tn));
        refused(m, &f, "hb_item_info", || hb_item_info(m, b, c("mass").as_ptr(), tn));
        refused(m, &f, "hb_get", || hb_get(m, b, c("mass").as_ptr(), tn));
        refused(m, &f, "hb_model_state", || hb_model_state(m, i32n));
        refused(m, &f, "hb_model_progress", || hb_model_progress(m, tn));
        refused(m, &f, "hb_model_diagnostics", || hb_model_diagnostics(m, tn));
        refused(m, &f, "hb_model_counts", || hb_model_counts(m, tn));
        refused(m, &f, "hb_model_motion_equation", || hb_model_motion_equation(m, tn));
        refused(m, &f, "hb_model_panels", || hb_model_panels(m, ptr::null(), 0, tn));
        refused(m, &f, "hb_model_free_surface_points", || hb_model_free_surface_points(m, c("panelled").as_ptr(), tn));
        refused(m, &f, "hb_model_tessellation", || hb_model_tessellation(m, ptr::null(), 2, 2, 0, tn));
        refused(m, &f, "hb_model_mesh_quality", || hb_model_mesh_quality(m, tn));
        refused(m, &f, "hb_model_mode_displacements", || {
            hb_model_mode_displacements(m, p3.as_ptr(), [0].as_ptr(), [1].as_ptr(), 1, tn)
        });
        refused(m, &f, "hb_model_memory_estimate", || hb_model_memory_estimate(m, tn));
        refused(m, &f, "hb_model_classify_points", || hb_model_classify_points(m, p3.as_ptr(), 1, 0, tn));
        refused(m, &f, "hb_body_hydrostatics", || hb_body_hydrostatics(m, b, tn));
        refused(m, &f, "hb_body_panels", || hb_body_panels(m, b, ptr::null(), 0, tn));
        refused(m, &f, "hb_body_mesh_quality", || hb_body_mesh_quality(m, b, tn));
        refused(m, &f, "hb_body_tessellation", || hb_body_tessellation(m, b, ptr::null(), 2, 2, 0, tn));
        refused(m, &f, "hb_mesh_geometry", || hb_mesh_geometry(m, mesh, tn));
        refused(m, &f, "hb_mesh_elements", || hb_mesh_elements(m, mesh, tn));
        refused(m, &f, "hb_mesh_tessellation", || hb_mesh_tessellation(m, mesh, 2, 2, 0, tn));
        refused(m, &f, "hb_mesh_geometry_quality", || hb_mesh_geometry_quality(m, mesh, tn));
        refused(m, &f, "hb_mesh_patch_points", || hb_mesh_patch_points(m, mesh, 1, uv.as_ptr(), 1, tn));
        // the results
        refused(m, &f, "hb_results_has", || hb_results_has(r, c("added_mass").as_ptr(), i32n));
        refused(m, &f, "hb_results_quantity", || hb_results_quantity(r, c("omegas").as_ptr(), ptr::null_mut()));
        refused(m, &f, "hb_results_select", || hb_results_select(r, c("added_mass").as_ptr(), -1, -1, tn));
        refused(m, &f, "hb_results_body_columns", || hb_results_body_columns(r, 0, tn));
        refused(m, &f, "hb_results_available", || hb_results_available(r, tn));
        refused(m, &f, "hb_results_diagnostics", || hb_results_diagnostics(r, tn));
        refused(m, &f, "hb_results_meta", || hb_results_meta(r, tn));
        refused(m, &f, "hb_results_normalized", || hb_results_normalized(r, c("omegas").as_ptr(), tn));
        // the tables and the functions without a model
        refused(m, &f, "hb_table_count", || hb_table_count(t, ptr::null_mut()));
        refused(m, &f, "hb_table_column", || hb_table_column(t, 0, ptr::null_mut()));
        refused(m, &f, "hb_table_find", || hb_table_find(t, c("bodies").as_ptr(), ptr::null_mut()));
        refused(m, &f, "hb_build_info", || hb_build_info(tn));
        refused(m, &f, "hb_schema_items", || hb_schema_items(tn));
        refused(m, &f, "hb_schema_lists", || hb_schema_lists(tn));
        refused(m, &f, "hb_quantities", || hb_quantities(tn));
        refused(m, &f, "hb_result_views", || hb_result_views(tn));
        refused(m, &f, "hb_result_groups", || hb_result_groups(tn));
        refused(m, &f, "hb_panel_roles", || hb_panel_roles(tn));
        refused(m, &f, "hb_wavenumber", || hb_wavenumber(1.0, f64::INFINITY, 9.80665, ptr::null_mut()));
        hb_table_free(t);
        // an index past the end of a list, as a negative one: the index and the count named
        for index in [1, -2] {
            assert_eq!(hb_move(m, b, index), HB_ERR_ARGUMENT);
            let want = format!("bodies: index {index} out of range (the list has 1 objects)");
            assert!(last_error().contains(&want), "{}", last_error());
        }
        ok(hb_move(m, b, 0));
        hb_results_free(r);
        ok(hb_model_free(m));
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_free_surface_queries_check_the_free_surface_alone() {
    unsafe {
        let m = c_box();
        let mut t = ptr::null_mut();
        ok(hb_model_free_surface_points(m, c("panelled").as_ptr(), &mut t));
        let valid = take(t, "weights", f64s);
        ok(hb_model_panels(m, c("free_surface").as_ptr(), 1, &mut t));
        let areas = take(t, "areas", f64s);
        // no frequencies: the model is refused, its free surface (the panel size given) is not
        let mut env = 0;
        ok(hb_object(m, c("environment").as_ptr(), &mut env));
        ok(hb_set_reals(m, env, c("omegas").as_ptr(), [0.0f64; 0].as_ptr(), [0usize].as_ptr(), 1));
        assert_eq!(hb_model_check(m, ptr::null_mut()), HB_ERR_VALUE);
        assert!(last_error().contains("omegas is empty"), "{}", last_error());
        ok(hb_model_free_surface_points(m, c("panelled").as_ptr(), &mut t));
        assert_eq!(take(t, "weights", f64s), valid);
        ok(hb_model_panels(m, c("free_surface").as_ptr(), 1, &mut t));
        assert_eq!(take(t, "areas", f64s), areas);
        // the auto panel size takes the frequencies: refused without them
        let mut second = 0;
        ok(hb_object(m, c("second_order").as_ptr(), &mut second));
        ok(hb_reset(m, second, c("free_surface_panel_size").as_ptr()));
        assert_eq!(hb_model_panels(m, c("free_surface").as_ptr(), 1, &mut t), HB_ERR_VALUE);
        let e = last_error();
        assert!(e.contains("no frequency between 0 and inf") && e.contains("free_surface_panel_size"), "{e}");
        ok(hb_model_free(m));
    }
}

#[test]
fn the_check_refuses_what_the_route_refuses_and_a_body_query_checks_the_body_alone() {
    unsafe {
        let m = c_box();
        let mut b = 0;
        ok(hb_object(m, c("bodies/b").as_ptr(), &mut b));
        // the mean drift by momentum conservation in a channel: valid, but no route
        let mut second = 0;
        ok(hb_object(m, c("second_order").as_ptr(), &mut second));
        ok(hb_set_bool(m, second, c("qtf_potential_indirect").as_ptr(), 0));
        ok(hb_set_bool(m, second, c("drift_momentum_conservation").as_ptr(), 1));
        let mut env = 0;
        ok(hb_object(m, c("environment").as_ptr(), &mut env));
        ok(hb_set_real(m, env, c("depth").as_ptr(), 10.0));
        ok(hb_set_text(m, env, c("domain").as_ptr(), c("channel").as_ptr()));
        ok(hb_set_real(m, env, c("channel_width").as_ptr(), 10.0));
        let mut route = ptr::null_mut();
        assert_eq!(hb_model_route(m, &mut route), HB_ERR_UNSUPPORTED);
        let refusal = last_error();
        assert!(refusal.contains("no solver route") && refusal.contains("channel"), "{refusal}");
        let mut t = ptr::null_mut();
        assert_eq!(hb_model_check(m, &mut t), HB_ERR_UNSUPPORTED);
        assert_eq!(last_error(), refusal);
        assert!(t.is_null());
        // without the drift the model has its route and passes the check
        ok(hb_set_bool(m, second, c("drift_momentum_conservation").as_ptr(), 0));
        ok(hb_model_check(m, ptr::null_mut()));
        // no frequencies: the model is refused, the hydrostatics of its body are not
        ok(hb_set_reals(m, env, c("omegas").as_ptr(), [0.0f64; 0].as_ptr(), [0usize].as_ptr(), 1));
        assert_eq!(hb_model_check(m, ptr::null_mut()), HB_ERR_VALUE);
        assert!(last_error().contains("omegas is empty"), "{}", last_error());
        ok(hb_body_hydrostatics(m, b, &mut t));
        hb_table_free(t);
        ok(hb_model_free(m));
    }
}
