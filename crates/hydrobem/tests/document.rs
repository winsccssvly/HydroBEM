//! The model as objects (document.rs, document_schema.rs, document_record.rs): the schema, the
//! object tree and its paths, the checks of a value alone and the binary record.
//! Most need the text functions (strip, py_repr, py_str_repr of text.rs).

use hydrobem_core::internals::document::{
    DType, Document, ItemKind, ItemValue, ObjectId, check_utf8, schema, schema_type, shape_text, symmetry_flags,
    symmetry_value, valid_utf8,
};
use hydrobem_core::internals::error::Result;

fn err<R>(f: impl FnOnce() -> Result<R>) -> String {
    match f() {
        Ok(_) => panic!("expected an error"),
        Err(e) => e.message().to_string(),
    }
}

fn real(x: f64) -> ItemValue {
    ItemValue::Real(x)
}
fn text(s: &str) -> ItemValue {
    ItemValue::Text(s.to_string())
}
fn reals(shape: Vec<i64>, data: Vec<f64>) -> ItemValue {
    ItemValue::Reals { shape, data }
}

fn id(d: &Document, path: &str) -> ObjectId {
    d.find(path).unwrap_or_else(|| panic!("no object '{path}'"))
}

#[test]
fn schema_types_and_lists() {
    let s = schema();
    let names: Vec<&str> = s.iter().map(|t| t.name).collect();
    assert_eq!(
        names,
        [
            "Model",
            "Environment",
            "Options",
            "FirstOrder",
            "SecondOrder",
            "Body",
            "Coupling",
            "Dipole",
            "DipoleDamper",
            "FixedDamper",
            "GeneralizedMode",
            "ModePiece",
            "Tank",
            "PressureSurface",
            "FreeSurfacePlate",
            "PressureMode",
            "DampingLid",
            "Mesh",
            "SplinePatch",
            "AnalyticPatch",
            "Tool"
        ]
    );
    let m = schema_type("Model");
    assert_eq!(m.list("bodies").unwrap().description, "The bodies, by name.");
    assert!(m.list("options").unwrap().always && m.list("bodies").unwrap().named);
    assert!(m.list("first_order").unwrap().always && m.list("second_order").unwrap().always);
    assert!(m.list("lids").unwrap().named && m.list("fixed_dampers").unwrap().optional_names);
    assert!(schema_type("SecondOrder").lists.is_empty());
    let b = schema_type("Body");
    assert_eq!(b.items.len(), 46);
    // the objects that can be suppressed, and only those
    for t in schema() {
        let can = [
            "Body",
            "Tank",
            "Dipole",
            "DipoleDamper",
            "FixedDamper",
            "PressureSurface",
            "FreeSurfacePlate",
            "DampingLid",
            "Coupling",
        ]
        .contains(&t.name);
        let e = t.item("enabled");
        assert_eq!(e.is_some(), can, "{}", t.name);
        assert!(e.is_none_or(|d| d.kind == ItemKind::Bool && d.fallback == Some("true")));
    }
    assert_eq!((b.item("mesh").unwrap().kind, b.item("mesh").unwrap().target), (ItemKind::Reference, "meshes"));
    assert_eq!((b.item("ranges").unwrap().kind, b.item("ranges").unwrap().shape), (ItemKind::Array, "N,2"));
    // the surfaces are items of the body, chosen by their switches
    let i = b.item("interior_surface").unwrap();
    assert_eq!((i.kind, i.fallback), (ItemKind::Choice, Some("\"detect\"")));
    let boundary = b.item("control_surface_boundary").unwrap();
    assert_eq!(
        (boundary.switch, boundary.switch_values),
        ("control_surface control_surface_outer", "automatic/polygon")
    );
    let ixx = b.item("Ixx").unwrap();
    assert_eq!((ixx.switch, ixx.switch_values), ("inertia_input", "components"));
    assert!(b.list("control_surface").is_none() && b.list("dipoles").unwrap().types == vec!["Dipole"]);
    let lid = schema_type("DampingLid");
    assert_eq!(lid.item("placed_with").unwrap().kind, ItemKind::Reference);
    assert!(schema_type("DipoleDamper").item("condition").is_none());
    let mass = b.item("mass").unwrap();
    assert_eq!((mass.kind, mass.unit, mass.choices.clone()), (ItemKind::Real, "kg", vec!["displaced"]));
    assert_eq!(shape_text(b.item("radii_of_gyration").unwrap()), "(3,) or (3, 3)");
    assert_eq!(shape_text(b.item("additional_stiffness").unwrap()), "(6, 6)");
}

#[test]
#[should_panic(expected = "no object type 'Ship'")]
fn schema_type_of_a_name_the_schema_does_not_have() {
    schema_type("Ship"); // a bug of the caller: the type names come from the schema
}

#[test]
fn utf8_and_values() {
    assert!(valid_utf8("hull \u{00e9}\u{10348}".as_bytes()));
    assert!(!valid_utf8(b"\xc0\xaf")); // overlong
    assert!(!valid_utf8(b"\xed\xa0\x80")); // a surrogate
    assert!(!valid_utf8(b"ab\xe2\x82")); // cut off
    assert!(!valid_utf8(b"\xf4\x90\x80\x80")); // above U+10FFFF
    assert_eq!(err(|| check_utf8("bodies/b: label: ", b"a\xff")), "bodies/b: label: the text is not valid UTF-8");
    assert_eq!(err(|| check_utf8("x: ", b"a\0b")), "x: the text has a NUL character");
    assert_eq!(check_utf8("", "ok".as_bytes()).unwrap(), "ok");

    let nan = real(f64::NAN);
    assert!(nan == nan.clone()); // bit for bit
    assert!(real(0.0) != real(-0.0));
    assert!(ItemValue::default() == ItemValue::Unset && ItemValue::Unset != real(0.0)); // unset: no value
    assert_eq!(symmetry_flags(&text("xz_and_yz")), [1, 1]);
    assert_eq!(symmetry_flags(&text("xz")), [0, 1]);
    assert_eq!(symmetry_value(1, 0).text(), "yz");
}

#[test]
fn defaults() {
    let d = Document::new();
    let env = d.get_obj(id(&d, "environment")).unwrap();
    assert!(env.get("depth").real().is_infinite());
    assert_eq!(env.get("gravity").real(), 9.80665);
    assert!(!env.is_set("gravity"));
    let opt = d.get_obj(id(&d, "options")).unwrap();
    let out = d.get_obj(id(&d, "first_order")).unwrap();
    let second = d.get_obj(id(&d, "second_order")).unwrap();
    let angles = second.get("drift_momentum_conservation_angles");
    assert!(angles.is_name() && angles.text() == "auto");
    assert_eq!(env.get("domain").text(), "open_sea");
    assert!(out.get("diffraction").boolean());
    assert_eq!(out.get("diffraction_formulation").text(), "auto");
    assert_eq!(second.get("free_surface_method").text(), "panels");
    assert_eq!(env.get("frequency_input").text(), "omegas");
    let f = d.get_obj(id(&d, "environment")).unwrap();
    let om = f.get("omegas");
    assert!(om.is_set() && om.is_array() && om.shape() == vec![0]);
    assert!(f.get("periods").is_set() && f.get("periods").shape() == vec![0]); // [] as omegas
    // an item the type does not have is a bug of the caller (the entry points check with item_def)
    assert_eq!(err(|| opt.item_def("nothing")), "options: no item 'nothing' in a Options");
}

#[test]
fn value_checks() {
    let mut d = Document::new();
    let env = id(&d, "environment");
    let r0 = d.revision();
    d.set(env, "density", real(1025.0)).unwrap();
    assert_eq!(d.get_obj(env).unwrap().get("density").real(), 1025.0);
    assert!(d.revision() > r0);
    let opt = id(&d, "options");
    d.set(opt, "inner_threads", ItemValue::Int(4)).unwrap();
    assert_eq!(err(|| d.set(opt, "inner_threads", ItemValue::Int(0))), "options: inner_threads must be in 1..1024");
    assert_eq!(d.get_obj(opt).unwrap().get("inner_threads").integer(), 4); // a failure leaves the item unchanged
    let f = id(&d, "environment");
    d.set(f, "omegas", reals(vec![2], vec![0.5, 1.0])).unwrap();
    d.set(f, "periods", reals(vec![1], vec![6.0])).unwrap();
    assert!(d.get_obj(f).unwrap().is_set("omegas")); // both kept: frequency_input chooses
    assert_eq!(
        err(|| d.set(f, "headings", reals(vec![2, 1], vec![0.0, 1.0]))),
        "environment: headings must have shape (H,), got (2, 1)"
    );
    assert_eq!(
        err(|| d.set(f, "headings", reals(vec![3], vec![0.0, 1.0]))),
        "environment: headings: the data do not fill the shape"
    );
    // integers given to a real array become reals
    d.set(f, "headings", ItemValue::Ints { shape: vec![2], data: vec![0, 90] }).unwrap();
    let h = d.get_obj(f).unwrap().get("headings");
    assert_eq!((h.dtype(), h.reals().to_vec()), (DType::Real, vec![0.0, 90.0]));
    let fp = id(&d, "first_order");
    assert_eq!(
        err(|| d.set(fp, "field_points", reals(vec![1, 3], vec![0.0, f64::INFINITY, 0.0]))),
        "first_order: field_points must be finite"
    );
    d.reset(f, "headings").unwrap();
    assert!(!d.get_obj(f).unwrap().is_set("headings"));
}

// A scalar holds the variant of its item's kind: an integer for a real item becomes a real (as in
// an array), another variant is refused, so that the typed reads and the record see the value.
#[test]
fn a_scalar_has_the_variant_of_its_kind() {
    let mut d = Document::new();
    let root = d.root_id();
    let b = d.add_child(root, "bodies", "", "hull", -1).unwrap();
    d.set(b, "yaw", ItemValue::Int(30)).unwrap();
    assert_eq!(d.get_obj(b).unwrap().get("yaw"), ItemValue::Real(30.0));
    let env = id(&d, "environment");
    d.set(env, "gravity", ItemValue::Int(10)).unwrap(); // the rule (positive) sees 10.0
    assert_eq!(d.get_obj(env).unwrap().get("gravity").real(), 10.0);
    let opt = id(&d, "options");
    assert_eq!(err(|| d.set(opt, "higher_order", text("yes"))), "options: higher_order must be a bool");
    assert_eq!(err(|| d.set(opt, "inner_threads", real(4.0))), "options: inner_threads must be an integer");
    assert_eq!(err(|| d.set(env, "gravity", ItemValue::Bool(true))), "environment: gravity must be a real number");
    assert_eq!(err(|| d.set(env, "domain", ItemValue::Int(1))), "environment: domain must be a string");
    assert_eq!(err(|| d.set(b, "mesh", ItemValue::Bool(false))), "bodies/hull: mesh must be a name");
    assert_eq!(
        err(|| d.set(b, "inertia_origin", ItemValue::Int(0))),
        "bodies/hull: inertia_origin must be one of center_of_gravity, rotation_center or a point (x, y, z)"
    );
    assert_eq!(err(|| d.set(b, "modes", text("heave"))), "bodies/hull: modes must be an array");
    assert_eq!(
        err(|| d.set(b, "mass", ItemValue::Bool(true))),
        "bodies/hull: mass must be a real number or one of displaced"
    );
    assert!(!d.get_obj(b).unwrap().is_set("mass")); // a failure leaves the item unchanged
}

#[test]
fn value_checks_with_text() {
    let mut d = Document::new();
    let env = id(&d, "environment");
    assert_eq!(
        err(|| d.set(env, "gravity", real(-1.0))),
        "environment: gravity must be a positive finite number, got -1.0"
    );
    assert_eq!(
        err(|| d.set(env, "domain", text("sea"))),
        "environment: domain must be one of open_sea, x_wall, y_wall, orthogonal_walls, channel, got 'sea'"
    );
    let root = d.root_id();
    let b = d.add_child(root, "bodies", "", "hull", -1).unwrap();
    assert_eq!(
        err(|| d.set(b, "modes", ItemValue::Texts { shape: vec![2], data: vec!["heave".into(), "surge".into()] })),
        "bodies/hull: modes must be unique and in the order surge, sway, heave, roll, pitch, yaw"
    );
    assert_eq!(err(|| d.set(b, "mesh", text("m"))), "bodies/hull: mesh: unknown mesh 'm'");
    d.set(b, "mass", ItemValue::Name("displaced".into())).unwrap();
    assert!(d.get_obj(b).unwrap().get("mass").is_name());
    // the mesh of a surface (a name) and its ranges of elements (sorted, without overlaps)
    let m = d.add_child(root, "meshes", "", "m", -1).unwrap();
    let ranges = |r: Vec<i64>| ItemValue::Ints { shape: vec![r.len() as i64 / 2, 2], data: r };
    d.set(b, "mesh", text("m")).unwrap();
    d.set(b, "ranges", ranges(vec![10, 12, 1, 4])).unwrap();
    assert_eq!(d.get_obj(b).unwrap().get("ranges").ints(), vec![1, 4, 10, 12]);
    assert_eq!(
        err(|| d.set(b, "ranges", ranges(vec![1, 4, 3, 6]))),
        "bodies/hull: ranges: overlapping element ranges (1, 4) and (3, 6)"
    );
    assert_eq!(
        err(|| d.set(b, "ranges", ranges(vec![0, 4]))),
        "bodies/hull: ranges: invalid element range (0, 4) (one-based, first <= last)"
    );
    assert_eq!(
        err(|| d.set(b, "ranges", ranges(vec![]))),
        "bodies/hull: ranges: the ranges select no elements (None for every element)"
    );
    d.rename(m, "hull_mesh").unwrap(); // the reference follows, the ranges stay
    assert_eq!(
        (d.get_obj(b).unwrap().get("mesh").text(), d.get_obj(b).unwrap().get("ranges").ints()),
        ("hull_mesh", &[1, 4, 10, 12][..])
    );
    d.reset(b, "ranges").unwrap();
    assert!(!d.get_obj(b).unwrap().get("ranges").is_set()); // every element
    let o = d.get_obj(b).unwrap();
    assert_eq!(o.get("name").text(), "hull");
    assert_eq!(err(|| o.item_text("name", b"\xff")), "bodies/hull: the text is not valid UTF-8");
    assert_eq!(err(|| o.item_text("mesh", b"m\xff")), "bodies/hull: mesh: the text is not valid UTF-8");
    assert_eq!(err(|| o.item_text("label", b"")), "bodies/hull: no item 'label' in a Body");
    assert_eq!(o.item_text("mesh", "m\u{e9}".as_bytes()).unwrap(), "m\u{e9}");
}

#[test]
fn tree_paths_and_names() {
    let mut d = Document::new();
    let root = d.root_id();
    let b = d.add_child(root, "bodies", "Body", "hull", -1).unwrap();
    d.add_child(b, "tanks", "", "", -1).unwrap();
    d.add_child(b, "tanks", "", "", -1).unwrap();
    let g = d.add_child(b, "generalized_modes", "", "bend", -1).unwrap();
    d.add_child(g, "pieces", "", "", -1).unwrap();
    d.add_child(root, "bodies", "", "barge", 0).unwrap();
    assert_eq!(d.get_obj(id(&d, "bodies/hull/tanks/2")).unwrap().path(), "bodies/hull/tanks/2");
    assert_eq!(
        d.get_obj(id(&d, "bodies/hull/generalized_modes/bend/pieces/1")).unwrap().prefix(),
        "bodies/hull/generalized_modes/bend/pieces/1: "
    );
    assert_eq!(d.get_obj(id(&d, "options")).unwrap().path(), "options");
    assert_eq!(d.root().children("bodies")[0].name(), "barge");
    for p in [
        "bodies/hull/tanks/3",
        "bodies/hull/tanks/0",
        "bodies/hull/tanks",
        "bodies/ship",
        "nothing",
        "bodies/hull/tanks/x",
    ] {
        assert!(d.find(p).is_none(), "{p}");
    }
    assert_eq!(err(|| d.add_child(root, "bodies", "", "hull", -1)), "duplicate body name 'hull'");
    assert_eq!(err(|| d.add_child(root, "bodies", "", " ", -1)), "bodies: names must not be blank");
    assert_eq!(
        err(|| d.add_child(root, "bodies", "", "a/b", -1)),
        "bodies: the name 'a/b' has a '/' or a control character"
    );
    assert_eq!(err(|| d.add_child(root, "bodies", "Mesh", "x", -1)), "bodies holds no Mesh");
    assert_eq!(err(|| d.add_child(root, "options", "", "", -1)), "options is already given");
    // a reference follows a rename; a referenced object stays
    let m = d.add_child(root, "meshes", "", "m1", -1).unwrap();
    d.set(id(&d, "bodies/hull"), "mesh", text("m1")).unwrap();
    assert_eq!(err(|| d.remove(m)), "meshes/m1: is used by bodies/hull: change or remove that reference first");
    d.rename(m, "m2").unwrap();
    assert_eq!(d.get_obj(id(&d, "bodies/hull")).unwrap().get("mesh").text(), "m2");
    d.set(id(&d, "bodies/hull"), "name", text("ship")).unwrap();
    assert!(d.find("bodies/ship/tanks/1").is_some());
    // move and remove
    let barge = id(&d, "bodies/barge");
    d.move_(barge, 1).unwrap();
    assert_eq!(d.root().children("bodies")[1].name(), "barge");
    assert_eq!(err(|| d.remove(root)), "the model itself cannot be removed");
    d.remove(barge).unwrap();
    assert!(!d.contains(barge)); // its id is refused from then on
    assert_eq!(d.root().children("bodies").len(), 1);
    // a new object in the slot of a removed one has another id
    let again = d.add_child(root, "bodies", "", "barge", -1).unwrap();
    assert!(again != barge && !d.contains(barge) && d.contains(again));
    d.remove(again).unwrap();
    // the clear of a list: refused while another object refers to one of its objects, nothing
    // removed; the references within the list go with it
    assert_eq!(
        err(|| d.clear_list(root, "meshes")),
        "bodies/ship refers to meshes/m2: the meshes are not cleared while an object refers to them"
    );
    assert_eq!(d.root().children("meshes")[0].name(), "m2");
    assert_eq!(err(|| d.clear_list(root, "environment")), "environment always holds its object: it is not emptied");
    // a copy of the items of an object, its children with them
    let c = d.add_child(root, "bodies", "", "copy", -1).unwrap();
    d.copy_items(c, id(&d, "bodies/ship")).unwrap();
    assert_eq!(d.get_obj(c).unwrap().children("tanks").len(), 2);
    assert_eq!(
        d.get_obj(id(&d, "bodies/copy/generalized_modes/bend/pieces/1")).unwrap().path(),
        "bodies/copy/generalized_modes/bend/pieces/1"
    );
    assert_eq!(d.get_obj(c).unwrap().get("mesh").text(), "m2");
    // the tanks of the copy cleared: those of the ship stay
    d.clear_list(c, "tanks").unwrap();
    assert!(d.get_obj(c).unwrap().children("tanks").is_empty() && d.find("bodies/copy/tanks/1").is_none());
    assert_eq!(d.get_obj(id(&d, "bodies/ship")).unwrap().children("tanks").len(), 2);
    // clear
    d.clear();
    assert!(d.root().children("bodies").is_empty());
    assert!(!d.contains(root) && !d.contains(m));
}

#[test]
fn record_round_trip_of_the_singles() {
    let mut d = Document::new();
    d.set(id(&d, "environment"), "depth", real(50.0)).unwrap();
    d.set(id(&d, "environment"), "omegas", reals(vec![3], vec![0.0, 1.0, f64::INFINITY])).unwrap();
    d.set(id(&d, "first_order"), "diffraction", ItemValue::Bool(false)).unwrap();
    let rec = d.record();
    let mut e = Document::new();
    e.load_record(&rec, "model.hbm").unwrap();
    assert_eq!(e.record(), rec);
    assert!(
        e.get_obj(id(&e, "environment")).unwrap().get("omegas")
            == d.get_obj(id(&d, "environment")).unwrap().get("omegas")
    );
    assert!(!e.get_obj(id(&e, "first_order")).unwrap().get("diffraction").boolean());
    // damaged records
    let mut f = Document::new();
    assert_eq!(err(|| f.load_record(&rec[..rec.len() - 3], "m.hbm")), "m.hbm: the record is damaged (cut short)");
    let mut g = Document::new();
    let mut longer = rec.clone();
    longer.push(0);
    assert_eq!(err(|| g.load_record(&longer, "m.hbm")), "m.hbm: the model record has data after its end");
    let mut h = Document::new();
    let mut v2 = rec.clone();
    v2[0] = 5; // a record of the version before
    assert_eq!(
        err(|| h.load_record(&v2, "m.hbm")),
        "m.hbm: model record version 5; this HydroBEM reads version 13 (no conversion)"
    );
    // a copy has the same objects (and ids)
    let c = d.clone();
    assert_eq!(c.record(), rec);
    assert_eq!(c.find("options"), d.find("options"));
}

#[test]
fn record_round_trip_of_a_tree() {
    let mut d = Document::new();
    let root = d.root_id();
    let m = d.add_child(root, "meshes", "", "m", -1).unwrap();
    d.set(m, "kind", text("spline_patches")).unwrap();
    let b = d.add_child(root, "bodies", "", "b", -1).unwrap();
    d.set(b, "mesh", text("m")).unwrap();
    d.set(b, "position", reals(vec![3], vec![1.0, 2.0, 3.0])).unwrap();
    let t = d.add_child(b, "tanks", "", "", -1).unwrap();
    let r14 = ItemValue::Ints { shape: vec![1, 2], data: vec![1, 4] };
    d.set(t, "mesh", text("m")).unwrap();
    d.set(t, "ranges", r14).unwrap();
    d.set(b, "interior_surface", text("graded")).unwrap();
    // a second body without its interior surface: none after the record too
    let b2 = d.add_child(root, "bodies", "", "b2", -1).unwrap();
    d.set(b2, "interior_surface", text("none")).unwrap();
    let l = d.add_child(root, "lids", "", "lid", -1).unwrap();
    d.set(l, "placed_with", text("b")).unwrap();
    let g = d.add_child(root, "tool", "", "gui", -1).unwrap();
    d.set(g, "text", text("{\"x\": 1}")).unwrap();
    let rec = d.record();
    let mut e = Document::new();
    e.load_record(&rec, "model.hbm").unwrap();
    assert_eq!(e.record(), rec);
    let tank = e.get_obj(id(&e, "bodies/b/tanks/1")).unwrap();
    assert_eq!((tank.get("mesh").text(), tank.get("ranges").ints()), ("m", &[1, 4][..]));
    assert_eq!(e.get_obj(id(&e, "bodies/b")).unwrap().get("interior_surface").text(), "graded");
    assert_eq!(e.get_obj(id(&e, "bodies/b2")).unwrap().get("interior_surface").text(), "none");
    assert_eq!(e.get_obj(id(&e, "lids/lid")).unwrap().get("placed_with").text(), "b");
    assert_eq!(e.get_obj(id(&e, "meshes/m")).unwrap().get("kind").text(), "spline_patches");
}

#[test]
fn references_to_a_body_follow_renames_and_keep_what_they_name() {
    let mut d = Document::new();
    let root = d.root_id();
    let a = d.add_child(root, "bodies", "", "a", -1).unwrap();
    let b = d.add_child(root, "bodies", "", "b", -1).unwrap();
    let c = d.add_child(a, "couplings", "", "", -1).unwrap();
    d.set(c, "body", text("b")).unwrap();
    // a rename of the body follows
    d.rename(b, "barge").unwrap();
    assert_eq!(d.get_obj(c).unwrap().get("body").text(), "barge");
    // the record keeps it (set after the objects it names)
    let mut e = Document::new();
    e.load_record(&d.record(), "model.hbm").unwrap();
    assert_eq!(e.get_obj(id(&e, "bodies/a/couplings/1")).unwrap().get("body").text(), "barge");
    // what is named stays
    assert_eq!(
        err(|| d.remove(b)),
        "bodies/barge: is used by bodies/a/couplings/1: change or remove that reference first"
    );
    d.remove(c).unwrap();
    d.remove(b).unwrap();
}
