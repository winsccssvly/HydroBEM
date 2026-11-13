//! The input and output group: the Fortran fields of the numeric files, the text of the input files (Python's repr of a double, the free-format
//! numbers, the lines and tokens), the geometry files read back from their writers, CRC-32 and the
//! results file with its lazy reading, and the quantity table.

use hydrobem_core::internals::mesh_desc::PatchKind;
use hydrobem_core::{ElementType, ErrorKind, QuantityId};
use std::fs;
use std::path::PathBuf;

use hydrobem_core::internals::error::Result;
use hydrobem_core::internals::gdf::{
    PatchOptions, gdf_text, gdf_text_patch, read_low_order_gdf, read_patch_gdf, read_spl, spl_text,
};
use hydrobem_core::internals::mesh_desc::*;
use hydrobem_core::internals::quantities::quantity_info;
use hydrobem_core::internals::results::parts::{diagnostics_mut, meta_mut, quantities, with_route};
use hydrobem_core::internals::results::{RunMeta, put};
use hydrobem_core::internals::results_file::{crc32, load_quantity, open_results, results_meta, save_results};
use hydrobem_core::internals::text::{is_number, parse_number, py_repr, split_lines, strip, tokens};
use hydrobem_core::internals::wamit_files::{fortran_e, fortran_i};

fn rejects<R>(f: impl FnOnce() -> Result<R>) -> bool {
    f().is_err()
}

/// A fresh folder for the files of a test.
fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("hydrobem-io-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

// the Fortran E14.6 and 1PE14.6 fields of the WAMIT-format numeric files
#[test]
#[allow(clippy::approx_constant)] // 6.2831853 is a test value of the formatter, not 2 pi
fn fortran_fields() {
    let cases: &[(f64, bool, &str)] = &[
        (1.0, false, "  0.100000E+01"),
        (-123.4567, false, " -0.123457E+03"),
        (0.0, false, "  0.000000E+00"),
        (-0.0, false, "  0.000000E+00"),
        (6.2831853, false, "  0.628319E+01"),
        (0.99999951, false, "  0.100000E+01"),
        (1e-5, false, "  0.100000E-04"),
        (9.0e98, false, "  0.900000E+99"),
        (1.0, true, "  1.000000E+00"),
        (-1.0, true, " -1.000000E+00"),
        (9.9999996, true, "  1.000000E+01"),
        (0.0, true, "  0.000000E+00"),
        (123456.78, true, "  1.234568E+05"),
        (-2.5e-99, true, " -2.500000E-99"),
        (0.1f32 as f64, true, "  1.000000E-01"),
    ];
    for &(x, one_p, text) in cases {
        assert_eq!(fortran_e(x, one_p, 14, 6).unwrap(), text, "fortran_e {text}");
    }
    assert_eq!(fortran_e(1.0, true, 15, 6).unwrap(), "   1.000000E+00", "fortran_e width 15");
    assert_eq!(fortran_i(12, 6).unwrap(), "    12", "fortran_i");
    // a three-digit exponent does not fit (0.1E+100 for 0P), nor a non-finite value or a long integer
    for x in [1e100, -3e101, 1e-100, f64::NAN, f64::INFINITY] {
        assert!(rejects(|| fortran_e(x, true, 14, 6)), "fortran_e rejects {x}");
    }
    assert!(rejects(|| fortran_e(1e99, false, 14, 6)), "fortran_e 0P rejects 1e99");
    assert!(rejects(|| fortran_i(1234567, 6)), "fortran_i rejects 7 digits");
}

// Python's repr and the E fields (correctly rounded, %.*e) of values of every magnitude (the table
// below, written by scripts/gen_repr_cases.py)
#[test]
fn repr_and_fields_against_python() {
    for &(b, repr, one_p, zero_p) in REPR_CASES {
        let v = f64::from_bits(b);
        assert_eq!(py_repr(v), repr, "repr of {b:#x}");
        assert_eq!(parse_number(repr).unwrap().to_bits(), b, "round trip of {repr}");
        for (field, one) in [(one_p, true), (zero_p, false)] {
            match fortran_e(v, one, 14, 6) {
                Ok(s) => assert_eq!(s, field, "E field of {repr} (1P {one})"),
                Err(_) => assert_eq!(field, "-", "E field of {repr} (1P {one}) is written"),
            }
        }
    }
}

#[test]
fn free_format_numbers() {
    for t in ["1", "+1", "-1.", ".5", "1.5e3", "1.5E+3", "2d-3", "2D3", "007", "1.e5"] {
        assert!(is_number(t), "{t} is a number");
    }
    for t in ["", ".", "+", "e3", "1e", "1e+", "1.2.3", "1,5", "nan", "inf", "0x10", "1_0", "--1", "1 "] {
        assert!(!is_number(t), "{t} is not a number");
    }
    assert_eq!(parse_number("2d-3").unwrap(), 2e-3);
    assert_eq!(parse_number("-1.5D+2").unwrap(), -150.0);
    assert_eq!(parse_number("1e400").unwrap(), f64::INFINITY);
    assert_eq!(parse_number("1e-400").unwrap(), 0.0);
    assert_eq!(parse_number(".1").unwrap(), 0.1);
    assert_eq!(parse_number("1.").unwrap(), 1.0);
    assert_eq!(parse_number("9007199254740993").unwrap(), 9007199254740992.0); // the tie to even
    let e = parse_number("1x").unwrap_err();
    assert_eq!(e.kind(), ErrorKind::Value);
    assert_eq!(e.message(), "invalid number '1x'");
}

#[test]
fn lines_and_whitespace() {
    assert_eq!(split_lines("a\r\nb\rc\n\nd"), vec!["a", "b", "c", "", "d"]);
    assert_eq!(split_lines("x\u{2029}y\u{85}z\x1e"), vec!["x", "y", "z"]);
    assert_eq!(tokens("1.0, 2.0 ,3.0\t\u{3000}x"), vec!["1.0", "2.0", "3.0", "x"]);
    assert_eq!(tokens(",,"), Vec::<String>::new());
    assert_eq!(strip("\u{a0} label \u{2000}\r"), "label");
}

fn flat_mesh() -> LowOrderMeshDesc {
    LowOrderMeshDesc {
        panel_count: 2,
        vertices: vec![
            0.0,
            0.0,
            -1.0,
            1.0,
            0.0,
            -1.0,
            1.0,
            1.0,
            -1.0,
            0.0,
            1.0,
            -1.0, // panel 1
            0.1,
            1.0 / 3.0,
            -2.5e-7,
            1e16,
            -0.0,
            -1.0,
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
        ],
        reference_length: 2.0,
        gravity: 9.81,
        symmetry_x: 0,
        symmetry_y: 1,
    }
}

#[test]
fn gdf_round_trip() {
    let d = scratch("gdf");
    let m = flat_mesh();
    let path = d.join("hull.gdf").to_string_lossy().into_owned();
    fs::write(&path, gdf_text(&m, "hull")).unwrap();
    let g = read_low_order_gdf(&path, false).unwrap();
    assert_eq!(g.label, "hull");
    assert_eq!((g.ulen, g.gravity, g.isx, g.isy, g.count), (2.0, 9.81, 0, 1, 2));
    assert_eq!(
        g.vertices.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        m.vertices.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    // a trailing record of zeros only when accepted
    fs::write(&path, gdf_text(&m, "hull") + "0 0 0\n").unwrap();
    let e = read_low_order_gdf(&path, false).unwrap_err();
    assert_eq!(e.message(), format!("{path}: trailing record of zeros after NPAN = 2 (pass zero_trailer = true)"));
    assert_eq!(read_low_order_gdf(&path, true).unwrap().count, 2);
    fs::write(&path, "x\n1 9.81\n0 0\n1\n1 2 3\n").unwrap();
    let e = read_low_order_gdf(&path, false).unwrap_err();
    assert_eq!(e.message(), format!("{path}: the file ends after 0 of NPAN = 1 records"));
    fs::write(&path, "x\n1 9.81\n2 0\n1\n").unwrap();
    let e = read_low_order_gdf(&path, false).unwrap_err();
    assert_eq!(e.message(), format!("{path}: ISX must be 0 or 1, not 2"));
    fs::write(&path, "x\n1 9.81\n0.5 0\n1\n").unwrap();
    let e = read_low_order_gdf(&path, false).unwrap_err();
    assert_eq!(e.message(), format!("{path}: ISX must be an integer, got 0.5"));
    let e = read_low_order_gdf(&d.join("none.gdf").to_string_lossy(), false).unwrap_err();
    assert_eq!(e.kind(), ErrorKind::Io);
}

#[test]
fn patch_gdf_and_spl_round_trip() {
    let d = scratch("patch");
    let pm = PatchMeshDesc {
        geometry: PatchKind::Flat,
        patch_count: 1,
        vertices: vec![0.0, 0.0, -1.0, 2.0, 0.0, -1.0, 2.0, 1.0, -1.0, 0.0, 1.0, -1.0],
        discretization: vec![PatchDiscretization { nu: 2, nv: 1, ku: 3, kv: 3, iquo: 4, iqui: 5 }],
        reference_length: 1.0,
        gravity: 9.80665,
        ..PatchMeshDesc::default()
    };
    let gdf = d.join("p.gdf").to_string_lossy().into_owned();
    let spl = d.join("p.spl").to_string_lossy().into_owned();
    fs::write(&gdf, gdf_text_patch(&pm, "flat\nmore").unwrap()).unwrap();
    fs::write(&spl, spl_text(&pm, "")).unwrap();
    assert!(fs::read_to_string(&spl).unwrap().starts_with("HydroBEM mesh\n"));
    let o = PatchOptions { spl: spl.clone(), ..PatchOptions::default() };
    let g = read_patch_gdf(&gdf, &o, false).unwrap();
    assert!(g.higher_order && g.geometry == PatchKind::Flat && g.count == 1 && g.label == "flat");
    assert_eq!(g.vertices, pm.vertices);
    assert_eq!(g.discretization, pm.discretization);
    // the pairs completed by their keywords
    let disc = read_spl(&spl, 1, 0, 0, 0).unwrap();
    assert_eq!(disc, pm.discretization);
    let e = read_spl(&spl, 1, 3, 0, 0).unwrap_err();
    assert_eq!(
        e.message(),
        format!(
            "{spl}: expected 3 data lines (3 pairs for each of 1 patches; pairs given by keywords: KU KV), found 4; a pair \
         must be given either in the file or by its keyword, never both"
        )
    );
    // the panel-size rule: floor(L / size) + 1 panels along each edge
    let o = PatchOptions { has_panel_size: true, panel_size: 0.5, ..PatchOptions::default() };
    let g = read_patch_gdf(&gdf, &o, false).unwrap();
    assert_eq!(g.discretization, vec![PatchDiscretization { nu: 5, nv: 3, ku: 3, kv: 3, iquo: 3, iqui: 4 }]);
    // an analytic shape
    fs::write(&gdf, "cyl\n1 9.80665\n1 1\n2 -1\n1\n2.0 3.0 0   RADIUS DRAFT INONUMAP\n").unwrap();
    let o = PatchOptions { has_panel_size: true, panel_size: 1.0, ..PatchOptions::default() };
    let g = read_patch_gdf(&gdf, &o, false).unwrap();
    assert_eq!(g.geometry, PatchKind::Analytic);
    assert_eq!(g.analytic.len(), 2);
    assert_eq!(g.analytic[0].p, [1.0, 0.0, 1.0, -3.0]);
    // the quarter circle of radius 2 (pi) and the side of 3 m
    assert_eq!((g.discretization[0].nu, g.discretization[0].nv), (4, 4));
}

#[test]
fn crc32_values() {
    assert_eq!(crc32(b"123456789", 0), 0xCBF43926);
    assert_eq!(crc32(&[0u8; 32], 0), 0x190A55AD);
    assert_eq!(crc32(&[0xffu8; 32], 0), 0xFF6CAB0B);
}

#[test]
fn quantity_table() {
    let a = quantity_info(QuantityId::AddedMass);
    assert_eq!(
        (a.name, a.dtype, a.dims),
        ("added_mass", ElementType::Float32, "frequency,coordinate,radiation_column")
    );
    let last = quantity_info(QuantityId::ALL[QuantityId::COUNT - 1]);
    assert_eq!(last.name, "hydrostatics_metacentric_height");
    // every name once
    let mut names: Vec<&str> = QuantityId::ALL.iter().map(|&q| quantity_info(q).name).collect();
    names.sort();
    names.dedup();
    assert_eq!(names.len(), QuantityId::COUNT);
}

#[test]
fn results_file_round_trip() {
    let d = scratch("hbr");
    let path = d.join("sub").join("run.hbr").to_string_lossy().into_owned();
    let mut r = with_route("low_order");
    *meta_mut(&mut r) = results_meta("low_order", 2, 1, &[0.25, 1.5]);
    put(&mut r, QuantityId::Omegas, vec![3], &[0.5f64, 1.0, 2.0]);
    put(&mut r, QuantityId::AddedMass, vec![1, 2, 2], &[1.0f32, 2.0, 3.0, 4.0]);
    let at = hydrobem_core::Location { body: Some(0), ..Default::default() };
    diagnostics_mut(&mut r).push(hydrobem_core::Diagnostic::warning_at(
        hydrobem_core::Warning::MeshOrientation,
        at,
        "a warning",
    ));
    save_results(&r, &path, false).unwrap(); // the folder made
    let e = save_results(&r, &path, false).unwrap_err();
    assert_eq!(e.kind(), ErrorKind::Exists);
    assert_eq!(e.message(), format!("{path} exists"));
    save_results(&r, &path, true).unwrap();
    let bytes = fs::read(&path).unwrap();
    assert_eq!(&bytes[..8], &[0x89, b'H', b'B', b'R', b'\r', b'\n', 0x1a, b'\n']);
    let back = open_results(&path).unwrap();
    assert_eq!(back.route(), "low_order");
    let meta = back.meta();
    assert!(meta.present && meta.frequency_threads == 2 && meta.engine_seconds == vec![0.25, 1.5]);
    assert_eq!(meta.api_version, hydrobem_core::internals::build_info::API_VERSION);
    assert_eq!(back.diagnostics().len(), 1);
    assert_eq!(back.diagnostics()[0].message, "a warning");
    // results without a model are saved, but they make no model
    let e = hydrobem_core::Model::load_results(std::path::Path::new(&path)).err().unwrap();
    assert_eq!(e.message(), format!("{path}: the results file holds no model"));
    let q = &quantities(&back)[&QuantityId::AddedMass];
    assert!(q.pending && q.data().is_empty() && q.byte_count() == 16);
    load_quantity(&back, q, "added_mass").unwrap();
    assert_eq!(q.data(), quantities(&r)[&QuantityId::AddedMass].data());
    // a copy of opened results writes the quantities it has not read
    let path2 = d.join("copy.hbr").to_string_lossy().into_owned();
    save_results(&back, &path2, false).unwrap();
    let again = open_results(&path2).unwrap();
    let q = &quantities(&again)[&QuantityId::Omegas];
    load_quantity(&again, q, "omegas").unwrap();
    assert_eq!(q.data(), quantities(&r)[&QuantityId::Omegas].data());
    // a damaged entry is found by its CRC
    let mut bad = fs::read(&path).unwrap();
    let at = quantities(&back)[&QuantityId::AddedMass].offset as usize;
    bad[at] ^= 1;
    fs::write(&path, &bad).unwrap();
    let damaged = open_results(&path).unwrap();
    let e = load_quantity(&damaged, &quantities(&damaged)[&QuantityId::AddedMass], "added_mass").unwrap_err();
    assert_eq!(e.message(), format!("{path}: the entry q/added_mass is damaged (CRC-32)"));
    // not a results file
    fs::write(&path, vec![0u8; 100]).unwrap();
    let e = open_results(&path).err().unwrap();
    assert_eq!(e.message(), format!("{path}: not a HydroBEM model or results file"));
    let _ = RunMeta::default();
}

// Generated by scripts/gen_repr_cases.py (Python's repr and %.*e; never edited by hand): bits, repr,
// 1PE14.6, 0PE14.6 (- when it does not fit)
#[rustfmt::skip]
const REPR_CASES: &[(u64, &str, &str, &str)] = &[
    (0x3defd3cf4e4665d2, "2.3157468229247737e-10", "  2.315747E-10", "  0.231575E-09"),
    (0xbbdc79efc388d17b, "-2.4120376814162823e-20", " -2.412038E-20", " -0.241204E-19"),
    (0x12a294bc602ab29c, "6.579641480536607e-219", "-", "-"),
    (0x0a76589383346ba8, "2.906722284453078e-258", "-", "-"),
    (0x3f42840f356fdc04, "0.0005650591145912602", "  5.650591E-04", "  0.565059E-03"),
    (0x389eef2f79d0ed1d, "5.8180986015919546e-36", "  5.818099E-36", "  0.581810E-35"),
    (0x28959ad6a19e2f62, "3.5092365449780766e-113", "-", "-"),
    (0xe969650a1b9f42f3, "-6.0744922313679114e+199", "-", "-"),
    (0x92c0e0d4c718e1f3, "-2.39067468889135e-218", "-", "-"),
    (0xc2309a74df6407f7, "-71310827364.03111", " -7.131083E+10", " -0.713108E+11"),
    (0xfcb4fc74c396af32, "-5.235627179900934e+292", "-", "-"),
    (0x085ae8c69639a735, "2.0374452967747454e-268", "-", "-"),
    (0x3a82ceb64541dbce, "7.596276124375404e-27", "  7.596276E-27", "  0.759628E-26"),
    (0x3a970264143f27cc, "1.858679649701464e-26", "  1.858680E-26", "  0.185868E-25"),
    (0x799db6559031fb3e, "6.583721617799052e+277", "-", "-"),
    (0xc5db956e2ec2bfa5, "-3.4146982140781424e+28", " -3.414698E+28", " -0.341470E+29"),
    (0x60c4d484f5ce5038, "1.4299479114418088e+158", "-", "-"),
    (0xabdc9cc7df83ffa0, "-2.0930330115259042e-97", " -2.093033E-97", " -0.209303E-96"),
    (0xcdaf9df15c1216c8, "-1.6648274728028485e+66", " -1.664827E+66", " -0.166483E+67"),
    (0x3d5818959d1f7841, "3.424252231074357e-13", "  3.424252E-13", "  0.342425E-12"),
    (0x0abd5d711b2641b3, "6.111584516935763e-257", "-", "-"),
    (0x28c9ad966c67dad3, "3.336676594016154e-112", "-", "-"),
    (0x6a39a63ea1fb9834, "5.02613537100905e+203", "-", "-"),
    (0xf928a4687d40b3da, "-4.265852840622474e+275", "-", "-"),
    (0x54e0a04389729c66, "7.273059466008778e+100", "-", "-"),
    (0x5d178e78747a54bf, "2.805234883182723e+140", "-", "-"),
    (0x5f997a216334579b, "3.3358563219170064e+152", "-", "-"),
    (0x650be66fa60dad38, "5.652933103732993e+178", "-", "-"),
    (0x1df313d95e29f180, "2.0705448728179672e-164", "-", "-"),
    (0xa43b868b8cb7ab96, "-3.787025366014939e-134", "-", "-"),
    (0x2f86a3754073967c, "9.546385244167348e-80", "  9.546385E-80", "  0.954639E-79"),
    (0x6f0c01030b284bb3, "8.29251159845966e+226", "-", "-"),
    (0x4995f6c713416dfb, "3.1348032850843857e+46", "  3.134803E+46", "  0.313480E+47"),
    (0xcaf3e77060920a33, "-1.1915187844417098e+53", " -1.191519E+53", " -0.119152E+54"),
    (0x9c71e00e1da339d0, "-1.1563618727415912e-171", "-", "-"),
    (0xe9f894f03d207ce3, "-3.010583111626828e+202", "-", "-"),
    (0xcbea798da8074264, "-5.193285139718622e+57", " -5.193285E+57", " -0.519329E+58"),
    (0x31ff256ce5718235, "7.220478136428582e-68", "  7.220478E-68", "  0.722048E-67"),
    (0x1ea9223520eaca2e, "5.586598082479412e-161", "-", "-"),
    (0xe6e376f18d7cfedc, "-4.234609029654424e+187", "-", "-"),
    (0xf52a9bb7334a7d89, "-2.497024887604564e+256", "-", "-"),
    (0x2aeeb86e158e2771, "6.858020285320859e-102", "-", "-"),
    (0x83541178f5ef8847, "-1.2568798633164662e-292", "-", "-"),
    (0x3ef5df19db7da349, "2.08582766555749e-05", "  2.085828E-05", "  0.208583E-04"),
    (0x16e5ea03dcb33bba, "2.2903240312592658e-198", "-", "-"),
    (0x04c6b52bbda8fbec, "1.193018686916227e-285", "-", "-"),
    (0xd81580753faaffe2, "-2.1180340657421303e+116", "-", "-"),
    (0x5efaa6b7a36d102c, "3.407809273175705e+149", "-", "-"),
    (0x8754318966e0e57e, "-2.3330046559236487e-273", "-", "-"),
    (0x391d58ea8d9aea4c, "1.4130224925759462e-33", "  1.413022E-33", "  0.141302E-32"),
    (0x5446cdc616678bbb, "9.74172586825236e+97", "  9.741726E+97", "  0.974173E+98"),
    (0xfea799a791f12fde, "-1.2643915769722706e+302", "-", "-"),
    (0x300a392f8a2cd7e8, "2.8308693000632085e-77", "  2.830869E-77", "  0.283087E-76"),
    (0x37de571bf6a2e42e, "1.39315689461946e-39", "  1.393157E-39", "  0.139316E-38"),
    (0x709309ce75292168, "1.8916681827916873e+234", "-", "-"),
    (0xa64985baa1cfcedc, "-3.0162890221359822e-124", "-", "-"),
    (0x21ef4d668ecda0f6, "3.133494904147158e-145", "-", "-"),
    (0xcf769c5480868cc6, "-6.391933764170844e+74", " -6.391934E+74", " -0.639193E+75"),
    (0x95dfcff493ab3851, "-2.536637216501011e-203", "-", "-"),
    (0xdedcafb50de9bd11, "-9.170082976594111e+148", "-", "-"),
    (0xb1d3b50e6b9cea47, "-1.142160276373689e-68", " -1.142160E-68", " -0.114216E-67"),
    (0x0fb87d8a26c74999, "6.161962528941486e-233", "-", "-"),
    (0x08febbee407bce9d, "2.382892180853261e-265", "-", "-"),
    (0x0e843c2beb1600ac, "9.710858472226679e-239", "-", "-"),
    (0xeaabd859b5359221, "-6.984191382268977e+205", "-", "-"),
    (0x1963258ccb19d872, "2.200220274239243e-186", "-", "-"),
    (0xba47bcb8007f87a8, "-5.992107324868653e-28", " -5.992107E-28", " -0.599211E-27"),
    (0x10d9d054a9323bf9, "1.702609648029679e-227", "-", "-"),
    (0xdab6d37c48bd9ad8, "-9.888949653388532e+128", "-", "-"),
    (0x5f13fc4132b71e46, "1.022186280362403e+150", "-", "-"),
    (0xe34aa0dd3d2e26ec, "-2.0098897970091228e+170", "-", "-"),
    (0xd015322bb855a1b2, "-6.135816907666352e+77", " -6.135817E+77", " -0.613582E+78"),
    (0x5bf97b8787264158, "1.1576059060883117e+135", "-", "-"),
    (0xc7227665ab2373a7, "-4.793136139803349e+34", " -4.793136E+34", " -0.479314E+35"),
    (0x4a2e29bbb1a85c34, "2.2041651728523798e+49", "  2.204165E+49", "  0.220417E+50"),
    (0x4c4e2619333aad2c, "3.784944460837012e+59", "  3.784944E+59", "  0.378494E+60"),
    (0xa79cdcda3b46425f, "-7.153502580887384e-118", "-", "-"),
    (0x5786ad918f6ef916, "4.363080019926334e+113", "-", "-"),
    (0x53ba3a019bae85f3, "2.1882675746064687e+95", "  2.188268E+95", "  0.218827E+96"),
    (0x2cf6c7fcd1bf1703, "4.368556533274916e-92", "  4.368557E-92", "  0.436856E-91"),
    (0xb3cfff6e7bcdbe3e, "-3.9824534256797277e-59", " -3.982453E-59", " -0.398245E-58"),
    (0x86383ed175e2e92c, "-1.068545657708891e-278", "-", "-"),
    (0x0d1da1420dd30a25, "1.6950945988795218e-245", "-", "-"),
    (0x4a12d51003f6c426, "6.880850415842988e+48", "  6.880850E+48", "  0.688085E+49"),
    (0x8bd9fabdef4af5b8, "-1.4174043608243084e-251", "-", "-"),
    (0x7e460695c24a9381, "1.843808702227286e+300", "-", "-"),
    (0x8d835082c4061ab8, "-1.414349956209413e-243", "-", "-"),
    (0x463ebe5490bb0aae, "2.4357492605339175e+30", "  2.435749E+30", "  0.243575E+31"),
    (0x258f58ac10fe7a4d, "9.044387967628548e-128", "-", "-"),
    (0x90cb5c2adc47cd73, "-9.022976876660724e-228", "-", "-"),
    (0x8cf3b44c3bef8ba4, "-2.8181513113244534e-246", "-", "-"),
    (0xec6ed5df9b8ee0d4, "-2.0761423614018925e+214", "-", "-"),
    (0x4c0537f8cb71e258, "1.6648945289715583e+58", "  1.664895E+58", "  0.166489E+59"),
    (0xfdc5a2c414c893ad, "-7.074875092725303e+297", "-", "-"),
    (0x8e6418acc2874e0d, "-2.4110788971815126e-239", "-", "-"),
    (0x6815ac5c42eb963c, "2.4720766437962953e+193", "-", "-"),
    (0x753d40b347f8ffea, "5.490385122606839e+256", "-", "-"),
    (0x1142c125369d77ee, "1.583354447765384e-225", "-", "-"),
    (0x93312691d9b5a737, "-3.109453276387464e-216", "-", "-"),
    (0x4080f7cc9e4ba4cc, "542.9749112996737", "  5.429749E+02", "  0.542975E+03"),
    (0xeb6223efb4c026f6, "-1.8636883795571043e+209", "-", "-"),
    (0x379c59be9bca9194, "8.136213050365403e-41", "  8.136213E-41", "  0.813621E-40"),
    (0xcc6c9814663ce2c0, "-1.4359026529388348e+60", " -1.435903E+60", " -0.143590E+61"),
    (0x29d4d86ac10541f6, "3.550351454465178e-107", "-", "-"),
    (0x128d2342c25927e2, "2.5794611712423885e-219", "-", "-"),
    (0xb5bf4129d30cd58f, "-8.353604340902591e-50", " -8.353604E-50", " -0.835360E-49"),
    (0xcebeb1909e85dd9d, "-2.1183953114634671e+71", " -2.118395E+71", " -0.211840E+72"),
    (0xa7e8e226d2cc3104, "-1.9735198233366533e-116", "-", "-"),
    (0x52ea051dc3a7c212, "2.650182319890879e+91", "  2.650182E+91", "  0.265018E+92"),
    (0x812cdebf7fa309d4, "-5.262387283707427e-303", "-", "-"),
    (0x14d196f44408f895, "2.1401320457361574e-208", "-", "-"),
    (0xc713b0a9d7d81598, "-2.55592001905941e+34", " -2.555920E+34", " -0.255592E+35"),
    (0xb13e3de64bb22e45, "-1.7116249553331827e-71", " -1.711625E-71", " -0.171162E-70"),
    (0x6b9a4955b6faab1c, "2.160474590913734e+210", "-", "-"),
    (0x50b5751dc6ba44b9, "6.360594143744832e+80", "  6.360594E+80", "  0.636059E+81"),
    (0x3b567a32d5d4b594, "7.437127347295083e-23", "  7.437127E-23", "  0.743713E-22"),
    (0xb9e1640b5825bf7d, "-6.859460366108081e-30", " -6.859460E-30", " -0.685946E-29"),
    (0x0d1e40996679108e, "1.7307029329130622e-245", "-", "-"),
    (0x09e5572ef528924e, "5.42175120970979e-261", "-", "-"),
    (0xc7f837e538617881, "-5.1506715783679136e+38", " -5.150672E+38", " -0.515067E+39"),
    (0x0000000000000001, "5e-324", "-", "-"),
    (0x0000000000000002, "1e-323", "-", "-"),
    (0x000fffffffffffff, "2.225073858507201e-308", "-", "-"),
    (0x0010000000000000, "2.2250738585072014e-308", "-", "-"),
    (0x7fefffffffffffff, "1.7976931348623157e+308", "-", "-"),
    (0x4341c37937e08000, "1e+16", "  1.000000E+16", "  0.100000E+17"),
    (0x4341c37937e07fff, "9999999999999998.0", "  1.000000E+16", "  0.100000E+17"),
    (0x4340000000000000, "9007199254740992.0", "  9.007199E+15", "  0.900720E+16"),
    (0x3f1a36e2eb1c432d, "0.0001", "  1.000000E-04", "  0.100000E-03"),
    (0x3f1a36e2eb1c432c, "9.999999999999999e-05", "  1.000000E-04", "  0.100000E-03"),
    (0x3ee4f8b588e368f1, "1e-05", "  1.000000E-05", "  0.100000E-04"),
    (0x3fb999999999999a, "0.1", "  1.000000E-01", "  0.100000E+00"),
    (0x3fc999999999999a, "0.2", "  2.000000E-01", "  0.200000E+00"),
    (0x3fd3333333333333, "0.3", "  3.000000E-01", "  0.300000E+00"),
    (0x3fd5555555555555, "0.3333333333333333", "  3.333333E-01", "  0.333333E+00"),
    (0x419d6f3454800000, "123456789.125", "  1.234568E+08", "  0.123457E+09"),
    (0x43e0000000000000, "9.223372036854776e+18", "  9.223372E+18", "  0.922337E+19"),
    (0x3e10000000000000, "9.313225746154785e-10", "  9.313226E-10", "  0.931323E-09"),
    (0x4480f0cf064dd592, "1e+22", "  1.000000E+22", "  0.100000E+23"),
    (0x44b52d02c7e14af6, "1e+23", "  1.000000E+23", "  0.100000E+24"),
    (0x4011666666666666, "4.35", "  4.350000E+00", "  0.435000E+01"),
    (0x4005666666666666, "2.675", "  2.675000E+00", "  0.267500E+01"),
    (0x3fe0000000000000, "0.5", "  5.000000E-01", "  0.500000E+00"),
    (0x8000000000000000, "-0.0", "  0.000000E+00", "  0.000000E+00"),
    (0x7e6ddd4baa009303, "1e+301", "-", "-"),
    (0x3e8421f5f40d8376, "1.5e-07", "  1.500000E-07", "  0.150000E-06"),
    (0x4057bc11a9c29830, "94.93857807162635", "  9.493858E+01", "  0.949386E+02"),
    (0x408640cacaaac3c6, "712.0990193692203", "  7.120990E+02", "  0.712099E+03"),
    (0xc07519bf17a1bae8, "-337.60915339637586", " -3.376092E+02", " -0.337609E+03"),
    (0xc088bf2be2d6722a, "-791.8964287522879", " -7.918964E+02", " -0.791896E+03"),
    (0x406a272acfdfacb0, "209.22397607504308", "  2.092240E+02", "  0.209224E+03"),
    (0x408011a22e070cee, "514.2041893530675", "  5.142042E+02", "  0.514204E+03"),
    (0xc0705fc9fdbed53e, "-261.9868142561353", " -2.619868E+02", " -0.261987E+03"),
    (0x408988c4ec9a5ad6, "817.0961544093104", "  8.170962E+02", "  0.817096E+03"),
    (0x406f66f2ae9b1c28, "251.2171242742054", "  2.512171E+02", "  0.251217E+03"),
    (0xc076d4ffa493ca74, "-365.31241281252665", " -3.653124E+02", " -0.365312E+03"),
    (0x406abad2d8ba04e0, "213.83823810893864", "  2.138382E+02", "  0.213838E+03"),
    (0xc079043ddb13689c, "-400.2651015051031", " -4.002651E+02", " -0.400265E+03"),
    (0xc08a38955450884d, "-839.0729147235485", " -8.390729E+02", " -0.839073E+03"),
    (0xc085b60d0f0914da, "-694.7563763341416", " -6.947564E+02", " -0.694756E+03"),
    (0xc08650164f6ceb80, "-714.0108936795586", " -7.140109E+02", " -0.714011E+03"),
    (0x405a4f0a787d52d0, "105.23501407850858", "  1.052350E+02", "  0.105235E+03"),
    (0x40760bc132414460, "352.7346670674797", "  3.527347E+02", "  0.352735E+03"),
    (0xc082bef8801f1f6d, "-599.8713381225076", " -5.998713E+02", " -0.599871E+03"),
    (0x407cb657a824a5f8, "459.39640058818577", "  4.593964E+02", "  0.459396E+03"),
    (0x40884d65cbd03fba, "777.6747051495979", "  7.776747E+02", "  0.777675E+03"),
    (0x4072265ae232e974, "290.3971883763072", "  2.903972E+02", "  0.290397E+03"),
    (0x405c563797990ac0, "113.34714307733066", "  1.133471E+02", "  0.113347E+03"),
    (0x407eb380455f0d88, "491.2188161576273", "  4.912188E+02", "  0.491219E+03"),
    (0xc07e4565809ca090, "-484.3372808569193", " -4.843373E+02", " -0.484337E+03"),
    (0x4075b7cbe04fc154, "347.4872744670331", "  3.474873E+02", "  0.347487E+03"),
    (0x40505d39374460f0, "65.45661718061069", "  6.545662E+01", "  0.654566E+02"),
    (0xc08264c1d93e06cf, "-588.5946526380375", " -5.885947E+02", " -0.588595E+03"),
    (0x4062513b63d39b80, "146.5384997494657", "  1.465385E+02", "  0.146538E+03"),
    (0xc072f39b2debf7a8, "-303.22538559126633", " -3.032254E+02", " -0.303225E+03"),
    (0xc07a5bbb09aa8b62, "-421.7331635152715", " -4.217332E+02", " -0.421733E+03"),
    (0xc0662f7491b3d380, "-177.4829796325903", " -1.774830E+02", " -0.177483E+03"),
    (0x408d209f69dc0a4a, "932.0778386297836", "  9.320778E+02", "  0.932078E+03"),
    (0x40599ccd51240ab0, "102.45003155250265", "  1.024500E+02", "  0.102450E+03"),
    (0xc07b88c8fc554340, "-440.549068768556", " -4.405491E+02", " -0.440549E+03"),
    (0xc07e42b248706384, "-484.16852611448144", " -4.841685E+02", " -0.484169E+03"),
    (0xc06bf8faa458ce44, "-223.7805959448325", " -2.237806E+02", " -0.223781E+03"),
    (0x408ac96e55b131b2, "857.1788743823329", "  8.571789E+02", "  0.857179E+03"),
    (0x4083769126133a70, "622.8208734037598", "  6.228209E+02", "  0.622821E+03"),
    (0x408f35efb457e294, "998.7420431962987", "  9.987420E+02", "  0.998742E+03"),
    (0x408c8414a27b4d28, "912.5100755341455", "  9.125101E+02", "  0.912510E+03"),
    (0x0f70042d3c8b446c, "2.5186393870397615e-234", "-", "-"),
    (0x6b7743af367b5c6f, "4.7802291050174946e+209", "-", "-"),
    (0x3a9edb0576f54af6, "2.4924916518749096e-26", "  2.492492E-26", "  0.249249E-25"),
    (0x16cf50254170c0bd, "8.181604889784833e-199", "-", "-"),
    (0x29d977656659cc30, "4.337394803921109e-107", "-", "-"),
    (0x59d679de8e159832, "5.943171376743487e+124", "-", "-"),
    (0x23e014b01f406335, "6.913849950514554e-136", "-", "-"),
    (0x7583784bc085075a, "1.1693651686496827e+258", "-", "-"),
    (0x203162b970bdcc7b, "1.2966804258182848e-153", "-", "-"),
    (0x0bd84590b2100fcd, "1.3242333748877851e-251", "-", "-"),
    (0x7e5195977293da44, "2.944027183187628e+300", "-", "-"),
    (0x49c4cad2b5cdf86b, "2.374058427830313e+47", "  2.374058E+47", "  0.237406E+48"),
    (0x1c789530fcc3652a, "1.590279470138663e-171", "-", "-"),
    (0x00000000206778c8, "2.685997963e-315", "-", "-"),
    (0x7e63400153d30b2b, "6.44580074745703e+300", "-", "-"),
    (0x1255d1f8d7a20e80, "2.4145820896985587e-220", "-", "-"),
    (0x11c2d1b697b14b7d, "4.067374895769051e-223", "-", "-"),
    (0x1a26a7c1cb309a88, "1.0663566301857469e-182", "-", "-"),
    (0x3f1663a7d870c38d, "8.54083246941818e-05", "  8.540832E-05", "  0.854083E-04"),
    (0x10861a610e2e5d59, "4.555807715460601e-229", "-", "-"),
    (0x30c856e4fa7f14c0, "1.076221093912841e-73", "  1.076221E-73", "  0.107622E-72"),
    (0x5b652fe8d24e2fb4, "1.8798386725801354e+132", "-", "-"),
    (0x531c942cc7de5abd, "2.3286424266995082e+92", "  2.328642E+92", "  0.232864E+93"),
    (0x282c57e6bd5eb5fc, "3.5966905362051616e-115", "-", "-"),
    (0x546cbbc50738395b, "4.9099464353994404e+98", "  4.909946E+98", "  0.490995E+99"),
    (0x2b758d5e888f4d35, "2.4633919961218292e-99", "  2.463392E-99", "  0.246339E-98"),
    (0x5088f754f85d8c98, "9.250820942733671e+79", "  9.250821E+79", "  0.925082E+80"),
    (0x6a83b68287ba653d, "1.2361131522786783e+205", "-", "-"),
    (0x7a70e779dd8cc960, "6.136922872356857e+281", "-", "-"),
    (0x69de5a7eb1acc10a, "9.293639593337973e+201", "-", "-"),
    (0x4e8101acea13a836, "1.4671857231234754e+70", "  1.467186E+70", "  0.146719E+71"),
    (0x09fe3271dcd893d8, "1.5343635763389147e-260", "-", "-"),
    (0x2af1280c688c3118, "7.659997594328884e-102", "-", "-"),
    (0x13cb95b969bac899, "2.5606125740727995e-213", "-", "-"),
    (0x00000005fe45f00a, "1.2717661171e-313", "-", "-"),
    (0x4ca2c43a6bd882d2, "1.5078314611114928e+61", "  1.507831E+61", "  0.150783E+62"),
    (0x3bcb222b536e2a6c, "1.1491473056651432e-20", "  1.149147E-20", "  0.114915E-19"),
    (0x575040a0614a4f7d, "3.908563387975802e+112", "-", "-"),
    (0x04e7714a3500050f, "4.926503026333872e-285", "-", "-"),
    (0x01a10340174dc8dc, "7.938665442077985e-301", "-", "-"),
];
