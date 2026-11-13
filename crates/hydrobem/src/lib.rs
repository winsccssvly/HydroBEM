//! hydrobem_core: the HydroBEM engine, a frequency-domain panel method (docs/development/architecture.md).
//!
//! The public API is at the crate root: a [`Model`] owns the model document (objects addressed by
//! [`ObjectId`], their items read and set as [`Value`]s), the problem built from it, the [`Results`] of
//! its last solve (quantities by [`QuantityId`]) and its state; a solve reports [`Progress`] and is
//! cancelled through a [`CancelToken`]; failures are [`Error`]s.
// an ignored Result would drop its error
#![deny(unused_must_use)]

// The kernels call AVX2 and FMA intrinsics from ordinary functions (the four-lane wave kernels, the
// finite-depth tiles, the float32 Rankine tiers): that is sound only when the whole crate is
// compiled for those features (.cargo/config.toml), so a build without them is refused here
// rather than left with undefined behaviour. A RUSTFLAGS or CARGO_ENCODED_RUSTFLAGS of its own
// replaces the flags of .cargo/config.toml and must name them too. rustdoc (cfg doc, doctest)
// only reads the code and runs none of it, with its own flags.
#[cfg(not(any(doc, doctest, all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))))]
compile_error!(
    "HydroBEM is built for x86-64 with AVX2 and FMA: -C target-feature=+avx2,+fma (.cargo/config.toml); \
     a RUSTFLAGS of its own replaces those flags"
);

pub(crate) mod analytic;
pub(crate) mod analytic_shapes;
pub(crate) mod binary;
pub(crate) mod bspline;
pub(crate) mod build_info;
pub(crate) mod codes;
pub(crate) mod constants;
pub(crate) mod control_drift;
pub(crate) mod dispersion;
pub(crate) mod document;
pub(crate) mod document_build;
pub(crate) mod document_enabled;
pub(crate) mod document_mesh;
pub(crate) mod document_record;
pub(crate) mod document_rules;
pub(crate) mod document_schema;
pub(crate) mod env_hooks;
pub(crate) mod error;
pub(crate) mod field_points;
pub(crate) mod free_surface_panels;
pub(crate) mod gdf;
pub(crate) mod geometry;
pub(crate) mod green_channel;
pub(crate) mod green_fin;
pub(crate) mod green_inf;
pub(crate) mod hbg;
pub(crate) mod heading_pairs;
pub(crate) mod ho_frequency;
pub(crate) mod ho_geometry;
pub(crate) mod ho_internal;
pub(crate) mod ho_merge;
pub(crate) mod ho_sheets;
pub(crate) mod ho_solver;
pub(crate) mod ho_static;
pub(crate) mod hydrostatics;
pub(crate) mod inspect;
pub(crate) mod interior;
pub(crate) mod json;
pub(crate) mod linalg;
pub(crate) mod lo_frequency;
pub(crate) mod lo_geometry;
pub(crate) mod lo_static;
pub(crate) mod lo_velocity;
pub(crate) mod low_order;
pub(crate) mod memory;
pub(crate) mod mesh;
pub(crate) mod mesh_desc;
pub(crate) mod model;
pub(crate) mod model_codes;
pub(crate) mod model_meshes;
pub(crate) mod model_queries;
pub(crate) mod model_solve;
pub(crate) mod model_trim;
pub(crate) mod model_types;
pub(crate) mod motion;
pub(crate) mod panel_integrals;
pub(crate) mod parallel;
pub(crate) mod patch_geometry;
pub(crate) mod plate_geometry;
pub(crate) mod plate_rules;
pub(crate) mod polygon;
pub(crate) mod potential_qtf;
pub(crate) mod preprocess;
pub(crate) mod pressure_drift;
pub(crate) mod problem;
pub(crate) mod progress;
pub(crate) mod quadratic_qtf;
pub(crate) mod quadrature;
pub(crate) mod quality;
pub(crate) mod quantities;
pub(crate) mod response;
pub(crate) mod result_tree;
pub(crate) mod results;
pub(crate) mod results_file;
pub(crate) mod route;
pub(crate) mod rules;
pub(crate) mod solve_common;
pub(crate) mod solve_driver;
pub(crate) mod solver;
pub(crate) mod tables;
pub(crate) mod tanks;
pub(crate) mod text;
pub(crate) mod wamit_files;

pub use build_info::{BuildInfo, build_info};
pub use codes::{
    ControlSurfaceKind, ElementType, ModelState, PanelRole, PointStatus, QualityFlags, QuantityId, Severity, Stage,
    Warning,
};
pub use document::{
    DType, Document, ItemDef, ItemKind, ItemValue, ListDef, Obj, ObjectId, TypeDef, chosen, is_reference_item, schema,
};
pub use error::{Diagnostic, Error, ErrorKind, Location, Result};
pub use hydrostatics::Hydrostatics;
pub use inspect::{MeshElements, MeshGeometry, Panels, Tessellation};
pub use model::{
    AnalyticShape, BodyCounts, CancelToken, Counts, FreeSurfacePoints, FrequencyProgress, FrequencyState, ItemInfo,
    Limit, LimitProgress, MeshFile, Model, MotionEquation, ObjectInfo, PatchPoints, Phase, PointClass, Progress,
    ProgressAt, ProgressFn, ProgressMonitor, SecondOrderProgress, SecondOrderStep, SolveJob, SolveProgress, Task,
    Value, WorkerProgress, WorkerState, wavenumber,
};
pub use model_codes::Domain;
pub use num_complex::{Complex32, Complex64};
pub use quality::MeshQuality;
pub use result_tree::{
    Owner, ResultGroup, ResultView, Rows, Selected, Selection, result_groups, result_view, result_views,
};
pub use results::{QuantityData, QuantityShape, Results, RunMeta, WamitOptions};
pub use route::{Method, Route};
pub use solver::MemoryEstimate;

/// The engine's modules for its component tests (tests/), not an API.
#[doc(hidden)]
pub mod internals {
    macro_rules! modules {
        ($($m:ident),* $(,)?) => {
            $(pub mod $m {
                pub use crate::$m::*;
            })*
        };
    }
    modules!(
        analytic,
        analytic_shapes,
        binary,
        bspline,
        build_info,
        codes,
        constants,
        control_drift,
        dispersion,
        document,
        document_build,
        document_enabled,
        document_mesh,
        document_rules,
        document_schema,
        env_hooks,
        error,
        gdf,
        geometry,
        green_channel,
        green_fin,
        green_inf,
        ho_frequency,
        ho_geometry,
        ho_internal,
        ho_sheets,
        ho_solver,
        ho_static,
        inspect,
        interior,
        linalg,
        lo_frequency,
        lo_geometry,
        lo_static,
        lo_velocity,
        low_order,
        memory,
        mesh,
        mesh_desc,
        model,
        model_codes,
        motion,
        panel_integrals,
        parallel,
        patch_geometry,
        preprocess,
        pressure_drift,
        problem,
        potential_qtf,
        progress,
        quadratic_qtf,
        quadrature,
        quality,
        quantities,
        response,
        result_tree,
        results,
        results_file,
        route,
        solve_common,
        solve_driver,
        solver,
        tanks,
        text,
        wamit_files,
    );
}
