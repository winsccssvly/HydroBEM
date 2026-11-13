//! The build of the library: the version, commit, build type and compiler written by
//! build.rs and the version of the linear-algebra library compiled into it (faer, linalg.rs).

use std::sync::OnceLock;

/// The version of the engine's interface, recorded in the meta data of a results file.
pub const API_VERSION: i32 = 56;

mod generated {
    include!(concat!(env!("OUT_DIR"), "/build_info_gen.rs"));
}

#[derive(Clone, Debug, Default)]
pub struct BuildInfo {
    pub version: String,
    pub commit: String,
    pub compiler: String,
    /// the linear-algebra library and its version (faer; linalg.rs)
    pub linear_algebra: String,
    pub build_type: String,
    pub avx2: bool,
    /// the version of the engine's interface, also in the meta data of a results file
    pub api_version: i32,
}

/// The build, made once on first use (thread-safe).
pub fn build_info() -> &'static BuildInfo {
    static INFO: OnceLock<BuildInfo> = OnceLock::new();
    INFO.get_or_init(|| BuildInfo {
        version: generated::VERSION.to_string(),
        commit: generated::COMMIT.to_string(),
        compiler: generated::COMPILER.to_string(),
        linear_algebra: crate::linalg::version(),
        build_type: generated::BUILD_TYPE.to_string(),
        avx2: cfg!(target_feature = "avx2"),
        api_version: API_VERSION,
    })
}
