//! The C ABI of the engine: `hydrobem.dll` (with its import library) and `include/hydrobem.h`,
//! written by `scripts/gen_c_header.py` from these sources. It binds the public Rust API of
//! `hydrobem_core` as the Python extension module does, for any program that calls C functions.
//!
//! Conventions (docs/reference/c-api.md):
//! - Every function returns an `HbStatus` (`HB_OK` or an `HB_ERR_` code; a few return a value
//!   that cannot fail); the message and location of the last failure of a thread are read with
//!   `hb_last_error_message` and `hb_last_error_location`. A panic never crosses the boundary.
//! - Texts are NUL-terminated UTF-8, paths too. Texts the library hands out are freed with
//!   `hb_string_free`, tables with `hb_table_free`, models with `hb_model_free`, results with
//!   `hb_results_free`.
//! - Objects of a model are named by `uint64_t` ids; answers are tables of named columns
//!   (`HbTable`, `HbColumn`); result quantities are views of the engine's elements.
//! - A model may be used from several threads (each call locks it); a solve releases the lock,
//!   so the model can be read and the solve cancelled meanwhile.

pub mod info;
pub mod mesh;
pub mod model;
pub mod object;
pub mod query;
pub mod results;
pub mod status;
pub mod table;

pub use info::*;
pub use mesh::*;
pub use model::*;
pub use object::*;
pub use query::*;
pub use results::*;
pub use status::{HB_ERR_ARGUMENT, HB_ERR_CANCELLED, HB_ERR_EXISTS, HB_ERR_INTERNAL, HB_ERR_IO, HB_ERR_NUMERICAL};
pub use status::{HB_ERR_STATE, HB_ERR_TYPE, HB_ERR_UNSUPPORTED, HB_ERR_VALUE, HB_OK, HbLocation, HbStatus};
pub use table::*;
