//! The generated tables of the Green functions (scripts/gen_green_tables.py,
//! scripts/gen_green_finite_tables.py) and of the log-Gauss rule (scripts/gen_log_gauss.py); never
//! edited by hand. The checksums of the Green tables (FNV-1a) are verified at run time
//! (`green_inf::verify_green_tables`, `green_fin::verify_fin_tables`).

#[rustfmt::skip]
pub mod green_fin_tables;
#[rustfmt::skip]
pub mod green_inf_tables;
#[rustfmt::skip]
pub mod log_gauss_table;
