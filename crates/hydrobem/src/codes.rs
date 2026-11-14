//! The codes of the public API as enums: the result quantities and their element types, the solve
//! stages, the model states, the diagnostics, the panel roles, the quality flags, the field-point
//! statuses and the control surfaces. The enums with explicit discriminants have the integer values
//! that the results files store and the Python package reads.

use crate::quantities::{QuantityInfo, quantity_info};

/// The element type of a result quantity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ElementType {
    Float64 = 0,
    Float32 = 1,
    /// complex of two float32 (real, imaginary): NumPy's complex64, the Rust `Complex32`
    ComplexF32 = 2,
    Int32 = 3,
    /// complex of two float64 (real, imaginary): NumPy's complex128, the Rust `Complex64`
    ComplexF64 = 4,
    Int64 = 5,
    Text = 6,
}

impl ElementType {
    /// The bytes of one element (0 for text).
    pub fn size(self) -> usize {
        match self {
            ElementType::Float32 | ElementType::Int32 => 4,
            ElementType::Float64 | ElementType::Int64 | ElementType::ComplexF32 => 8,
            ElementType::ComplexF64 => 16,
            ElementType::Text => 0,
        }
    }
    pub fn is_complex(self) -> bool {
        matches!(self, ElementType::ComplexF32 | ElementType::ComplexF64)
    }
}

/// A result quantity: one per row of the quantity table, in its order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum QuantityId {
    Omegas = 0,
    Headings = 1,
    RadiationModeIds = 2,
    Coordinates = 3,
    AddedMass = 4,
    RadiationDamping = 5,
    AddedMassZeroFrequency = 6,
    AddedMassInfiniteFrequency = 7,
    ExcitationDiffraction = 8,
    ExcitationFroudeKrylov = 9,
    ExcitationScattering = 10,
    ExcitationHaskind = 11,
    Motion = 12,
    ModeRestoring = 13,
    PressureModeDamping = 14,
    BodyPoints = 15,
    BodyPointsIds = 16,
    BodyPointsNormals = 17,
    BodyPointsAreas = 18,
    BodyPointsPressureRadiation = 19,
    BodyPointsPressureDiffraction = 20,
    BodyPointsVelocityRadiation = 21,
    BodyPointsVelocityDiffraction = 22,
    FieldPoints = 23,
    FieldPointsPressureRadiation = 24,
    FieldPointsPressureIncident = 25,
    FieldPointsPressureScattering = 26,
    FieldPointsPressureDiffraction = 27,
    FieldPointsVelocityRadiation = 28,
    FieldPointsVelocityIncident = 29,
    FieldPointsVelocityScattering = 30,
    FieldPointsVelocityDiffraction = 31,
    Lids = 32,
    LidsIds = 33,
    LidsElevationRadiation = 34,
    LidsElevationDiffraction = 35,
    PressureSurfaces = 36,
    PressureSurfacesIds = 37,
    PressureSurfacesElevationRadiation = 38,
    PressureSurfacesElevationDiffraction = 39,
    Tanks = 40,
    TanksIds = 41,
    TanksPressureRadiation = 42,
    TanksVelocityRadiation = 43,
    KochinAngles = 44,
    KochinRadiation = 45,
    KochinRadiationDerivative = 46,
    KochinDiffraction = 47,
    KochinDiffractionDerivative = 48,
    DriftPressureIntegration = 49,
    DriftPressureIntegrationBodyAxes = 50,
    DriftControlSurface = 51,
    DriftControlSurfaceBodyAxes = 52,
    DriftMomentumConservation = 53,
    DriftHeadingPairs = 54,
    QtfQuadraticPressureIntegrationDifference = 55,
    QtfQuadraticPressureIntegrationSum = 56,
    QtfQuadraticControlSurfaceDifference = 57,
    QtfQuadraticControlSurfaceSum = 58,
    QtfPotentialDirectDifference = 59,
    QtfPotentialDirectSum = 60,
    QtfPotentialIndirectDifference = 61,
    QtfPotentialIndirectSum = 62,
    QtfNewmanPressureIntegrationDifference = 63,
    QtfNewmanControlSurfaceDifference = 64,
    QtfNewmanMomentumConservationDifference = 65,
    QtfHeadingPairs = 66,
    HydrostaticsVolumes = 67,
    HydrostaticsVolume = 68,
    HydrostaticsDisplacedMass = 69,
    HydrostaticsCenterOfBuoyancy = 70,
    HydrostaticsWaterplaneArea = 71,
    HydrostaticsWaterplaneCentroid = 72,
    HydrostaticsStiffnessRhoG = 73,
    HydrostaticsMass = 74,
    HydrostaticsCenterOfGravity = 75,
    HydrostaticsStiffness = 76,
    HydrostaticsMetacentricHeight = 77,
}

impl QuantityId {
    /// The number of quantities (the rows of the quantity table).
    pub const COUNT: usize = 78;
    /// Every quantity in the order of the table.
    #[rustfmt::skip]
    pub const ALL: [QuantityId; QuantityId::COUNT] = [
        QuantityId::Omegas, QuantityId::Headings, QuantityId::RadiationModeIds, QuantityId::Coordinates,
        QuantityId::AddedMass, QuantityId::RadiationDamping, QuantityId::AddedMassZeroFrequency,
        QuantityId::AddedMassInfiniteFrequency, QuantityId::ExcitationDiffraction,
        QuantityId::ExcitationFroudeKrylov, QuantityId::ExcitationScattering, QuantityId::ExcitationHaskind,
        QuantityId::Motion, QuantityId::ModeRestoring, QuantityId::PressureModeDamping, QuantityId::BodyPoints,
        QuantityId::BodyPointsIds, QuantityId::BodyPointsNormals, QuantityId::BodyPointsAreas,
        QuantityId::BodyPointsPressureRadiation, QuantityId::BodyPointsPressureDiffraction,
        QuantityId::BodyPointsVelocityRadiation, QuantityId::BodyPointsVelocityDiffraction, QuantityId::FieldPoints,
        QuantityId::FieldPointsPressureRadiation, QuantityId::FieldPointsPressureIncident,
        QuantityId::FieldPointsPressureScattering, QuantityId::FieldPointsPressureDiffraction,
        QuantityId::FieldPointsVelocityRadiation, QuantityId::FieldPointsVelocityIncident,
        QuantityId::FieldPointsVelocityScattering, QuantityId::FieldPointsVelocityDiffraction, QuantityId::Lids,
        QuantityId::LidsIds, QuantityId::LidsElevationRadiation, QuantityId::LidsElevationDiffraction,
        QuantityId::PressureSurfaces, QuantityId::PressureSurfacesIds,
        QuantityId::PressureSurfacesElevationRadiation, QuantityId::PressureSurfacesElevationDiffraction,
        QuantityId::Tanks, QuantityId::TanksIds, QuantityId::TanksPressureRadiation,
        QuantityId::TanksVelocityRadiation, QuantityId::KochinAngles, QuantityId::KochinRadiation,
        QuantityId::KochinRadiationDerivative, QuantityId::KochinDiffraction,
        QuantityId::KochinDiffractionDerivative, QuantityId::DriftPressureIntegration,
        QuantityId::DriftPressureIntegrationBodyAxes, QuantityId::DriftControlSurface,
        QuantityId::DriftControlSurfaceBodyAxes, QuantityId::DriftMomentumConservation,
        QuantityId::DriftHeadingPairs, QuantityId::QtfQuadraticPressureIntegrationDifference,
        QuantityId::QtfQuadraticPressureIntegrationSum, QuantityId::QtfQuadraticControlSurfaceDifference,
        QuantityId::QtfQuadraticControlSurfaceSum, QuantityId::QtfPotentialDirectDifference,
        QuantityId::QtfPotentialDirectSum, QuantityId::QtfPotentialIndirectDifference,
        QuantityId::QtfPotentialIndirectSum, QuantityId::QtfNewmanPressureIntegrationDifference,
        QuantityId::QtfNewmanControlSurfaceDifference, QuantityId::QtfNewmanMomentumConservationDifference,
        QuantityId::QtfHeadingPairs, QuantityId::HydrostaticsVolumes, QuantityId::HydrostaticsVolume,
        QuantityId::HydrostaticsDisplacedMass, QuantityId::HydrostaticsCenterOfBuoyancy,
        QuantityId::HydrostaticsWaterplaneArea, QuantityId::HydrostaticsWaterplaneCentroid,
        QuantityId::HydrostaticsStiffnessRhoG, QuantityId::HydrostaticsMass,
        QuantityId::HydrostaticsCenterOfGravity, QuantityId::HydrostaticsStiffness,
        QuantityId::HydrostaticsMetacentricHeight,
    ];

    /// The quantity of a name of the table, e.g. "added_mass".
    pub fn from_name(name: &str) -> Option<QuantityId> {
        QuantityId::ALL.iter().copied().find(|q| q.name() == name)
    }
    fn row(self) -> &'static QuantityInfo {
        quantity_info(self)
    }
    /// Its name, e.g. "added_mass".
    pub fn name(self) -> &'static str {
        self.row().name
    }
    pub fn element_type(self) -> ElementType {
        self.row().dtype
    }
    /// Its axes, e.g. "frequency,coordinate,radiation_column".
    pub fn dims(self) -> &'static str {
        self.row().dims
    }
    pub fn unit(self) -> &'static str {
        self.row().unit
    }
    pub fn description(self) -> &'static str {
        self.row().description
    }
    /// The path of its view in the result tree, e.g. "drift.pressure_integration" ("" when it has
    /// none: the ids of the coordinates, the lid points and the pressure-surface points).
    pub fn group(self) -> &'static str {
        self.row().group
    }
    /// The rule that selects the part of an owner (`Results::select`).
    pub fn owner(self) -> crate::result_tree::Owner {
        self.row().owner
    }
}

/// The stage of a progress report.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Stage {
    Statics,
    Assembly,
    Lu,
    Outputs,
    /// one more frequency (or limit) finished
    FrequencyDone,
    /// the parts of an item of a second-order step (the report's `at` names the step and the item)
    SecondOrder,
    /// the phase of the solve changed (always reported)
    Phase,
    /// a frequency worker has no item left in its phase (the report's worker)
    WorkerIdle,
}

impl Stage {
    /// Its name, e.g. "assembly".
    pub fn name(self) -> &'static str {
        match self {
            Stage::Statics => "statics",
            Stage::Assembly => "assembly",
            Stage::Lu => "lu",
            Stage::Outputs => "outputs",
            Stage::FrequencyDone => "frequency_done",
            Stage::SecondOrder => "second_order",
            Stage::Phase => "phase",
            Stage::WorkerIdle => "worker_idle",
        }
    }
}

/// The state of a model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ModelState {
    /// no results of its present inputs
    Reset,
    /// a solve runs
    Calculating,
    /// it holds the results of its present inputs
    Calculated,
}

/// The severity of a diagnostic; its code is stored in the results files.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Severity {
    Warning = 1,
    Error = 2,
}

impl Severity {
    /// The code of a results file.
    pub(crate) fn code(self) -> i32 {
        self as i32
    }
    /// The severity of a code of a results file (an unknown code is a warning).
    pub(crate) fn from_code(code: i32) -> Severity {
        if code == Severity::Error as i32 { Severity::Error } else { Severity::Warning }
    }
}

/// The warnings of the model rules and of a solve; the code of each is the `code` of its
/// diagnostics.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Warning {
    PressureDriftPressureSurfaces = 1,
    CoarseMesh = 2,
    MeshOrientation = 3,
    MeshOpen = 4,
    MeshVolume = 5,
    FieldPointNearBody = 6,
    TankZeroFrequency = 7,
    /// a disabled object would fail the rules were it enabled (`Model::check`)
    DisabledObject = 8,
    /// a hook of the environment that changes the numbers is set (HB_DIAG_DOUBLE,
    /// HB_VALIDATE_<name>, HB_WATERLINE_CENTROID; internal studies only)
    EnvironmentHook = 9,
    /// wetted elements of one body or of two lie on each other with opposite normals
    MeshFacing = 10,
}

impl Warning {
    const ALL: [Warning; 10] = [
        Warning::PressureDriftPressureSurfaces,
        Warning::CoarseMesh,
        Warning::MeshOrientation,
        Warning::MeshOpen,
        Warning::MeshVolume,
        Warning::FieldPointNearBody,
        Warning::TankZeroFrequency,
        Warning::DisabledObject,
        Warning::EnvironmentHook,
        Warning::MeshFacing,
    ];
    /// The code of the warning's diagnostics.
    pub fn code(self) -> i32 {
        self as i32
    }
    /// The warning of a diagnostic's code.
    pub fn from_code(code: i32) -> Option<Warning> {
        Warning::ALL.into_iter().find(|w| w.code() == code)
    }
}

/// The role of an element as the engine uses it (`Panels::role`, `Tessellation::role`): the
/// singular of the input that makes the surface.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PanelRole {
    /// the wetted surface of a body (its mesh, `ranges`)
    Wetted = 0,
    /// the interior free surface of a body (detected, generated or a mesh)
    InteriorSurface = 1,
    /// a dipole (`Body.dipoles`, `Tank.dipoles`)
    Dipole = 2,
    /// a dipole damper (`Body.dipole_dampers`, `Tank.dipole_dampers`)
    DipoleDamper = 3,
    /// a pressure surface (`Body.pressure_surfaces`)
    PressureSurface = 4,
    /// a damping lid (`m.lids`)
    Lid = 5,
    /// the walls of a tank (`Body.tanks`)
    Tank = 6,
    /// a fixed damper (`m.fixed_dampers`)
    FixedDamper = 7,
    /// the control surface of a body (no boundary of the problem; on request)
    ControlSurface = 8,
    /// the panelled zone of the potential QTF's free surface (no boundary of the problem; on request)
    FreeSurface = 9,
    /// a free-surface plate of a body (`Body.free_surface_plates`)
    FreeSurfacePlate = 10,
}

impl PanelRole {
    pub(crate) const ALL: [PanelRole; 11] = [
        PanelRole::Wetted,
        PanelRole::InteriorSurface,
        PanelRole::Dipole,
        PanelRole::DipoleDamper,
        PanelRole::PressureSurface,
        PanelRole::Lid,
        PanelRole::Tank,
        PanelRole::FixedDamper,
        PanelRole::ControlSurface,
        PanelRole::FreeSurface,
        PanelRole::FreeSurfacePlate,
    ];
    /// The role of its value.
    pub fn from_value(v: i32) -> Option<PanelRole> {
        PanelRole::ALL.get(usize::try_from(v).ok()?).copied()
    }
    /// The role of its name (`name`).
    pub fn from_name(s: &str) -> Option<PanelRole> {
        PanelRole::ALL.iter().copied().find(|r| r.name() == s)
    }
    /// Its value (the code of the role).
    pub fn value(self) -> i32 {
        self as i32
    }
    /// Its name, e.g. "fixed_damper".
    pub fn name(self) -> &'static str {
        match self {
            PanelRole::Wetted => "wetted",
            PanelRole::InteriorSurface => "interior_surface",
            PanelRole::Dipole => "dipole",
            PanelRole::DipoleDamper => "dipole_damper",
            PanelRole::PressureSurface => "pressure_surface",
            PanelRole::Lid => "lid",
            PanelRole::Tank => "tank",
            PanelRole::FixedDamper => "fixed_damper",
            PanelRole::ControlSurface => "control_surface",
            PanelRole::FreeSurface => "free_surface",
            PanelRole::FreeSurfacePlate => "free_surface_plate",
        }
    }
}

/// The findings of the mesh quality of an element (`MeshQuality::flags`): a set of bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct QualityFlags(pub u32);

impl QualityFlags {
    pub const ORIENTATION: QualityFlags = QualityFlags(1);
    pub const FREE_EDGE: QualityFlags = QualityFlags(2);
    pub const NON_MANIFOLD: QualityFlags = QualityFlags(4);
    pub const COARSE: QualityFlags = QualityFlags(8);
    pub fn contains(self, f: QualityFlags) -> bool {
        self.0 & f.0 == f.0
    }
    /// Whether it has one of the flags of f.
    pub fn intersects(self, f: QualityFlags) -> bool {
        self.0 & f.0 != 0
    }
}

impl std::ops::BitOr for QualityFlags {
    type Output = QualityFlags;
    fn bitor(self, o: QualityFlags) -> QualityFlags {
        QualityFlags(self.0 | o.0)
    }
}

impl std::ops::BitOrAssign for QualityFlags {
    fn bitor_assign(&mut self, o: QualityFlags) {
        self.0 |= o.0;
    }
}

/// The status of a point under the field-point rules (reference/inputs-model-objects §3; `Model::classify_points`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PointStatus {
    Fluid = 0,
    Tank = 1,
    NoTank = 2,
    EmptyTank = 3,
    AboveTankSurface = 4,
    AboveFreeSurface = 5,
    BelowSeaBed = 6,
    OutsideDomain = 7,
    OnBody = 8,
    InsideWaterplane = 9,
    OnLid = 10,
    NotFinite = 11,
    InsideBody = 12,
    NearBody = 13,
}

impl PointStatus {
    /// Its name, e.g. "above_free_surface".
    pub fn name(self) -> &'static str {
        match self {
            PointStatus::Fluid => "fluid",
            PointStatus::Tank => "tank",
            PointStatus::NoTank => "no_tank",
            PointStatus::EmptyTank => "empty_tank",
            PointStatus::AboveTankSurface => "above_tank_surface",
            PointStatus::AboveFreeSurface => "above_free_surface",
            PointStatus::BelowSeaBed => "below_sea_bed",
            PointStatus::OutsideDomain => "outside_domain",
            PointStatus::OnBody => "on_body",
            PointStatus::InsideWaterplane => "inside_waterplane",
            PointStatus::OnLid => "on_lid",
            PointStatus::NotFinite => "not_finite",
            PointStatus::InsideBody => "inside_body",
            PointStatus::NearBody => "near_body",
        }
    }
}

/// The kind of a control surface made by the engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ControlSurfaceKind {
    /// around a body: a circle of a radius about its origin, or a polygon
    Automatic,
    /// a truncated circular cylinder about the body's vertical axis, with its free-surface part
    Cylinder,
    /// the lower half of an ellipsoid about the body's origin, with its free-surface part
    Ellipsoid,
}

impl ControlSurfaceKind {
    /// The geometry code of the analytic control surface in a control-surface file: -1001 the
    /// cylinder, -1003 the ellipsoid; 0 for the automatic surface.
    pub(crate) fn icdef(self) -> i32 {
        match self {
            ControlSurfaceKind::Automatic => 0,
            ControlSurfaceKind::Cylinder => -1001,
            ControlSurfaceKind::Ellipsoid => -1003,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_codes_are_the_rows_of_the_tables() {
        assert_eq!(QuantityId::ALL.len(), QuantityId::COUNT);
        for (k, q) in QuantityId::ALL.iter().enumerate() {
            assert_eq!(*q as usize, k);
            assert_eq!(QuantityId::from_name(q.name()), Some(*q));
        }
        assert_eq!(QuantityId::AddedMass as i32, 4);
        assert_eq!(QuantityId::DriftControlSurfaceBodyAxes as i32, 52);
        assert_eq!(QuantityId::QtfHeadingPairs as i32, 66);
        for w in Warning::ALL {
            assert_eq!(Warning::from_code(w.code()), Some(w));
        }
        for (k, r) in PanelRole::ALL.iter().enumerate() {
            assert_eq!(PanelRole::from_value(k as i32), Some(*r));
        }
    }
}
