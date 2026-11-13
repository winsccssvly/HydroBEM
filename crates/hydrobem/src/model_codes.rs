//! The choices of the engine's problem (internal): the choices of the model's items as the
//! solvers read them (problem.rs). The model fills them from its objects (document_build.rs).

use crate::codes::QuantityId;

/// Unknown of the diffraction problem.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Formulation {
    /// scattered potential
    #[default]
    Scattered,
    /// total potential; required with lids
    Total,
}

/// Fluid domain (non-exhaustive: a later domain is not a breaking change).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum Domain {
    /// no walls
    #[default]
    OpenSea,
    /// wall at x = 0
    XWall,
    /// wall at y = 0
    YWall,
    /// walls at x = 0 and y = 0
    OrthogonalWalls,
    /// walls at y = +-W/2 (finite depth)
    Channel,
}

/// Point about which inertia or additional matrices are given.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Origin {
    /// the centre of gravity G
    #[default]
    CenterOfGravity,
    /// the rotation centre R
    RotationCenter,
    /// a body-local point (the *_point field)
    Point,
}

/// Exciting force of a body's motion equation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Excitation {
    /// pressure integration of the diffraction potential
    #[default]
    Pressure,
    /// Haskind relation
    Haskind,
}

/// Interior free surface for irregular-frequency removal (reference/inputs-bodies §2.6).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum InteriorMethod {
    /// none
    #[default]
    None,
    /// elements of the body mesh
    PanelRange,
    /// a separate body-local mesh
    ExternalMesh,
    /// generated: vertical projection of the wetted surface
    Projected,
    /// generated: triangulation inside the waterline
    Waterline,
    /// generated: fan from the region centroid, graded towards the waterline
    Graded,
}

/// Shape n(x, y) of a prescribed-pressure mode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PressureShape {
    /// 1
    #[default]
    Uniform,
    /// sign of the local x
    SignX,
    /// sign of the local y
    SignY,
}

/// Restoring convention of the pressure modes (theory §9.3).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RestoringConvention {
    /// variational form
    #[default]
    Variational,
    /// the overlap stiffness with the opposite sign, the roof coupling in the rigid-mode rows only
    NumericCompatibility,
}

/// Condition of a dipole damper.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DamperCondition {
    /// a fixed damper: at rest, impervious; in no force
    #[default]
    FreeImpervious,
    /// a damper of the body: moving with it, porous; in the body forces
    FixedPorous,
}

/// Heading pairs of a QTF (theory §12.2).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum QtfHeadings {
    /// not computed
    #[default]
    None,
    /// the equal headings
    Equal,
    /// every heading pair (bidirectional)
    Pairs,
}

impl QtfHeadings {
    /// Whether the QTF is computed.
    pub fn any(self) -> bool {
        self != QtfHeadings::None
    }
}

/// Frequencies of the QTFs (theory §12.2).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum QtfFrequencies {
    /// the difference and the sum frequencies
    #[default]
    Both,
    /// the difference frequencies only
    Difference,
    /// the sum frequencies only
    Sum,
}

/// Where the control-surface drift takes the zeta^2 term of the waterline (theory §11.3; the item
/// control_surface_waterline_term).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WaterlineTerm {
    /// -rho g / 2 int_CL n' zeta^2 on the waterline of the control surface
    #[default]
    ControlSurface,
    /// +rho g / 2 int_WL n' zeta^2 on the body waterline minus rho g int_Sf zeta grad' zeta
    Body,
}

/// The drifts, by their methods, that Newman's approximation (theory §12.13) is made from.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NewmanDrifts {
    /// the mean drift by pressure integration
    pub pressure_integration: bool,
    /// the mean drift by the control surface
    pub control_surface: bool,
    /// the mean drift by momentum conservation
    pub momentum_conservation: bool,
}

impl NewmanDrifts {
    /// The number of the drifts.
    pub fn count(self) -> usize {
        [self.pressure_integration, self.control_surface, self.momentum_conservation].iter().filter(|&&on| on).count()
    }
}

/// The mass of a body when it has one (reference/inputs-bodies: mass).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Mass {
    /// the given mass (kg)
    Given(f64),
    /// displaced: the density times the displaced volume, less the fluid of its tanks
    Displaced,
}

/// A set of result quantities: the quantities a solve computes for others but the model did not
/// ask for (Options::hidden), dropped from its results at the end of the solve.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct QuantitySet(u128);

const _: () = assert!(QuantityId::COUNT <= 128, "QuantitySet holds 128 quantities");

impl QuantitySet {
    /// Adds the quantities.
    pub fn extend(&mut self, quantities: impl IntoIterator<Item = QuantityId>) {
        for q in quantities {
            self.0 |= 1u128 << q as u32;
        }
    }
    /// Whether q is in the set.
    pub fn contains(self, q: QuantityId) -> bool {
        self.0 & (1u128 << q as u32) != 0
    }
    /// The quantities of the set, in the order of the quantity table.
    pub fn iter(self) -> impl Iterator<Item = QuantityId> {
        QuantityId::ALL.into_iter().filter(move |&q| self.contains(q))
    }
}

/// The QTFs at the difference frequencies, by part and method.
pub const QTF_DIFFERENCES: [QuantityId; 4] = [
    QuantityId::QtfQuadraticPressureIntegrationDifference,
    QuantityId::QtfQuadraticControlSurfaceDifference,
    QuantityId::QtfPotentialDirectDifference,
    QuantityId::QtfPotentialIndirectDifference,
];

/// The QTFs at the sum frequencies, by part and method.
pub const QTF_SUMS: [QuantityId; 4] = [
    QuantityId::QtfQuadraticPressureIntegrationSum,
    QuantityId::QtfQuadraticControlSurfaceSum,
    QuantityId::QtfPotentialDirectSum,
    QuantityId::QtfPotentialIndirectSum,
];
