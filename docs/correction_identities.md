# Correction identities

Lifecycle: active infrastructure contract.

Authority: qlab's machine-readable correction identity constants and the
versioned project mapping published by the workspace and active blueprints.

This document defines the legacy correction identities emitted by qlab's
historical Layer B/C simulation route. The short `C0`, `C1`, and `C2` values
remain compatibility aliases only; new artifacts must also carry the fields
`identity_schema_version`, `namespace`, `method_id`, `algorithm`, and
`legacy_code`.

| Legacy code | Canonical method ID | Algorithm |
|---|---|---|
| `C0` | `legacy.family_adjustment.raw@v1` | raw p-values, no family adjustment |
| `C1` | `legacy.family_adjustment.holm@v1` | Holm adjusted p-values |
| `C2` | `legacy.family_adjustment.stepdown_maxT@v1` | synchronized step-down maxT |

The only registered compatibility route is `legacy.layer_bc`. Readers must
declare this route explicitly to interpret a pre-schema Layer B/C artifact;
the canonical fields are added to an in-memory copy only.

The active L5.5 blueprint uses a separate namespace:

| Blueprint code | Canonical method ID | Algorithm |
|---|---|---|
| `C0` | `l55.multiplicity.BH@v1` | Benjamini-Hochberg |
| `C1` | `l55.multiplicity.TWO_STAGE_BKY@v1` | two-stage Benjamini-Krieger-Yekutieli |

An artifact must never rely on a bare short code after leaving its route
context. Historical files remain immutable; compatibility mappings are
additive and do not change their numerical results.
