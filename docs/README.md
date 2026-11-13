# HydroBEM documentation

The documents are grouped by reader and topic; together they describe what HydroBEM accepts,
computes and returns, by which methods, how accurately, and how this is tested. The code cites
them by folder, file and section (for example `guide/meshes-and-bodies §1`,
`reference/inputs-bodies §1`, `theory §11.1`, `theory §8`, `test-plan B`; the form is in
[development/rules.md](development/rules.md)).

## Reading paths

*Table 1. Where to start, by reader.*

| Reader | Path |
|---|---|
| New user | [guide/getting-started.md](guide/getting-started.md) (installation and a first problem, step by step), then [working with the model](guide/working-with-the-model.md) and the topics of the [guide](#the-folders) as needed: [meshes and bodies](guide/meshes-and-bodies.md), [water, waves and options](guide/environment-and-options.md), [reading results](guide/results.md), [special surfaces](guide/special-surfaces.md), [tanks and generalized modes](guide/tanks-and-generalized-modes.md), [mean drift and second-order loads](guide/second-order-loads.md), [performance, threads and progress](guide/performance.md) |
| Wants the theory | [theory/README.md](theory/README.md), the index of the chapters: the boundary-value problem (1) and the quantities (2 to 4), the methods (5 to 8, 13, 14), the special surfaces and modes (9, 10), the mean drift (11) and the second-order loads (12) |
| Looks something up | The [reference](#the-folders): every input by object ([the model and its options](reference/inputs-model.md), [meshes](reference/inputs-meshes.md), [bodies](reference/inputs-bodies.md), [lids, dampers and field points](reference/inputs-model-objects.md), [first and second order](reference/inputs-orders.md), [the rules](reference/inputs-rules.md)), the [queries](reference/queries.md), the [results](reference/results.md), the [output files](reference/file-formats.md), the [Rust API](reference/rust-api.md) and the [C ABI](reference/c-api.md) |
| Checks the validation | [validation/test-plan.md](validation/test-plan.md) (what is tested and to what accuracy), the [validation report](validation/README.md) (one page per case), [validation/not_verified.md](validation/not_verified.md) |
| Contributor | [development/rules.md](development/rules.md), [development/design.md](development/design.md), [development/architecture.md](development/architecture.md), [CONTRIBUTING.md](../CONTRIBUTING.md) |

## The folders

*Table 2. The folders of the documentation.*

| Folder | Contents |
|---|---|
| [guide/](guide/getting-started.md) | The user guide, one page per topic, each with worked examples: installation, troubleshooting and a first problem ([getting-started.md](guide/getting-started.md)); the objects and items of the model, changing and checking it, model and results files ([working-with-the-model.md](guide/working-with-the-model.md)); meshes, parts of a mesh, checking the geometry, bodies and their mass properties, several bodies, trim, leaving an object out ([meshes-and-bodies.md](guide/meshes-and-bodies.md)); the water, walls and channels, frequencies and headings, what the first order computes, the two methods ([environment-and-options.md](guide/environment-and-options.md)); the results, their axes and the output files ([results.md](guide/results.md)); the interior surface, thin and porous plates, air cushions, free-surface plates, damping lids and field points ([special-surfaces.md](guide/special-surfaces.md)); internal tanks and generalized modes ([tanks-and-generalized-modes.md](guide/tanks-and-generalized-modes.md)); the mean drift and which method to trust, the control surface, the QTFs, Newman's approximation and the free surface of the second-order potential ([second-order-loads.md](guide/second-order-loads.md)); threads, memory, symmetry and the progress of a solve ([performance.md](guide/performance.md)) |
| [theory/](theory/README.md) | The theory in numbered chapters, each with the physics, the method and the accuracy of its topic: the boundary-value problems, conventions and the definition of every computed quantity; the discretization, the parameters of every numerical component, their derivation and accuracy; the second-order wave loads |
| [reference/](reference/inputs-model.md) | The model: editing it (either-or inputs, lists), the tree of the inputs, the environment, the waves, the options, the model file and suppressed objects ([inputs-model.md](reference/inputs-model.md)); meshes, the items mesh and ranges, the geometry file .hbg ([inputs-meshes.md](reference/inputs-meshes.md)); bodies, trim, couplings and the surfaces of a body ([inputs-bodies.md](reference/inputs-bodies.md)); lids, fixed dampers and field points ([inputs-model-objects.md](reference/inputs-model-objects.md)); first and second order and the free surface of the potential QTF ([inputs-orders.md](reference/inputs-orders.md)); the model rules and the supported combinations ([inputs-rules.md](reference/inputs-rules.md)); every query ([queries.md](reference/queries.md)); results and the results file ([results.md](reference/results.md)); normalization, the numeric files and the `.out` summary ([file-formats.md](reference/file-formats.md)); the public Rust API of the engine ([rust-api.md](reference/rust-api.md)); the C ABI of the engine, `hydrobem.dll`, `hydrobem.h` ([c-api.md](reference/c-api.md)) |
| [validation/](validation/README.md) | Analytic solutions, identities, published references, regression baselines ([test-plan.md](validation/test-plan.md)); the validation report, one page per case (`validation/cases/`): the meshes and the comparisons of the low and higher order against published and analytic solutions, convergence studies, the sensitivity of the fixed constants, how the comparisons are made (`method.md`) and what is not verified (`not_verified.md`); regenerated by `validation/make_report.py` |
| [development/](development/rules.md) | Sources, records, repository standard ([rules.md](development/rules.md)); components, layering, precision, threading, error model ([design.md](development/design.md)); module map, solve flow, tables, tests, conventions ([architecture.md](development/architecture.md)); the error maps of the Green-function tables and panel rules, written by their scripts ([error-maps/](development/error-maps/)) |
| [references.md](references.md) | Public literature used |

The changes of each version: [CHANGELOG.md](../CHANGELOG.md). How to contribute:
[CONTRIBUTING.md](../CONTRIBUTING.md).

The engine's Rust documentation: `cargo doc -p hydrobem_core --no-deps` (output in
`target/doc`).

## Source tags

Some statements carry a tag that shows where the knowledge behind them comes from (Table 3).

*Table 3. The source tags.*

| Tag | Meaning |
|---|---|
| **[P]** | Public literature (listed in references.md) |
| **[H]** | HydroBEM's own design (API, input model, layout of its own files, rules) |
| **[B]** | Conventions of the WAMIT file formats, kept for interoperability |
| **[C]** | A numerical choice of HydroBEM: the accuracy is fixed and the method chosen to meet it; method, parameters and accuracy in the chapters of the theory |
