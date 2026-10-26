# Researchflow commands

Run from the installed project using `uv run researchflow COMMAND ...`.
Every command accepts `--root PATH` (also accepted before the command), defaulting
to the current directory. Relative input and output arguments resolve against
that root. Absolute paths remain absolute. Quote paths containing spaces.
The CLI delegates validation, evidence selection, and execution to public services.

| Command | Required / optional arguments | Result |
| --- | --- | --- |
| `doctor` | none | Toolchain JSON report |
| `smoke` | optional `--no-tex` | Integrated smoke JSON report |
| `experiment` | `--config PATH`; optional `--run-id ID`, `--freeze PATH` | Experiment bundle directory |
| `analyze` | `--config PATH` | Analysis bundle directory |
| `view` | `--config PATH --output PATH` | Directory containing `view.json` and `view.md` |
| `review` | `--config PATH --output PATH` | Context packet directory |
| `artifacts` | `--publication PATH` | Assembled publication snapshot directory |
| `freeze-study` | repeatable `--config PATH`, `--output PATH` | Study freeze JSON document |
| `paper` | `--publication PATH` | Compiled bundle directory |
| `paper-check` | `--compiled PATH` | Inspection JSON report |
| `release` | `--snapshot PATH --compiled PATH --output PATH --visual-review PATH` | Release bundle directory |

Path results print as one path; mapping results print as JSON. Failures print
command/root context, the error, and any service-provided details to stderr.
Exit codes: 0 success/help; 2 argument syntax, CONTRACT or IDENTITY; 3 EXECUTION;
4 DEPENDENCY (including unavailable imports); 1 unexpected failure/unknown code.
The Python `main(argv)` returns the code, including for help and parser errors;
the console entry point and `python -m researchflow.cli` use it as process status.
Services must raise WorkflowError for failed operations; the CLI does not infer
scientific acceptance from fields inside a returned diagnostic report.

`review` builds a context packet via `context.build_packet`.
`artifacts` assembles a pinned publication snapshot without compiling.
`paper` assembles that snapshot and compiles it into `ROOT/build/compiled` using
the compiler's default timeout (120 seconds). It launches no experiments. Both publication commands use the supplied explicit configuration.
`paper-check` inspects the explicitly selected compiled directory.

```sh
uv run researchflow experiment --root '/work/my project' --config 'configs/pilot.json' --run-id R-pilot
uv run researchflow freeze-study --root '/work/my project' --config configs/a.json --config configs/b.json --output freezes/F-final.json
uv run researchflow smoke --root '/work/my project' --no-tex
uv run researchflow paper --root '/work/my project' --publication configs/publication.json
```

# Make facade

Plain `make` displays usage. Targets have the same names as CLI commands.
`PROJECT_ROOT` defaults to Make's current directory; `UV=uv` selects the executable
(path allowed, shell fragments are not). Make must run in the installed source
project; `PROJECT_ROOT` selects workflow data and does not change uv's project.

| Target | Required variables | Optional variables |
| --- | --- | --- |
| `doctor`, `smoke` | none | none |
| `experiment` | `CONFIG` | `RUN_ID`, `FREEZE` |
| `analyze` | `CONFIG` | none |
| `view`, `review` | `CONFIG`, `OUTPUT` | none |
| `artifacts`, `paper` | `PUBLICATION` | none |
| `freeze-study` | `CONFIG`, `OUTPUT` | none |
| `paper-check` | `COMPILED` | none |
| `release` | `SNAPSHOT`, `COMPILED`, `OUTPUT`, `VISUAL_REVIEW` | none |

Required variables are checked before invoking the CLI. Use the CLI for multiple
freeze configurations or `smoke --no-tex`; Make smoke includes compilation.
Shell arguments are quoted, including embedded apostrophes. Literal dollar signs
in Make assignments follow Make's own escaping rules (`$$`); supply ordinary
filesystem paths, not shell expressions. No target has experiment prerequisites.
Make reports failures with its own nonzero status (normally 2), not necessarily
the exact CLI exit code.

```sh
make paper PROJECT_ROOT='/work/my project' PUBLICATION='configs/publication draft.json'
make experiment CONFIG='configs/pilot study.json' RUN_ID=R-pilot FREEZE='freezes/F-final.json'
make -n release SNAPSHOT='build/paper/P-pinned' COMPILED='build/compiled/pinned' OUTPUT='releases/R1' VISUAL_REVIEW='reviews/accepted.json'
```

After dependencies merge, run no-TeX smoke, then actual Mac BasicTeX smoke:

```sh
uv run researchflow smoke --no-tex
uv run researchflow doctor
uv run researchflow smoke
```

These are acceptance commands, not a claim that isolated CLI tests exercise the
whole infrastructure. Real PDF visual acceptance must be supplied separately;
fixture review records are synthetic and must not be used as acceptance.
