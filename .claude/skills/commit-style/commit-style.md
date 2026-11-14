# Commit style

Follows [Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/).

## Format

```
<type>(<scope>): <description>
```

- One line. Imperative mood, lowercase, no trailing period, at most 72 characters.
- **No body and no footer by default.** Add a body only when the diff cannot show
  the why (a measured number, a constraint from outside the repo); keep it to a
  few lines, separated from the subject by a blank line.
- **No `Co-Authored-By`, `Assisted-by`, `Generated-by` or any other agent
  attribution trailer.** The human owns every line.
- A breaking change is marked with `!` after the scope (`feat(kvcache)!: ...`);
  a `BREAKING CHANGE:` footer is the one footer that is allowed, and only then.

## Types

`feat`, `fix`, `perf`, `refactor`, `test`, `docs`, `build`, `ci`, `chore`.
`build` covers the C++/Metal extension, pins and packaging; `chore` is for
everything that touches no source and no docs.

## Scopes

The subsystem directory under `python/ftmetal/`: `core`, `message`, `layers`,
`attention`, `kvcache`, `moe`, `models`, `kernel`, `engine`, `scheduler`,
`server`, plus `tests`, `bench`, `docs`. Omit the scope only for a change that
spans most of the repository.

## Examples

```
feat(kvcache): add the KVWrite primitive and rebind discipline
fix(scheduler): fence page frees until the reading step drains
perf(attention): port fdecode v4 to page size 32
docs: split the offload notes out of DESIGN.md
build(kernel): pin mlx 0.32.1 and nanobind for the _C extension
```

## Process

- Commit only when the user asks. One logical change per commit; do not bundle
  an unrelated lint fix or rename into a feature commit.
- Never `git push`, open or comment on PRs, or create issues on the user's behalf.
- Report test results as they are in the conversation, never in the commit message.

## Releases

A release is one commit that only changes `python/ftmetal/version.py`, subject
`build: release X.Y.Z`, tagged `vX.Y.Z` (the first release tags the commit that already says
0.0.1). The wheels that go to PyPI are built from a clean checkout of that tag with
`scripts/build-wheels.sh --release`, which refuses anything else; no sdist is published.
