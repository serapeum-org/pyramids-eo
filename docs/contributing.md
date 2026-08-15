# Contributing

When contributing to this repository, please first discuss the change you wish to make via issue,
email, or any other method with the owners of this repository before making a change.

Please note we have a [code of conduct](CODE_OF_CONDUCT.md), please follow it in all your interactions
with the project.

## Development setup

pyramids-eo is pure Python on top of [pyramids](https://github.com/serapeum-org/pyramids). Environments
are managed with [pixi](https://pixi.sh):

```console
pixi install -e dev          # create the dev environment
pixi run -e dev pytest       # run the test suite
pixi run -e dev mypy         # type-check
pre-commit install           # enable the git hooks
```

## Pull Request Process

1. Create a feature branch off `main` — never commit directly to `main`.
2. Write tests for your change and make sure the full suite passes locally.
3. Update `docs/change-log.md` and bump the version in `pyproject.toml` when appropriate. The versioning
   scheme we use is [SemVer](http://semver.org/) via [Conventional Commits](https://www.conventionalcommits.org).
4. Update the documentation for any user-facing change.
5. Open a pull request and fill in the template; a maintainer will review and merge.
