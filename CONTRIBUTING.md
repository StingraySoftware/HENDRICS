Contributions from everyone, experienced and inexperienced, are welcome!
If you don't know where to start, look at the open [Issues](https://github.com/StingraySoftware/HENDRICS/issues)
and/or get involved in our [Slack channel ](https://join.slack.com/t/stingraysoftware/shared_invite/zt-49kv4kba-mD1Y~s~rlrOOmvqM7mZugQ).

Please follow the [Astropy contribution guidelines](https://docs.astropy.org/en/stable/development/workflow/development_workflow.html),
and the [Astropy coding guidelines](https://docs.astropy.org/en/stable/development/codeguide.html#coding-style-conventions).
This code is written in Python 3.11+
Tests run at each commit during Pull Requests, so it is easy to single out points in the code that break this compatibility.

## Changelog entries

Every pull request should come with a short changelog fragment: a file in
`docs/changes/` named after the pull request number, e.g. `123.bugfix.rst`.
The fragments are collected into `CHANGELOG.rst` at release time by
[towncrier](https://towncrier.readthedocs.io/), so nothing needs to be edited
in `CHANGELOG.rst` by hand.

See [`docs/changes/README.rst`](docs/changes/README.rst) for the available
entry types, naming rules and writing guidelines.
