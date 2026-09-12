Changelog fragments
===================

``CHANGELOG.rst`` is assembled by `towncrier <https://towncrier.readthedocs.io>`__
from the files in this directory, so **the changelog itself is never edited by
hand**. Every pull request adds a small file here instead. That way two PRs
touching the changelog never conflict with each other.

Naming
------

One file per entry, named::

    <pull request number>.<type>.rst

for example ``201.bugfix.rst``. If one pull request needs more than one entry of
the same type, number them::

    201.bugfix.1.rst
    201.bugfix.2.rst

The ``check-changelog`` GitHub workflow fails a pull request that adds no file
matching ``docs/changes/<PR number>.*.rst``. If the change genuinely needs no
entry — a typo fix in a comment, say — put the ``no-changelog-entry-needed``
label on the pull request instead.

Types
-----

The types are configured under ``[tool.towncrier]`` in ``pyproject.toml``:

``breaking``
    Changes that can break existing scripts: a command or option removed or
    renamed, a default changed, a supported Python version dropped.
``deprecation``
    Something still works but is on its way out.
``removal``
    Something previously deprecated is now gone.
``feature``
    New commands, new options, or a substantial speed-up.
``bugfix``
    Something did not work and now does.
``doc``
    Documentation only.
``trivial``
    Internal changes with no effect on users: refactoring, packaging, CI, tests.

Writing the entry
-----------------

The file holds reStructuredText, and towncrier renders it as a single bullet
with a link to the pull request appended, so:

* write one or two full sentences, no leading bullet marker and no trailing
  pull-request link — towncrier adds that;
* name the command or the function, so a reader can tell whether it affects
  them: "``HENcalibrate`` now says ..." beats "fixed a bug in calibration";
* say what changed for the user, not how it was implemented.

Building the changelog
----------------------

Only done at release time, and only by a maintainer::

    towncrier build --version <the new version>

That consumes the files in this directory, rewrites ``CHANGELOG.rst`` and
deletes the fragments. To see what it would produce without touching anything::

    towncrier build --draft --version <the new version>
