# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Tests for the pulse profile template helpers."""

import os

import numpy as np
import pytest

from hendrics.fold import (
    create_template_from_profile,
    create_template_from_profile_harm,
    create_template_from_profile_sins,
)


@pytest.mark.parametrize(
    "func",
    [
        create_template_from_profile,
        create_template_from_profile_harm,
        create_template_from_profile_sins,
    ],
)
def test_create_template_from_profile_saves_image(func, tmp_path):
    """``imagefile`` writes a diagnostic plot; it defaults to writing nothing.

    The default used to be ``template.png`` in the current directory, which
    scattered files around whenever a template was built.
    """
    phase = np.arange(0.005, 1, 0.01)
    profile = np.cos(2 * np.pi * phase) + 2
    profile_err = profile * 0

    imagefile = str(tmp_path / "template.png")
    template, additional_phase = func(phase, profile, profile_err, imagefile=imagefile)

    assert os.path.exists(imagefile)
    assert template.size > 0
    assert 0 <= additional_phase <= 1

    # ...and no file is written when ``imagefile`` is left alone
    template_noplot, _ = func(phase, profile, profile_err)
    assert not os.path.exists("template.png")
    assert np.allclose(template_noplot, template)
