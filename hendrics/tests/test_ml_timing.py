import pytest
import numpy as np
from hendrics.ml_timing import normalized_template, normalized_template_func
from hendrics.ml_timing import ml_pulsefit, minimum_phase_diff


class TestTiming(object):
    @classmethod
    def setup_class(cls):
        cls.phases = np.arange(0.5 / 64, 1, 1 / 64)
        phases_fine = np.arange(0.5 / 512, 1, 1 / 512)
        cls.real_base = 20
        cls.real_amp = 200
        cls.template = cls.real_base + cls.real_amp * np.exp(
            -((phases_fine - 0.5) ** 2) / (2 * 0.1**2)
        )
        cls.normt = normalized_template(cls.template, tomax=True)

    @pytest.mark.parametrize("case", [(True, None), (True, 0.4), (False, 0.4)])
    def test_fit_correct(self, case):
        fit_base, err = case
        phase = np.random.uniform(0, 1)
        temp_func = normalized_template_func(self.template, tomax=True)

        prof_smooth = self.real_base + self.real_amp * temp_func(self.phases - phase)

        if err is not None:
            profile = np.random.normal(prof_smooth, err)
        else:
            profile = np.random.poisson(prof_smooth)

        pars, errs = ml_pulsefit(
            profile,
            self.normt,
            profile_err=err,
            calculate_errors=True,
            fit_base=fit_base,
        )
        assert np.abs(minimum_phase_diff(pars[1], phase)) < 3 * errs[1]
