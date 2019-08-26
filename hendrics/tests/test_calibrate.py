import os
import numpy as np
from hendrics.calibrate import default_nustar_rmf
import logging


def test_default_nustar_rmf(caplog):
    caldb_path = "fake_caldb"
    os.environ['CALDB'] = caldb_path
    path_to_rmf = os.path.join(
        caldb_path,
        *"data/nustar/fpm/cpf/rmf/nuAdet3_20100101v002.rmf".split('/'))
    newpath = default_nustar_rmf()

    assert np.any(["Rmf not specified. Using default NuSTAR rmf." in r.msg
            for r in caplog.records])
    assert newpath == path_to_rmf
