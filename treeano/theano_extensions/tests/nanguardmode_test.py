import nose.tools as nt
import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams

import theano.compile.nanguardmode
import treeano.theano_extensions.nanguardmode


if "gpu" in theano.config.device:
    def test_nanguardmode():
        # this is the case which requires a custom nanguardmode
        srng = MRG_RandomStreams()
        x = srng.uniform((3, 4, 5))

        def random_number(mode):
            return theano.function([], [x], mode=mode)()

        @nt.raises(AssertionError)
        def fails():
            random_number(theano.compile.nanguardmode.NanGuardMode(
                nan_is_error=True,
                inf_is_error=True,
                big_is_error=True
            ))

        fails()

        random_number(treeano.theano_extensions.nanguardmode.NanGuardMode(
            nan_is_error=True,
            inf_is_error=True,
            big_is_error=True
        ))
