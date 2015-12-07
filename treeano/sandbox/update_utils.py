import theano
import theano.tensor as T

fX = theano.config.floatX


def update_counter(network,
                   update_deltas,
                   name,
                   shape=(),
                   default_inits=None):
    """
    stores and updates a counter
    """
    new_name = "counter(%s)" % name
    if default_inits is None:
        default_inits = []
    old = network.create_vw(
        new_name,
        shape=shape,
        is_shared=True,
        tags={"state"},
        default_inits=[],
    ).variable
    new = old + 1
    update_deltas[old] = new
    return old, new


def exponential_moving_average(network,
                               update_deltas,
                               var,
                               name,
                               shape,
                               smoothing,
                               unbias=False,
                               counter=None,
                               default_inits=None):
    """
    stores and updates an exponential moving average
    (which is optionally unbiased)
    """
    new_name = "exponential_moving_average(%s)" % name
    if default_inits is None:
        default_inits = []
    old = network.create_vw(
        new_name,
        shape=shape,
        is_shared=True,
        tags={"state"},
        default_inits=[],
    ).variable
    new = smoothing * old + (1 - smoothing) * var
    update_deltas[old] = new - old
    if unbias:
        if counter is None:
            _, counter = update_counter(
                network,
                update_deltas,
                new_name)
        new = new / (1 - smoothing ** counter)
    return old, new


def root_mean_square_exponential_moving_average(network,
                                                update_deltas,
                                                var,
                                                name,
                                                shape,
                                                smoothing,
                                                unbias=False,
                                                counter=None,
                                                default_inits=None):
    old_ema, new_ema = exponential_moving_average(network,
                                                  update_deltas,
                                                  T.sqr(var),
                                                  "squared(%s)" % name,
                                                  shape,
                                                  smoothing,
                                                  unbias,
                                                  counter,
                                                  default_inits)
    return T.sqrt(old_ema), T.sqrt(new_ema)


def update_previous(network,
                    update_deltas,
                    var,
                    name,
                    shape,
                    default_inits=None):
    """
    stores and updates the previous value of a var
    """
    new_name = "previous(%s)" % name
    if default_inits is None:
        default_inits = []
    old = network.create_vw(
        new_name,
        shape=shape,
        is_shared=True,
        tags={"state"},
        default_inits=[],
    ).variable
    new = var
    update_deltas[old] = new - old
    return old, new
