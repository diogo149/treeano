import theano
import theano.tensor as T
import treeano


@treeano.register_node("monitor_update_ratio")
class MonitorUpdateRatioNode(treeano.Wrapper1NodeImpl):

    """
    monitor's the ratio between the a statistic (eg. norm, max, min) between an
    update of a parameter and the parameter itself

    monitor's parameters of this nodes children, based on the updates already
    defined when traversing the architectural tree (most probably this node's
    parents)

    see:
    http://yyue.blogspot.in/2015/01/a-brief-overview-of-deep-learning.html
    http://cs231n.github.io/neural-networks-3/#ratio

    both links recommend a value of approximately 1e-3
    """

    hyperparameter_names = ("statistics",)

    @staticmethod
    def statistic_to_fn(statistic):
        return {
            "2-norm": lambda x: x.norm(2),
            "max": T.max,
            "min": T.min,
        }[statistic]

    def mutate_update_deltas(self, network, update_deltas):
        if not network.find_hyperparameter(["monitor"]):
            return
        if not network.find_hyperparameter(["monitor_updates"], True):
            # don't do anything if manually asking to ignore
            # ---
            # rationale: want the ability for a validation network to turn
            # this off
            return
        # these need to be strings so that we can print their names
        # ---
        # by default, only show 2-norm
        # because max and min don't always have the same sign and are harder
        # to compare
        statistics = network.find_hyperparameter(["statistics"],
                                                 ["2-norm"])
        # TODO parameterize search tags (to affect not only "parameters"s)
        vws = network.find_vws_in_subtree(tags={"parameter"},
                                          is_shared=True)
        for vw in vws:
            if vw.variable not in update_deltas:
                continue
            delta = update_deltas[vw.variable]
            for stat in statistics:
                assert isinstance(stat, str)
                name = "%s_%s" % (vw.name, stat)
                stat_fn = self.statistic_to_fn(stat)
                # computing the value of the stat after the update instead of
                # before
                # ---
                # rationale: avoiding 0-division errors for 0-initialized
                # shared variables
                shared_stat = stat_fn(vw.variable + delta)
                delta_stat = stat_fn(delta)
                ratio = delta_stat / shared_stat
                network.create_vw(
                    name,
                    variable=ratio,
                    shape=(),
                    tags={"monitor"}
                )
