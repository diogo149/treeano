"""
taken from https://github.com/diogo149/python-walk
"""

import pickle
import io
import base64


class CyclicWalkException(Exception):
    pass


class DoneWalkingException(Exception):

    """
    exception to signify that the returned data should no longer be walked
    should only be thrown in a prewalk function
    """

    def __init__(self, data):
        self.data = data


def _identity(e):
    return e


# ############################## walk w/ pickle ##############################


def walk(obj,
         prewalk_fn=_identity,
         postwalk_fn=_identity,
         protocol=pickle.HIGHEST_PROTOCOL):
    """
    walks an arbitrary* python object using pickle.Pickler with a prewalk
    and postwalk function

    * maybe not arbitrary - but probably anything that can be pickled (:
    """
    parent_ids = set()

    def perform_walk(obj, ignore_first_obj):
        """
        ignore_first_obj:
        whether or not to ignore walking the first object encountered
        this is set to ignore walking the first object, to allow recursing
        down objects
        """
        ignore = [ignore_first_obj]

        def persistent_id(obj):
            if ignore[0]:
                # don't walk this element
                ignore[0] = False
                return None
            else:
                if id(obj) in parent_ids:
                    raise CyclicWalkException(
                        "Cannot walk recursive structures")

                # add id to list of parent ids, to watch for cycles
                parent_ids.add(id(obj))

                try:
                    prewalked = prewalk_fn(obj)
                except DoneWalkingException as e:
                    postwalked = e.data
                else:
                    inner_walked = perform_walk(obj=prewalked,
                                                ignore_first_obj=True)
                    postwalked = postwalk_fn(inner_walked)

                # pop id off the set of ids
                parent_ids.remove(id(obj))

                # TODO does this really need to be converted to a string
                # seems like it does for python2?
                # base64 encoding to avoid unsafe string errors
                return base64.urlsafe_b64encode(
                    pickle.dumps(postwalked, protocol=protocol))

        def persistent_load(persid):
            return pickle.loads(base64.urlsafe_b64decode(persid))

        src = io.BytesIO()
        pickler = pickle.Pickler(src)
        pickler.persistent_id = persistent_id
        pickler.dump(obj)
        datastream = src.getvalue()
        dst = io.BytesIO(datastream)
        unpickler = pickle.Unpickler(dst)
        unpickler.persistent_load = persistent_load
        return unpickler.load()

    return perform_walk(obj=obj, ignore_first_obj=False)


# ############################ collection walking ############################


def collection_walk(obj,
                    prewalk_fn=_identity,
                    postwalk_fn=_identity):
    """
    like walk, but more efficient while only working on (predefined)
    collections

    NOTE: does not talk in the same order as walk
    """
    parent_ids = set()

    def perform_walk(obj):
        if id(obj) in parent_ids:
            raise CyclicWalkException(
                "Cannot walk recursive structures")

        # add id to list of parent ids, to watch for cycles
        parent_ids.add(id(obj))

        try:
            prewalked = prewalk_fn(obj)
        except DoneWalkingException as e:
            postwalked = e.data
        else:
            # TODO add more collections
            # eg. namedtuple, ordereddict, numpy array
            # TODO maybe use prewalked.__class__ to construct new instance of
            # same collection
            if isinstance(prewalked, list):
                inner_walked = [perform_walk(item) for item in prewalked]
            elif isinstance(prewalked, dict):
                inner_walked = {perform_walk(key): perform_walk(value)
                                for key, value in prewalked.items()}
            elif isinstance(prewalked, tuple):
                inner_walked = tuple([perform_walk(item)
                                      for item in prewalked])
            elif isinstance(prewalked, set):
                inner_walked = {perform_walk(item) for item in prewalked}
            else:
                inner_walked = prewalked

            postwalked = postwalk_fn(inner_walked)

        # pop id off the set of ids
        parent_ids.remove(id(obj))

        return postwalked

    return perform_walk(obj)


def collection_prewalk(obj, prewalk_fn):
    return collection_walk(obj, prewalk_fn=prewalk_fn)


def collection_postwalk(obj, postwalk_fn):
    return collection_walk(obj, postwalk_fn=postwalk_fn)
