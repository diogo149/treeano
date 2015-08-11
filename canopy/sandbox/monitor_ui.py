"""
UI for monitoring data

features
- easy setup (simply run `python -m SimpleHTTPServer` in the directory)
- live monitoring (before all the data is done)
- customizable with data
  - eg. linear/log scale, rolling mean window
- multiple charts
- tooltips of the highlighted data point
- click the legend to hide lines
- draggable legend
"""

import re
import os
import json

import numpy as np

from .. import templates


class ResultWriter(object):

    """
    use cases:
    - live monitoring w/ a different process
    """

    def __init__(self,
                 dirname,
                 pattern,
                 remove_matched=False,
                 symlink=False):
        """
        remove_matched:
        whether or not to remove the matched results from the output map

        symlink:
        whether or not to make a symlink instead of copying the html/js files
        """
        self.dirname = dirname
        self.pattern = pattern
        self.remove_matched = remove_matched
        if symlink:
            # create directory
            os.mkdir(dirname)
            # symlink javascript and html
            for f in ["index.html", "monitor.js"]:
                os.symlink(templates.template_path("monitor_ui", f),
                           os.path.join(dirname, f))
            # create monitor.json file
            with open(os.path.join(dirname, "monitor.json"), "w") as f:
                pass
        else:
            templates.copy_template("monitor_ui", dirname)
        self._json_path = os.path.join(self.dirname, "monitor.json")
        self._regex = re.compile(self.pattern)

    def write(self, res):
        # prepare data
        monitor_keys = []
        monitor_data = {}
        for key in res:
            if self._regex.match(key):
                monitor_keys.append(key)
                val = res[key]
                # convert numpy arrays into json serializable format
                if isinstance(val, (np.ndarray, np.number)):
                    val = val.tolist()
                monitor_data[key] = val

        # convert to json and write to file
        json_monitor_data = json.dumps(monitor_data, allow_nan=False)
        with open(self._json_path, "a") as f:
            f.write(json_monitor_data)
            f.write("\n")

        # optionally remove keys (mutating the result)
        if self.remove_matched:
            for key in monitor_keys:
                res.pop(key)
