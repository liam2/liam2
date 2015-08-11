from __future__ import print_function

import os

debug = os.environ.get("DEBUG", False)
input_directory = "."
output_directory = "."
skip_shows = False
assertions = "raise"
show_timings = True
log_level = "procedures"  # should be one of periods, procedures, processes
autodump = None
autodump_file = None
autodiff = None
