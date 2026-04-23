# conftest.py
# ------------
# Shared pytest configuration.
# Adds the project root to sys.path so all tests can import
# prunable_layer, model, train, evaluate, and main without the
# user needing to install the package.

import sys
import os

# Project root is the directory this file lives in
sys.path.insert(0, os.path.dirname(__file__))
