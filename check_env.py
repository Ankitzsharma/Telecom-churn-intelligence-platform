import sys
print("Python version:", sys.version)
try:
    import pandas as pd
    print("Pandas version:", pd.__version__)
except ImportError as e:
    print("Pandas import failed:", e)

try:
    import numpy as np
    print("Numpy version:", np.__version__)
except ImportError as e:
    print("Numpy import failed:", e)
