"""Use the installed Genesis without writing into its installation."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for key, folder in {
    'NUMBA_CACHE_DIR': 'numba', 'GS_CACHE_FILE_PATH': 'genesis',
    'QD_OFFLINE_CACHE_FILE_PATH': 'quadrants', 'XDG_CACHE_HOME': 'xdg',
    'MPLCONFIGDIR': 'matplotlib',
}.items():
    path = ROOT / '.cache' / folder
    path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(key, str(path))
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
os.environ.setdefault('EGL_PLATFORM', 'surfaceless')
os.environ.setdefault('LIBGL_ALWAYS_SOFTWARE', '1')
os.environ.setdefault('OMP_NUM_THREADS', '8')

