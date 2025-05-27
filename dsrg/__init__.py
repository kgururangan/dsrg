import os
import sys
import platform
import datetime
import subprocess
import importlib.util
from importlib.metadata import version as dist_version, PackageNotFoundError

import numpy as np
try:
    import pyscf
except ImportError:
    pyscf = None
try:
    import fcipy
except ImportError:
    fcipy = None

# your own version module
try:
    from .version import __version__
except ImportError:
    __version__ = "unknown"

def get_git_commit(cwd=None):
    """Return short commit hash for the repo at cwd (or current cwd)."""
    cmd = ["git", "rev-parse", "--short", "HEAD"]
    try:
        return (
            subprocess
            .check_output(cmd, stderr=subprocess.DEVNULL, cwd=cwd)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"

def get_package_location(name):
    spec = importlib.util.find_spec(name)
    if not spec:
        return "not installed"
    if spec.submodule_search_locations:
        return spec.submodule_search_locations[0]
    if spec.origin and spec.origin != "namespace":
        return os.path.dirname(spec.origin)
    return "not found"

def get_dist_version(name):
    try:
        return dist_version(name)
    except PackageNotFoundError:
        mod = sys.modules.get(name)
        return getattr(mod, "__version__", "unknown")

def get_banner():
    if os.environ.get("DSRG_NO_BANNER", "0") == "1":
        return ""

    # Core info
    dsrg_commit  = get_git_commit(os.path.dirname(__file__))
    pyver        = sys.version.split()[0]
    system       = platform.system()
    host         = platform.node()
    now          = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Locations
    dsrg_loc     = os.path.dirname(__file__)
    numpy_loc    = get_package_location("numpy")
    pyscf_loc    = get_package_location("pyscf")
    fcipy_loc    = get_package_location("fcipy")

    # Versions
    numpy_ver    = np.__version__
    blas_lib     = "unknown"
    try:
        import numpy.__config__ as npconf
        info = npconf.get_info("blas_opt_info") or {}
        libs = info.get("libraries") or []
        blas_lib = libs[0] if libs else "unknown"
    except Exception:
        pass

    pyscf_ver    = get_dist_version("pyscf")
    fcipy_ver    = get_dist_version("fcipy")

    # FCIpy commit
    fcipy_commit = get_git_commit(fcipy_loc) if os.path.isdir(fcipy_loc) else "unknown"

    banner = f"""
    DSRG v1.0.0 — Driven Similarity Renormalization Group
    ------------------------------------------------------------
    GitHub Repository: https://github.com/kgururangan/dsrg.git
    Python version:  {pyver}
    Host system:     {host} ({system})
    Loaded at:       {now}
    
    Install Paths:
      DSRG (commit id {dsrg_commit}): {dsrg_loc}
      NumPy: {numpy_loc}
      PySCF: {pyscf_loc}
      FCIpy (commit id {fcipy_commit}): {fcipy_loc}
    
    Versions:
      NumPy:          {numpy_ver}
      PySCF:          {pyscf_ver}
      FCIpy:          {fcipy_ver}
      NumPy BLAS:     {blas_lib}
    ------------------------------------------------------------
    """
    return banner

print(get_banner())