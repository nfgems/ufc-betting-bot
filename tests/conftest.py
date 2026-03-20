import os
import sys
import tempfile
import uuid
from pathlib import Path


_WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
_PYTEST_RUNTIME_ROOT = _WORKSPACE_ROOT / ".pytest-runtime"
_TMP_ROOT = _PYTEST_RUNTIME_ROOT / "tmp"
_MPLCONFIG_ROOT = _PYTEST_RUNTIME_ROOT / "mplconfig"

for path in (_PYTEST_RUNTIME_ROOT, _TMP_ROOT, _MPLCONFIG_ROOT):
    path.mkdir(parents=True, exist_ok=True)

# Keep test temp files inside the repo so pytest and matplotlib avoid locked
# or sandboxed system temp locations on this machine.
for env_name in ("TMP", "TEMP", "TMPDIR"):
    os.environ[env_name] = str(_TMP_ROOT)
os.environ["MPLCONFIGDIR"] = str(_MPLCONFIG_ROOT)
tempfile.tempdir = str(_TMP_ROOT)


def _best_effort_chmod(path: str | os.PathLike[str], mode: int) -> None:
    try:
        os.chmod(path, mode)
    except OSError:
        pass


if os.name == "nt" and sys.version_info >= (3, 14):
    import _pytest.pathlib as pytest_pathlib
    import _pytest.tmpdir as pytest_tmpdir

    _original_make_numbered_dir = pytest_pathlib.make_numbered_dir
    _original_getbasetemp = pytest_tmpdir.TempPathFactory.getbasetemp
    _original_mkdtemp = tempfile.mkdtemp

    def _pytest_safe_mode(mode: int) -> int:
        return 0o777 if mode == 0o700 else mode

    def _make_numbered_dir(root: Path, prefix: str, mode: int = 0o700) -> Path:
        return _original_make_numbered_dir(root, prefix, _pytest_safe_mode(mode))

    def _mkdtemp(*args, **kwargs) -> str:
        path = _original_mkdtemp(*args, **kwargs)
        _best_effort_chmod(path, 0o777)
        return path

    def _getbasetemp(self) -> Path:
        if self._basetemp is not None:
            return self._basetemp

        if self._given_basetemp is not None:
            basetemp = self._given_basetemp.parent / (
                f"{self._given_basetemp.name}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
            )
            basetemp.mkdir(mode=_pytest_safe_mode(0o700))
            basetemp = basetemp.resolve()
            self._basetemp = basetemp
            self._trace("new basetemp", basetemp)
            return basetemp

        return _original_getbasetemp(self)

    pytest_pathlib.make_numbered_dir = _make_numbered_dir
    pytest_tmpdir.make_numbered_dir = _make_numbered_dir
    pytest_tmpdir.TempPathFactory.getbasetemp = _getbasetemp
    tempfile.mkdtemp = _mkdtemp
