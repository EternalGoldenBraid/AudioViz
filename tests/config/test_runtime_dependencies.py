import ast
from pathlib import Path
import re
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[2]


def test_runtime_imports_are_declared_in_default_dependencies():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    declared = {
        re.match(r"[A-Za-z0-9_.-]+", requirement).group().lower().replace("_", "-")
        for requirement in project["dependencies"]
    }
    imports = set()
    for path in [ROOT / "main.py", *sorted((ROOT / "src/audioviz").rglob("*.py"))]:
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imports.add(node.module.split(".")[0])

    distributions = {"cv2": "opencv-python", "OpenGL": "pyopengl", "PIL": "pillow"}
    # CuPy is loaded lazily only when the optional CUDA backend is selected.
    external = imports - sys.stdlib_module_names - {"audioviz", "cupy"}
    required = {distributions.get(name, name).lower() for name in external}
    assert not required - declared, f"Undeclared runtime dependencies: {sorted(required - declared)}"
