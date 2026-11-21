"""
Check what AI agents are given: that the skills reach the wheel, that their code runs. The quick checks are in test_other.py.
"""
import re
import sys
import shutil
import tomllib
import zipfile
import subprocess
from pathlib import Path
from starsim_cli import ai as cli

root = Path(cli.__file__).parents[1]


def test_wheel(tmp_path):
    """ Every skill file is in a wheel: hidden folders are left out unless package-data names them, with no error """
    patterns = tomllib.loads((root/'pyproject.toml').read_text())['tool']['setuptools']['package-data']['starsim']
    shutil.copytree(cli.agents_folder, tmp_path/'starsim'/'.agents', ignore=shutil.ignore_patterns('__pycache__'))
    (tmp_path/'starsim'/'__init__.py').touch()
    (tmp_path/'pyproject.toml').write_text( # Only the skills, since a full wheel compiles the kernels
        '[build-system]\nrequires = ["setuptools"]\nbuild-backend = "setuptools.build_meta"\n'
        '[project]\nname = "skills-test"\nversion = "0"\n'
        f'[tool.setuptools.package-data]\nstarsim = {patterns!r}\n'
    )
    code = "import setuptools.build_meta as build; build.build_wheel('.')"
    subprocess.run([sys.executable, '-c', code], cwd=tmp_path, check=True, capture_output=True)
    with zipfile.ZipFile(next(tmp_path.glob('*.whl'))) as wheel:
        built = set(wheel.namelist())
    files = [path for folder in [cli.ai_folder, cli.meta_folder] for path in folder.rglob('*') if path.is_file() and '__pycache__' not in path.parts]
    expected = {path.relative_to(cli.agents_folder.parents[1]).as_posix() for path in files}
    assert expected and expected <= built, f'Not in the wheel: {sorted(expected - built)}'
    return


def test_guide_code():
    """ The code in the guide runs: it's the first thing an agent copies """
    text = (cli.ai_folder/'starsim'/'SKILL.md').read_text(encoding='utf-8')
    blocks = re.findall(r'```python\n(.*?)```', text, flags=re.S)
    assert blocks
    for block in blocks:
        exec(block.replace('sim.plot()', ''), {}) # Without the figure
    return

