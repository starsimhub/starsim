"""
Check what AI agents are given: that the skills reach the wheel, that their code runs, and that the docs' index covers the docs. The quick checks are in test_other.py.
"""
import re
import sys
import shutil
import tomllib
import zipfile
import subprocess
import importlib.util
from pathlib import Path
import sciris as sc
from starsim.internals import cli

root = Path(cli.__file__).parents[2]
docs = root / 'docs'


def test_wheel(tmp_path):
    """ Every skill file is in a wheel: hidden folders are left out unless package-data names them, with no error """
    sc.heading('Testing that the skills are in the wheel')
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
    sc.heading('Testing the code in the guide')
    text = (cli.ai_folder/'starsim'/'SKILL.md').read_text(encoding='utf-8')
    blocks = re.findall(r'```python\n(.*?)```', text, flags=re.S)
    assert blocks
    for block in blocks:
        exec(block.replace('sim.plot()', ''), {}) # Without the figure
    return


def test_guide_pages():
    """ The pages that the guide sends agents to exist """
    sc.heading('Testing the pages in the guide')
    text = (cli.ai_folder/'starsim'/'SKILL.md').read_text(encoding='utf-8')
    table = text.split('| Task | Pages |')[1].split('\n\n')[0]
    pages = re.findall(r'`([a-z0-9_]+)`', table)
    assert pages
    missing = [page for page in pages if not list(docs.glob(f'*/{page}.qmd'))]
    assert not missing, f'No such pages: {missing}'
    return


def test_docs_index():
    """ The index for agents lists every tutorial and user guide page, each with a summary """
    sc.heading('Testing the docs index')
    spec = importlib.util.spec_from_file_location('make_api', docs/'make_api.py')
    make_api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(make_api)
    guide = make_api.make_guide()
    pages = [page for section in guide for page in section['pages']]
    missing = [page['url'] for page in pages if not page['summary']]
    assert not missing, f'Pages with no summary: {missing}'
    listed = {page['url'].split('/docs/')[1] for page in pages}
    for folder in make_api.guide_sections:
        on_disk = {path.relative_to(docs).as_posix() for path in (docs/folder).rglob('*.qmd') if not path.name.startswith('_') and '.jupyter_cache' not in path.parts}
        unlisted = {path for path in on_disk - listed if path.count('/') and not path.endswith(('/index.qmd', '/solutions.qmd'))}
        assert not unlisted, f'Pages not in the index: {sorted(unlisted)}'
    return
