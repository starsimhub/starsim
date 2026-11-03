"""
Compiling Starsim, and setting up a clone of it for development: `starsim init compile` and `starsim init dev`
"""
import os
import sys
import shutil
import subprocess
import importlib.util
from pathlib import Path
from . import ai

__all__ = ['compile', 'init_dev']


def compile(args=None):
    """ Compile Starsim's Numba kernels now, so that no run waits for them (see `ss.compile()`), or with --check, say whether they are compiled """
    import starsim as ss
    if getattr(args, 'check', False):
        status = ss.compile_status()
        print(status.summary)
        sys.exit(0 if status.compiled else 1)
    ss.compile()
    return


def find_cargo():
    """ The path to a working cargo, or None; it may not be on the path yet, since rustup's installer only adds it for new shells """
    folder = Path(os.environ.get('CARGO_HOME', '~/.cargo')).expanduser() / 'bin'
    for cargo in [shutil.which('cargo'), shutil.which('cargo', path=folder)]:
        if cargo and subprocess.run([cargo, '--version'], capture_output=True).returncode == 0: # Not so if rustup has no toolchain yet
            return cargo
    return None


def install_rust():
    """ Check for a Rust toolchain, install it with rustup if there is none, and return the path to cargo """
    cargo = find_cargo()
    if cargo:
        print(f'Found Rust: {subprocess.run([cargo, "--version"], capture_output=True, text=True).stdout.strip()}')
        return cargo
    if sys.platform == 'win32' or not shutil.which('curl'):
        sys.exit('Rust is not installed: install it from https://rustup.rs, then run "starsim init dev" again')
    print('Rust is not installed: installing it with rustup (https://rustup.rs), to ~/.cargo and ~/.rustup')
    subprocess.run("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal", shell=True)
    cargo = find_cargo()
    if not cargo:
        sys.exit('Rust could not be installed: install it from https://rustup.rs, then run "starsim init dev" again')
    return cargo


def init_dev(args):
    """ Set up a clone of Starsim for development: the dev dependencies, Rust, both sets of kernels, and the skills for AI agents """
    root = ai.package_folder.parent
    crate = root / 'starsim' / 'solvers' / 'rust'
    if not (crate / 'Cargo.toml').exists():
        sys.exit(f'"starsim init dev" is for a clone of Starsim, but this one ({root}) is installed from a wheel: git clone https://github.com/starsimhub/starsim, then pip install -e . there')
    if not shutil.which('bash'):
        sys.exit('The Rust kernels are built by a bash script: install bash (on Windows, Git Bash), then run "starsim init dev" again')

    print('Installing the dependencies for development (tests, docs, and JAX) ...')
    pip = [sys.executable, '-m', 'pip'] if importlib.util.find_spec('pip') else ['uv', 'pip'] # uv's environments have no pip
    target = [] if pip[0] != 'uv' else ['--python', sys.executable]
    subprocess.run([*pip, 'install', *target, '-e', '.[dev]'], cwd=root, env=os.environ | dict(CARGO='false'), check=True) # CARGO=false since ./build compiles them below

    print('\nBuilding the Rust kernels ...')
    cargo = install_rust()
    path = os.pathsep.join([str(Path(cargo).parent), os.environ['PATH']])
    if subprocess.run(['bash', 'build'], cwd=crate, env=os.environ | dict(PATH=path)).returncode:
        sys.exit('The Rust kernels could not be built; if the error is about the version of Rust, run "rustup update" and try again')

    print('\nCompiling the Numba kernels, which the tests compare with the Rust ones (a few minutes, once) ...')
    subprocess.run([sys.executable, '-m', 'starsim', 'init', 'compile'], env=os.environ | dict(STARSIM_KERNELS='numba'), check=True) # A new process, since this one started before the build

    print()
    ai.install(args)
    print()
    subprocess.run([sys.executable, '-m', 'starsim', 'init', 'compile', '--check'], check=True)
    print('Ready for development. To check: cd tests && ./run_tests')
    return
