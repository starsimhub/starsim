"""
Compiling Starsim, and setting up a clone of it for development: `starsim init compile` (or building the Rust kernels in its place) and `starsim init dev`
"""
import os
import sys
import shutil
import subprocess
import importlib.util
from pathlib import Path
from . import ai

__all__ = ['compile', 'build_rust', 'init_dev']


def compile(args=None, needed=False):
    """
    Compile Starsim's Numba kernels now, so that no run waits for them (see `ss.compile()`), even if the Rust kernels are in use;
    or with --check, say whether they are compiled. With `needed` (as for "starsim init"), only if Numba's are the ones in use.
    In a clone of Starsim whose Rust kernels are not built, offer to build them instead (or do so with --rust; --numba skips asking).
    """
    import starsim as ss
    if getattr(args, 'check', False):
        status = ss.compile_status()
        print(status.summary)
        sys.exit(0 if status.compiled else 1)
    if choose_rust(args, built=ss.solvers.backends.rust.built()):
        build_rust(again='starsim init compile --rust')
        print()
        subprocess.run([sys.executable, '-m', 'starsim', 'init', 'compile', '--check']) # A new process, since this one started before the build
        return
    if needed and ss.solvers.backends.rust.on:
        print('Nothing to compile: the Rust kernels are in use, and are compiled already ("starsim init compile" compiles Numba\'s too)')
        return
    with ss.options.context(kernels='numba'): # Called for, so Numba's, even if the Rust ones are in use
        ss.compile()
    return


def rust_crate():
    """ The folder of the Rust kernels' source, or None if this Starsim was installed from a wheel (which has none) """
    crate = ai.package_folder.parent / 'starsim' / 'solvers' / 'rust'
    return crate if (crate / 'Cargo.toml').exists() else None


def choose_rust(args, built):
    """ Whether to build the Rust kernels rather than compile Numba's: as --rust or --numba say, or else, in a clone without them, as the user answers """
    if getattr(args, 'rust', False):
        if built:
            print('The Rust kernels are built already: to rebuild them, run "./build" in starsim/solvers/rust/')
            return False
        if not rust_crate():
            sys.exit('The Rust kernels can only be built in a clone of Starsim (git clone https://github.com/starsimhub/starsim, then pip install -e . there); without one, Starsim uses its Numba kernels')
        return True
    if built or getattr(args, 'numba', False) or not rust_crate():
        return False
    if getattr(args, 'yes', False) or not sys.stdin.isatty():
        print('The Rust kernels are not built, so compiling the Numba ones ("starsim init compile --rust" builds the Rust ones instead)')
        return False
    steps = ['Install Rust with rustup (https://rustup.rs), in ~/.cargo and ~/.rustup, since it is not installed'] if not find_cargo() else []
    steps += ['Build the Rust kernels with cargo, from the source in starsim/solvers/rust/ (with "./build" there; about a minute)']
    print('The Rust kernels are not built. They give the same numbers as the Numba ones, and run at least as fast in nearly every case.')
    print('Building them will:')
    for i,step in enumerate(steps):
        print(f'  {i+1}. {step}')
    print('Otherwise, the Numba kernels are compiled now (about five minutes, once per install).')
    try:
        answer = input('Build the Rust kernels? [Y/n] ').strip().lower()
    except EOFError:
        answer = 'n'
    print()
    return answer in ('', 'y', 'yes')


def find_cargo():
    """ The path to a working cargo, or None; it may not be on the path yet, since rustup's installer only adds it for new shells """
    folder = Path(os.environ.get('CARGO_HOME', '~/.cargo')).expanduser() / 'bin'
    for cargo in [shutil.which('cargo'), shutil.which('cargo', path=folder)]:
        if cargo and subprocess.run([cargo, '--version'], capture_output=True).returncode == 0: # Not so if rustup has no toolchain yet
            return cargo
    return None


def install_rust(again='starsim init dev'):
    """ Check for a Rust toolchain, install it with rustup if there is none, and return the path to cargo; `again` is the command to suggest rerunning """
    cargo = find_cargo()
    if cargo:
        print(f'Found Rust: {subprocess.run([cargo, "--version"], capture_output=True, text=True).stdout.strip()}')
        return cargo
    if sys.platform == 'win32' or not shutil.which('curl'):
        sys.exit(f'Rust is not installed: install it from https://rustup.rs, then run "{again}" again')
    print('Rust is not installed: installing it with rustup (https://rustup.rs), to ~/.cargo and ~/.rustup')
    subprocess.run("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal", shell=True)
    cargo = find_cargo()
    if not cargo:
        sys.exit(f'Rust could not be installed: install it from https://rustup.rs, then run "{again}" again')
    return cargo


def build_rust(again='starsim init dev'):
    """ Build the Rust kernels in a clone of Starsim, installing Rust first if it is missing; `again` is the command to suggest rerunning """
    crate = rust_crate()
    if not shutil.which('bash'):
        sys.exit(f'The Rust kernels are built by a bash script: install bash (on Windows, Git Bash), then run "{again}" again')
    print('Building the Rust kernels ...')
    cargo = install_rust(again)
    path = os.pathsep.join([str(Path(cargo).parent), os.environ['PATH']])
    if subprocess.run(['bash', 'build'], cwd=crate, env=os.environ | dict(PATH=path)).returncode:
        sys.exit(f'The Rust kernels could not be built; if the error is about the version of Rust, run "rustup update", then run "{again}" again')
    return


def init_dev(args):
    """ Set up a clone of Starsim for development: the dev dependencies, Rust, both sets of kernels, and the skills for AI agents """
    root = ai.package_folder.parent
    if not rust_crate():
        sys.exit(f'"starsim init dev" is for a clone of Starsim, but this one ({root}) is installed from a wheel: git clone https://github.com/starsimhub/starsim, then pip install -e . there')
    if not shutil.which('bash'): # Checked here too, so as not to stop after installing the dependencies
        sys.exit('The Rust kernels are built by a bash script: install bash (on Windows, Git Bash), then run "starsim init dev" again')

    print('Installing the dependencies for development (tests, docs, and JAX) ...')
    pip = [sys.executable, '-m', 'pip'] if importlib.util.find_spec('pip') else ['uv', 'pip'] # uv's environments have no pip
    target = [] if pip[0] != 'uv' else ['--python', sys.executable]
    subprocess.run([*pip, 'install', *target, '-e', '.[dev]'], cwd=root, env=os.environ | dict(CARGO='false'), check=True) # CARGO=false since ./build compiles them below

    print()
    build_rust()

    print('\nCompiling the Numba kernels, which the tests compare with the Rust ones (a few minutes, once) ...')
    subprocess.run([sys.executable, '-m', 'starsim', 'init', 'compile'], env=os.environ | dict(STARSIM_KERNELS='numba'), check=True) # A new process, since this one started before the build

    print()
    ai.install(args)
    print()
    subprocess.run([sys.executable, '-m', 'starsim', 'init', 'compile', '--check'], check=True)
    print('Ready for development. To check: cd tests && ./run_tests')
    return
