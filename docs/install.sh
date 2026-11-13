#!/bin/sh
# Starsim installer for Linux and macOS.
# Installs Miniforge (if conda isn't found) and pip-installs Starsim and Jupyter.
#
# Usage:
#   curl -LsSf https://docs.starsim.org/install.sh | sh
# or:
#   sh install.sh

set -eu

MINIFORGE_DIR="${HOME}/miniforge3"

detect_installer() {
    case "$(uname -s)" in
        Linux)  os="Linux" ;;
        Darwin) os="MacOSX" ;;
        *) echo "Unsupported OS: $(uname -s)" >&2; exit 1 ;;
    esac
    case "$(uname -m)" in
        x86_64)
            arch="x86_64" ;;
        arm64|aarch64)
            if [ "$os" = "Linux" ]; then arch="aarch64"; else arch="arm64"; fi ;;
        *)
            echo "Unsupported CPU architecture: $(uname -m)" >&2; exit 1 ;;
    esac
    echo "Miniforge3-${os}-${arch}.sh"
}

find_conda() {
    if command -v conda >/dev/null 2>&1; then
        command -v conda
    elif [ -x "${MINIFORGE_DIR}/bin/conda" ]; then
        echo "${MINIFORGE_DIR}/bin/conda"
    fi
}

install_miniforge() {
    installer="$(detect_installer)"
    url="https://github.com/conda-forge/miniforge/releases/latest/download/${installer}"
    tmpdir="$(mktemp -d)"  # Miniforge's installer requires its filename to end in .sh

    echo ">>> Downloading ${url}"
    curl -fsSL "${url}" -o "${tmpdir}/${installer}"

    echo ">>> Installing Miniforge to ${MINIFORGE_DIR}"
    sh "${tmpdir}/${installer}" -b -p "${MINIFORGE_DIR}"
    rm -rf "${tmpdir}"

    echo ">>> Initializing conda for bash and zsh"
    "${MINIFORGE_DIR}/bin/conda" init bash zsh >/dev/null || true
}

main() {
    echo "=== Starsim installer ==="

    conda="$(find_conda || true)"
    if [ -z "${conda}" ]; then
        install_miniforge
        conda="${MINIFORGE_DIR}/bin/conda"
    else
        echo ">>> Found existing conda at: ${conda}"
    fi

    # Starsim needs Python 3.12 or later; an older base env would otherwise get an old Starsim
    if ! "${conda}" run -n base python -c "import sys; sys.exit(sys.version_info < (3, 12))"; then
        version="$("${conda}" run -n base python --version)"
        echo "Starsim needs Python 3.12 or later, but the base conda env has ${version}." >&2
        echo "Update it with 'conda install -n base python=3.13', or install Starsim into a new env." >&2
        exit 1
    fi

    echo ">>> Installing Starsim and Jupyter into the base conda env"
    "${conda}" run --no-capture-output -n base pip install --upgrade pip
    "${conda}" run --no-capture-output -n base pip install starsim jupyter

    cat <<EOF

=== Done ===
Open a new terminal and check the install with:
    python -c "import starsim as ss; print(ss.__version__)"

Launch Jupyter with:
    jupyter notebook
EOF
}

main "$@"
