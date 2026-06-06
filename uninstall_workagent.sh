#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"
REQUIREMENTS_FILE="$BACKEND_DIR/requirements.txt"
NODE_MODULES_DIR="$ROOT_DIR/frontend/node_modules"
LATEX_WARMUP_DIR="$ROOT_DIR/outputs/latex_install_warmup"
PYTHON_CMD=()

confirm_action() {
    local prompt="$1"
    local answer
    read -r -p "$prompt [y/N] " answer || true
    [[ "$answer" =~ ^([yY][eE][sS]?|[yY])$ ]]
}

fail() {
    printf '%s\n' "$1" >&2
    exit 1
}

run_checked() {
    local failure_message="$1"
    shift
    "$@" || fail "$failure_message"
}

resolve_python_command() {
    if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
        PYTHON_CMD=("${CONDA_PREFIX}/bin/python")
        return
    fi

    if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
        PYTHON_CMD=("${VIRTUAL_ENV}/bin/python")
        return
    fi

    if command -v python >/dev/null 2>&1; then
        PYTHON_CMD=("$(command -v python)")
        return
    fi

    if command -v python3 >/dev/null 2>&1; then
        PYTHON_CMD=("$(command -v python3)")
        return
    fi

    PYTHON_CMD=()
}

python_has_pip() {
    [[ ${#PYTHON_CMD[@]} -gt 0 ]] || return 1
    "${PYTHON_CMD[@]}" -m pip --version >/dev/null 2>&1
}

remove_directory_inside_workspace() {
    local target_path="$1"
    local label="$2"
    local resolved_root
    local resolved_target

    if [[ ! -e "$target_path" ]]; then
        printf '%s not found; skipping.\n' "$label"
        return
    fi

    resolved_root="$(realpath "$ROOT_DIR")"
    resolved_target="$(realpath "$target_path")"

    case "$resolved_target" in
        "$resolved_root"/*) ;;
        *)
            fail "Refusing to remove path outside workspace: $resolved_target"
            ;;
    esac

    printf 'Removing %s...\n' "$label"
    rm -rf -- "$resolved_target"
}

is_ubuntu_like() {
    [[ -f /etc/os-release ]] && grep -Eiq '(^ID=ubuntu$|^ID=debian$|^ID_LIKE=.*(ubuntu|debian))' /etc/os-release
}

apt_package_installed() {
    dpkg -s "$1" >/dev/null 2>&1
}

uninstall_python_packages() {
    if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
        printf 'Backend requirements file not found; skipping Python package uninstall.\n'
        return
    fi

    resolve_python_command
    if [[ ${#PYTHON_CMD[@]} -eq 0 ]]; then
        printf 'Python interpreter not found; skipping Python package uninstall.\n'
        return
    fi

    if ! python_has_pip; then
        printf 'pip is not available for %s; skipping Python package uninstall.\n' "${PYTHON_CMD[*]}"
        return
    fi

    printf 'Uninstalling backend Python packages from the current Python environment...\n'
    (
        cd "$BACKEND_DIR"
        run_checked "Python package uninstall failed." "${PYTHON_CMD[@]}" -m pip uninstall -r requirements.txt -y
    )
}

uninstall_apt_packages_if_present() {
    local packages=("$@")
    local installed_packages=()
    local sudo_cmd=()
    local package

    for package in "${packages[@]}"; do
        if apt_package_installed "$package"; then
            installed_packages+=("$package")
        fi
    done

    if [[ ${#installed_packages[@]} -eq 0 ]]; then
        printf 'No matching Ubuntu packages detected; skipping.\n'
        return
    fi

    if ! command -v apt-get >/dev/null 2>&1; then
        printf 'apt-get not found; uninstall these packages manually if needed: %s\n' "${installed_packages[*]}"
        return
    fi

    if [[ $EUID -ne 0 ]]; then
        command -v sudo >/dev/null 2>&1 || fail "sudo not found. Rerun this script as root or uninstall these packages manually: ${installed_packages[*]}"
        sudo_cmd=(sudo)
    fi

    printf 'Uninstalling Ubuntu packages: %s\n' "${installed_packages[*]}"
    run_checked "apt-get remove failed." "${sudo_cmd[@]}" apt-get remove -y "${installed_packages[@]}"
}

printf 'WorkAgent environment uninstall\n'
printf 'Workspace: %s\n\n' "$ROOT_DIR"

remove_directory_inside_workspace "$NODE_MODULES_DIR" "frontend node_modules"
remove_directory_inside_workspace "$LATEX_WARMUP_DIR" "LaTeX package warmup files"

printf '\nPython packages were installed into the current Python environment.\n'
printf 'Only uninstall them if this Python environment is dedicated to WorkAgent.\n'
if confirm_action "Uninstall backend Python packages from backend/requirements.txt?"; then
    uninstall_python_packages
else
    printf 'Skipping Python package uninstall.\n'
fi

printf '\nUbuntu LaTeX packages may be shared by other projects.\n'
if confirm_action "Uninstall Ubuntu LaTeX packages installed for PDF export?"; then
    if is_ubuntu_like; then
        uninstall_apt_packages_if_present \
            texlive-xetex \
            texlive-latex-extra \
            texlive-fonts-recommended \
            texlive-fonts-extra \
            texlive-lang-english \
            lmodern
    else
        printf 'Non-Ubuntu system detected; uninstall LaTeX packages manually if needed.\n'
    fi
else
    printf 'Skipping LaTeX package uninstall.\n'
fi

printf '\nPerl may be shared by other tools.\n'
if confirm_action "Uninstall perl installed for latexmk support?"; then
    if is_ubuntu_like; then
        uninstall_apt_packages_if_present perl
    else
        printf 'Non-Ubuntu system detected; uninstall perl manually if needed.\n'
    fi
else
    printf 'Skipping perl uninstall.\n'
fi

printf '\nWorkAgent environment uninstall completed.\n'
