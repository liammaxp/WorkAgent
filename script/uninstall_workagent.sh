#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NODE_MODULES_DIR="$ROOT_DIR/frontend/node_modules"
LATEX_WARMUP_DIR="$ROOT_DIR/outputs/latex_install_warmup"
VENV_DIR="$ROOT_DIR/.venv"

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

remove_directory_inside_workspace() {
    local target_path="$1"
    local label="$2"
    local resolved_root
    local resolved_target

    if [[ ! -e "$target_path" ]]; then
        printf '%s not found; skipping.\n' "$label"
        return
    fi

    resolved_root="$(cd "$ROOT_DIR" && pwd -P)"
    resolved_target="$(cd "$target_path" && pwd -P)"

    case "$resolved_target" in
        "$resolved_root"/*) ;;
        *)
            fail "Refusing to remove path outside workspace: $resolved_target"
            ;;
    esac

    printf 'Removing %s...\n' "$label"
    rm -rf -- "$resolved_target"
}

apt_package_installed() {
    dpkg -s "$1" >/dev/null 2>&1
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

uninstall_system_latex() {
    if [[ "$(uname -s)" == "Darwin" ]]; then
        if ! command -v brew >/dev/null 2>&1; then
            printf 'Homebrew not found; remove MacTeX manually if needed.\n'
        elif brew list --cask mactex-no-gui >/dev/null 2>&1; then
            run_checked "Homebrew MacTeX uninstall failed." brew uninstall --cask mactex-no-gui
        else
            printf 'Homebrew mactex-no-gui is not installed; skipping.\n'
        fi
    elif command -v apt-get >/dev/null 2>&1; then
        uninstall_apt_packages_if_present \
            texlive-xetex texlive-latex-extra texlive-fonts-recommended \
            texlive-fonts-extra latexmk perl
    elif command -v dnf >/dev/null 2>&1; then
        run_checked "dnf remove failed." run_as_root dnf remove -y \
            texlive-scheme-medium texlive-collection-fontsextra latexmk perl
    elif command -v yum >/dev/null 2>&1; then
        run_checked "yum remove failed." run_as_root yum remove -y \
            texlive-scheme-medium texlive-collection-fontsextra latexmk perl
    elif command -v pacman >/dev/null 2>&1; then
        local packages=()
        local package
        for package in texlive-basic texlive-latex texlive-latexextra texlive-fontsrecommended texlive-fontsextra latexmk perl; do
            pacman -Q "$package" >/dev/null 2>&1 && packages+=("$package")
        done
        if [[ ${#packages[@]} -gt 0 ]]; then
            run_checked "pacman remove failed." run_as_root pacman -Rns --noconfirm "${packages[@]}"
        else
            printf 'No matching pacman LaTeX packages detected; skipping.\n'
        fi
    elif command -v zypper >/dev/null 2>&1; then
        run_checked "zypper remove failed." run_as_root zypper --non-interactive remove \
            texlive-scheme-medium texlive-xetex latexmk perl
    else
        printf 'No supported package manager found; remove the LaTeX toolchain manually if needed.\n'
    fi
}

run_as_root() {
    if [[ "${EUID:-$(id -u)}" -eq 0 ]]; then
        "$@"
    elif command -v sudo >/dev/null 2>&1; then
        sudo "$@"
    else
        fail "Administrator privileges are required. Install sudo or rerun as root."
    fi
}

printf 'WorkAgent environment uninstall\n'
printf 'Workspace: %s\n\n' "$ROOT_DIR"

remove_directory_inside_workspace "$NODE_MODULES_DIR" "frontend node_modules"
remove_directory_inside_workspace "$LATEX_WARMUP_DIR" "LaTeX package warmup files"
remove_directory_inside_workspace "$VENV_DIR" "WorkAgent Python virtual environment"

printf '\nThe system LaTeX/Perl toolchain may be shared by other projects.\n'
if confirm_action "Uninstall the LaTeX/Perl packages used for PDF export?"; then
    uninstall_system_latex
else
    printf 'Skipping system LaTeX/Perl uninstall.\n'
fi

printf '\nWorkAgent environment uninstall completed.\n'
