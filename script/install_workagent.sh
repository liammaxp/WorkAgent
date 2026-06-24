#!/usr/bin/env bash

set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="$ROOT/backend"
FRONTEND_DIR="$ROOT/frontend"
REQUIREMENTS_FILE="$BACKEND_DIR/requirements.txt"
PACKAGE_FILE="$FRONTEND_DIR/package.json"
LATEX_WARMUP_DIR="$ROOT/outputs/latex_install_warmup"
VENV_DIR="$ROOT/.venv"

fail() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

require_command() {
    local name="$1"
    local hint="$2"
    command -v "$name" >/dev/null 2>&1 || fail "Required command not found: $name. $hint"
}

run_as_root() {
    if [[ "${EUID:-$(id -u)}" -eq 0 ]]; then
        "$@"
    elif command -v sudo >/dev/null 2>&1; then
        sudo "$@"
    else
        fail "Administrator privileges are required. Install sudo or run this script as root."
    fi
}

find_python() {
    if command -v python3 >/dev/null 2>&1; then
        printf '%s\n' "python3"
    elif command -v python >/dev/null 2>&1; then
        printf '%s\n' "python"
    else
        fail "Python 3 is required. Install it and make python3 available in PATH."
    fi
}

create_virtualenv() {
    local system_python="$1"

    if [[ -x "$VENV_DIR/bin/python" ]]; then
        printf 'WorkAgent virtual environment already available.\n'
        return
    fi

    printf 'Creating WorkAgent virtual environment...\n'
    if "$system_python" -m venv "$VENV_DIR"; then
        return
    fi

    if command -v apt-get >/dev/null 2>&1; then
        printf 'Python venv support is missing. Installing python3-venv...\n'
        run_as_root apt-get update
        run_as_root apt-get install -y python3-venv
        "$system_python" -m venv "$VENV_DIR"
        return
    fi

    fail "Could not create .venv. Install the Python venv module for your platform, then rerun this script."
}

has_latex_compiler() {
    command -v latexmk >/dev/null 2>&1 ||
        command -v xelatex >/dev/null 2>&1 ||
        command -v pdflatex >/dev/null 2>&1
}

install_latex_toolchain() {
    if has_latex_compiler; then
        printf 'LaTeX compiler already available.\n'
        return
    fi

    printf '\nLaTeX compiler not found. Installing TeX Live for PDF export...\n'

    if command -v apt-get >/dev/null 2>&1; then
        run_as_root apt-get update
        run_as_root apt-get install -y \
            texlive-xetex texlive-latex-extra texlive-fonts-recommended \
            texlive-fonts-extra latexmk perl
    elif command -v dnf >/dev/null 2>&1; then
        run_as_root dnf install -y texlive-scheme-medium texlive-collection-fontsextra latexmk perl
    elif command -v yum >/dev/null 2>&1; then
        run_as_root yum install -y texlive-scheme-medium texlive-collection-fontsextra latexmk perl
    elif command -v pacman >/dev/null 2>&1; then
        run_as_root pacman -Syu --needed --noconfirm \
            texlive-basic texlive-latex texlive-latexextra texlive-fontsrecommended \
            texlive-fontsextra latexmk perl
    elif command -v zypper >/dev/null 2>&1; then
        run_as_root zypper --non-interactive install \
            texlive-scheme-medium texlive-xetex latexmk perl
    elif command -v brew >/dev/null 2>&1; then
        brew install --cask mactex-no-gui
        export PATH="/Library/TeX/texbin:$PATH"
    else
        fail "No supported package manager found. Install TeX Live with xelatex or pdflatex, then rerun this script."
    fi

    has_latex_compiler || fail "TeX Live was installed, but latexmk/xelatex/pdflatex was not found in PATH."
}

initialize_latex_packages() {
    local compiler warmup_tex

    if command -v xelatex >/dev/null 2>&1; then
        compiler="$(command -v xelatex)"
    elif command -v pdflatex >/dev/null 2>&1; then
        compiler="$(command -v pdflatex)"
    else
        printf 'No xelatex or pdflatex command found; skipping LaTeX package warmup.\n'
        return
    fi

    printf '\nWarming up TeX Live packages for resume PDF export...\n'
    mkdir -p "$LATEX_WARMUP_DIR"
    warmup_tex="$LATEX_WARMUP_DIR/workagent_latex_warmup.tex"

    if [[ "$(basename "$compiler")" == "xelatex" ]]; then
        cat >"$warmup_tex" <<'EOF'
\documentclass[11pt]{article}
\usepackage[margin=0.5in]{geometry}
\usepackage{fontspec}
\usepackage{xcolor}
\usepackage{hyperref}
\usepackage{enumitem}
\usepackage{titlesec}
\usepackage{tabularx}
\usepackage{array}
\usepackage{ragged2e}
\usepackage{fontawesome5}
\hypersetup{colorlinks=true,urlcolor=blue}
\titleformat{\section}{\large\bfseries}{}{0pt}{}
\begin{document}
\section{WorkAgent LaTeX Warmup}
\begin{tabularx}{\textwidth}{X r}
\textbf{Tailored Resume PDF Export} & \href{https://example.com}{example link} \\
\end{tabularx}
\begin{itemize}[leftmargin=*]
\item Common resume packages are installed and ready. \faGithub
\end{itemize}
\end{document}
EOF
    else
        cat >"$warmup_tex" <<'EOF'
\documentclass[11pt]{article}
\usepackage[margin=0.5in]{geometry}
\usepackage{xcolor}
\usepackage{hyperref}
\usepackage{enumitem}
\usepackage{titlesec}
\usepackage{tabularx}
\usepackage{array}
\usepackage{ragged2e}
\hypersetup{colorlinks=true,urlcolor=blue}
\titleformat{\section}{\large\bfseries}{}{0pt}{}
\begin{document}
\section{WorkAgent LaTeX Warmup}
\begin{tabularx}{\textwidth}{X r}
\textbf{Tailored Resume PDF Export} & \href{https://example.com}{example link} \\
\end{tabularx}
\begin{itemize}[leftmargin=*]
\item Common resume packages are installed and ready.
\end{itemize}
\end{document}
EOF
    fi

    "$compiler" -interaction=nonstopmode -halt-on-error \
        -output-directory="$LATEX_WARMUP_DIR" "$warmup_tex"
    "$compiler" -interaction=nonstopmode -halt-on-error \
        -output-directory="$LATEX_WARMUP_DIR" "$warmup_tex"
    printf 'LaTeX package warmup completed.\n'
}

[[ -f "$REQUIREMENTS_FILE" ]] || fail "Backend requirements file not found: $REQUIREMENTS_FILE"
[[ -f "$PACKAGE_FILE" ]] || fail "Frontend package file not found: $PACKAGE_FILE"

SYSTEM_PYTHON="$(find_python)"
require_command npm "Install Node.js and npm, then rerun this script."
create_virtualenv "$SYSTEM_PYTHON"
PYTHON="$VENV_DIR/bin/python"

printf 'Installing WorkAgent backend dependencies...\n'
(cd "$BACKEND_DIR" && "$PYTHON" -m pip install --upgrade pip && "$PYTHON" -m pip install -r requirements.txt)

printf '\nInstalling WorkAgent frontend dependencies...\n'
(cd "$FRONTEND_DIR" && npm install)

install_latex_toolchain
initialize_latex_packages

printf '\nDependency installation completed successfully.\n'
