#!/usr/bin/bash
set -e

main() {
    build_if_not_exists "$IMAGE_NAME" "$SCRIPT_DIR/texlive"

    if [[ "$clean" == true ]]; then
        remove_aux_files
        exit 0
    fi

    if [[ "$autocompile" == true ]]; then
        run_latexmk -pvc "${EXTRA_ARGS[@]}"
    else
        run_latexmk "${EXTRA_ARGS[@]}"
    fi
}

_setConfigArgs() {
    autocompile=false
    clean=false
    EXTRA_ARGS=()
    MAIN_TEX=""

    ## Options
    while [ $# -gt 0 ]; do
        case "$1" in
        '-a' | '--autocompile')
            autocompile=true
            ;;
        '-c' | '--clean')
            clean=true
            ;;

        ## end of Options
        -*)
            EXTRA_ARGS+=("$1")
            ;;
        *)
            if [ -z "$MAIN_TEX" ]; then
                MAIN_TEX="$1"
            else
                EXTRA_ARGS+=("$1")
            fi
            ;;
        esac
        shift
    done

    MAIN_TEX=${MAIN_TEX:-"$SCRIPT_DIR/docs/main.tex"}
    MAIN_FILE_ONLY=$(basename "$MAIN_TEX")
    MAIN_DIR=$(realpath "$(dirname "$MAIN_TEX")")
    AUX_DIR="$MAIN_DIR/.tmp"
}

IMAGE_NAME=arc-agi-1:latest

build_if_not_exists() {
    if [[ "$(docker images -q "$1" 2>/dev/null)" == "" ]]; then
        echo "Image $1 does not exist. Building..."
        docker build --force-rm --tag "$1" "$2"
    else
        echo "Image $1 already exists. Skipping build."
    fi
}
run_docker() { docker run --rm -v "$MAIN_DIR:/data" -w /data "$IMAGE_NAME" "$@"; }
remove_aux_files() { run_docker latexmk -aux-directory=.tmp -c "$MAIN_FILE_ONLY"; }
run_latexmk() { run_docker latexmk -aux-directory=.tmp -pdf "$@" "$MAIN_FILE_ONLY"; }
count_on_log() { grep --ignore-case --count --perl-regexp --regexp="$1" "${@:2}" "$AUX_DIR/${MAIN_FILE_ONLY%.tex}.log"; }

SCRIPT_DIR=$(dirname "$(readlink -e "${BASH_SOURCE[0]}")")
_setConfigArgs "$@"
main