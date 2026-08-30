#!/usr/bin/bash

main() {
    build_if_not_exists $IMAGE_NAME "$SCRIPT_DIR/texlive"

    if [[ "$clean" == true ]]; then
        remove_aux_files
        exit 0
    fi

    if [[ "$autocompile" == true ]]; then
        run_latexmk -pvc "$@"
    else
        run_latexmk "$@"
    fi
}

_setConfigArgs() {
    ## Options
    while [ "${1:-}" != '' ]; do
        case "$1" in
        '-a' | '--autocompile')
            autocompile=true
            shift
            ;;
        '-c' | '--clean')
            clean=true
            shift
            ;;

        ## end of Options
        [!-]*)
            break
            ;;
        *)
            log "$WARN" "Unknown option \"$1\", ignoring" 0
            ;;
        esac
        shift
    done

    ## Positional
    if [ "${1:-}" != '' ]; then MAIN_TEX=$1; fi

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
run_docker() { echo $MAIN_DIR; docker run --rm -v "$MAIN_DIR:/data" -w /data $IMAGE_NAME "$@"; }
remove_aux_files() { run_docker latexmk -aux-directory=.tmp -c; }
run_latexmk() { run_docker latexmk -aux-directory=.tmp -pdf "$@" "$MAIN_FILE_ONLY"; }
count_on_log() { grep --ignore-case --count --perl-regexp --regexp="$1" "${@:2}" main.log; }

SCRIPT_DIR=$(dirname "$(readlink -e "${BASH_SOURCE[0]}")")
_setConfigArgs "$@"
main "$@"