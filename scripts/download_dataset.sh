#!/usr/bin/env bash

DATADIR="${UVCGAN_S_DATA:-data}"

declare -A URL_LIST=(
    [afhq]="https://www.dropbox.com/s/t9l9o3vsx2jai3z/afhq.zip"
    [sphenix]="https://zenodo.org/record/17783990/files/2025-06-05_jet_bkg_sub.tar"
)

declare -A CHECKSUMS=(
    [afhq]="7f63dcc14ef58c0e849b59091287e1844da97016073aac20403ae6c6132b950f"
)

die ()
{
    echo "${*}"
    exit 1
}

usage ()
{
    cat <<EOF
USAGE: download_dataset.sh DATASET

where DATASET is one of afhq or sphenix
EOF

    if [[ $# -gt 0 ]]
    then
        die "${*}"
    else
        exit 0
    fi
}

exec_or_die ()
{
    "${@}" || die "Failed to execute: '${*}'"
}

calc_sha256_hash ()
{
    local path="${1}"
    sha256sum "${path}" | cut -d ' ' -f 1 | tr -d '\n'
}

download_archive ()
{
    local url="${1}"
    local archive="${2}"
    local checksum="${3}"

    exec_or_die mkdir -p "${DATADIR}"

    local path="${DATADIR}/${archive}"

    if [[ ! -e "${DATADIR}/${archive}" ]]
    then
        exec_or_die wget --no-check-certificate \
            "${url}" --output-document "${path}"
    fi

    if [[ -n "${checksum}" ]]
    then
        # shellcheck disable=SC2155
        local test_csum="$(calc_sha256_hash "${path}")"

        if [[ "${test_csum}" == "${checksum}" ]]
        then
            echo " - Checksum valid"
        else
            die "Checksum mismatch for '${path}' ${test_csum} vs ${checksum}"
        fi
    fi
}

download_and_extract_zip ()
{
    local url="${1}"
    local zip="${2}"
    local checksum="${3}"

    download_archive  "${url}" "${zip}" "${checksum}"
    exec_or_die unzip "${DATADIR}/${zip}" -d "${DATADIR}"

    # exec_or_die rm "${dst}/${zip}"

    echo " - Dataset is unpacked to '${path}'"
}

check_dset_exists ()
{
    local path="${1}"

    if [[ -e "${path}" ]]
    then

        read -r -p "Dataset '${path}' exists. Overwrite? [yN]: " ret
        case "${ret}" in
            [Yy])
                exec_or_die rm -rf "${path}"
                ;;
            *)
                exit 0
                ;;
        esac
    fi
}

display_hq_resize_warning ()
{
    cat <<'EOF'

NOTE: If you would like to reproduce UVCGANv2 paper results with any
of the high-quality datasets (Celeba-HQ or AFHQ), please resize the
downloaded dataset with `scripts/downsize_right.py` script, like:

python scripts/downsize_right.py SOURCE TARGET -i lanczos -s 256 256

EOF
}

download_afhq ()
{
    local url="${URL_LIST["afhq"]}"
    local zip="afhq.zip"
    local path="${DATADIR}/afhq"

    check_dset_exists "${path}"
    download_and_extract_zip "${url}" "${zip}" "${CHECKSUMS[afhq]}"

    display_hq_resize_warning
}

download_sphenix ()
{
    local root="${DATADIR}/sphenix/"
    local path="${root}/2025-06-05_jet_bkg_sub"
    local archive="2025-06-05_jet_bkg_sub.tar"

    exec_or_die mkdir -p "${root}"

    check_dset_exists "${path}"
    download_archive  \
        "${URL_LIST[sphenix]}" "${archive}" "${CHECKSUMS[sphenix]}"

    echo "Unpacking archive to '${root}'"
    exec_or_die tar xvf "${DATADIR}/${archive}" -C "${root}"
}

dataset="${1}"

case "${dataset}" in
    afhq)
        download_afhq
        ;;
    sphenix)
        download_sphenix
        ;;
    help|-h|--help)
        usage
        ;;
    *)
        usage "Unknown dataset '${dataset}'"
esac

