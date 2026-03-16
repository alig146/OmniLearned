#!/bin/bash
# submit_grid.sh — Submit prepare_data_v2.py jobs to the CERN ATLAS grid via prun.
#
# Prerequisites (run once per session):
#   setupATLAS
#   lsetup panda rucio
#   voms-proxy-init --voms atlas
#
# Usage:
#   bash submit_grid.sh
#
# Each prun call can process one or many input files per job (--nFilesPerJob).
# Outputs are collected in per-sample Rucio output datasets under your account.
# After all jobs finish, download with rucio and run scripts/merge_h5.py.

set -euo pipefail

# -----------------------------------------------------------------------
# Configuration — edit these
# -----------------------------------------------------------------------
RUCIO_ACCOUNT="${RUCIO_ACCOUNT:-mcochran}"    # your ATLAS grid username
VERSION="v03"                                  # bump when reprocessing
NFILES_PER_JOB="${NFILES_PER_JOB:-5}"          # set >1 to process multiple ROOT files per grid job
CHUNK_SIZE="${CHUNK_SIZE:-500 MB}"             # passed to prepare_data_v2.py --chunk_size
PRUN_MEMORY="${PRUN_MEMORY:-2000}"             # MB per core requested from PanDA brokerage
USE_FORCE_STAGED="${USE_FORCE_STAGED:-0}"      # 1=force local staging, 0=allow remote/direct access
PRUN_SITE="${PRUN_SITE:-}"                     # optional fixed site/queue, e.g. DESY-ZN/SCORE
PRUN_EXCLUDED_SITE="${PRUN_EXCLUDED_SITE:-}"   # optional excluded site pattern

# Grid workers have CVMFS mounted, so we source the same 'scikit recommended'
# stack directly in --exec rather than maintaining a custom container image.
# The ALRB setup path is standard on all WLCG worker nodes.
ATLAS_LOCAL_ROOT_BASE="/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase"
LCG_BASE="/cvmfs/sft.cern.ch/lcg/views/LCG_104/x86_64-el9-gcc13-opt/setup.sh"
SCIKIT_SETUP="source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh --quiet && source ${LCG_BASE}"

# Input Rucio dataset names
DS_JZ2="user.nkyriaco.JZ2.Ntuple_03_03_26_Prod1_EXT0"
DS_TAU="user.nkyriaco.Gammatautau.Ntuple_03_03_26_Prod1_EXT0"
DS_ELE="user.nkyriaco.Gammaee.Ntuple_03_03_26_Prod1_EXT0"

# -----------------------------------------------------------------------
# Helper: build the prun command for one sample
# -----------------------------------------------------------------------
# Arguments: inDS  label  outDS_suffix
submit_sample() {
    local inDS="$1"
    local label="$2"
    local suffix="$3"
    local outDS="user.${RUCIO_ACCOUNT}.${suffix}.${VERSION}/"
    local force_staged_opt=""
    local site_opt=""
    local excluded_site_opt=""

    if [[ "${USE_FORCE_STAGED}" == "1" ]]; then
        force_staged_opt="--forceStaged"
    fi
    if [[ -n "${PRUN_SITE}" ]]; then
        site_opt="--site ${PRUN_SITE}"
    fi
    if [[ -n "${PRUN_EXCLUDED_SITE}" ]]; then
        excluded_site_opt="--excludedSite ${PRUN_EXCLUDED_SITE}"
    fi

    echo "Submitting: ${inDS}  (label=${label})  ->  ${outDS}"

    prun \
        --exec "${SCIKIT_SETUP} && for INFILE in \$(echo %IN | tr ',' ' '); do python prepare_data_v2.py --input_file \"\${INFILE}\" --label ${label} --chunk_size \"${CHUNK_SIZE}\" --output_file output_%RNDM8_\$(basename \"\${INFILE}\" .root).h5; done" \
        --inDS "${inDS}" \
        --outDS "${outDS}" \
        --outputs "*.h5" \
        --memory "${PRUN_MEMORY}" \
        --nFilesPerJob "${NFILES_PER_JOB}" \
        --extFile "prepare_data_v2.py" \
        ${force_staged_opt} \
        ${site_opt} \
        ${excluded_site_opt} \
        --noBuild
}

# -----------------------------------------------------------------------
# Submit one job array per sample type
# -----------------------------------------------------------------------
# submit_sample "${DS_JZ2}" 0 "OmniTau_JZ2_h5"
submit_sample "${DS_TAU}" 1 "OmniTau_Tautau_h5"
# submit_sample "${DS_ELE}" 2 "OmniTau_Gammaee_h5"

echo ""
echo "All submissions sent. Monitor with:"
echo "  pbook"
echo "  pbook> show()"
echo ""
echo "Once complete, download outputs:"
echo "  rucio download user.${RUCIO_ACCOUNT}.OmniTau_JZ2_h5.${VERSION}/"
echo "  rucio download user.${RUCIO_ACCOUNT}.OmniTau_Tautau_h5.${VERSION}/"
echo "  rucio download user.${RUCIO_ACCOUNT}.OmniTau_Gammaee_h5.${VERSION}/"
echo ""
echo "Then merge:"
echo "  python scripts/merge_h5.py --raw_dir <download_dir> --output_dir <training_data_dir>"
