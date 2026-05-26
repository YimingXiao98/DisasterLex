#!/usr/bin/env bash
#
# Download the DuckDB backend (disaster_ai_db.duckdb, ~4.7 GB) from Zenodo.
#
# The DuckDB binary is hosted separately from the code repository because of
# its size. It contains 36 H3-resolution-8 tables covering the state of Texas
# (~827,648 hexagonal cells per table) populated from FEMA NRI, HIFLD, US
# Census/ACS, and CDC/ATSDR SVI public-domain sources.
#
# Reviewer note (anonymous submission): the Zenodo deposit uses an anonymous
# reviewer-access link. After acceptance the deposit will be de-anonymized
# and assigned a DOI.
#
# Usage:
#     bash scripts/download_data.sh [--output-dir data]
#
set -euo pipefail

OUTPUT_DIR="${1:-data}"
ZENODO_URL="${ZENODO_URL:-PLACEHOLDER_ANONYMOUS_ZENODO_URL}"
EXPECTED_SHA256="${EXPECTED_SHA256:-PLACEHOLDER_SHA256_AFTER_UPLOAD}"

mkdir -p "${OUTPUT_DIR}"

if [[ "${ZENODO_URL}" == "PLACEHOLDER_ANONYMOUS_ZENODO_URL" ]]; then
    echo "ERROR: this script ships with a placeholder URL." >&2
    echo "Set ZENODO_URL to the anonymous Zenodo reviewer link before running." >&2
    exit 1
fi

echo "Downloading DuckDB from Zenodo (~4.7 GB) ..."
curl -L --fail -o "${OUTPUT_DIR}/disaster_ai_db.duckdb" "${ZENODO_URL}"

echo "Verifying SHA-256 ..."
COMPUTED=$(sha256sum "${OUTPUT_DIR}/disaster_ai_db.duckdb" | awk '{print $1}')
if [[ "${COMPUTED}" != "${EXPECTED_SHA256}" ]]; then
    echo "ERROR: SHA-256 mismatch." >&2
    echo "  expected: ${EXPECTED_SHA256}" >&2
    echo "  computed: ${COMPUTED}" >&2
    exit 1
fi

echo "Done. DuckDB available at ${OUTPUT_DIR}/disaster_ai_db.duckdb"
echo "Set DISASTER_DB_PATH=\${PWD}/${OUTPUT_DIR}/disaster_ai_db.duckdb in your .env"
