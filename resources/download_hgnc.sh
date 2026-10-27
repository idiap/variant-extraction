# SPDX-FileCopyrightText: © <2026> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Andrei Blindu <andrei.blindu@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

#!/usr/bin/env bash

DATE="$1"
FILE="hgnc_complete_set_${DATE}.tsv"

wget "https://storage.googleapis.com/public-download-files/hgnc/archive/archive/monthly/tsv/${FILE}"
mv "${FILE}" hgnc_complete_set.tsv

# usage:
# ./download_hgnc.sh 2026-05-05