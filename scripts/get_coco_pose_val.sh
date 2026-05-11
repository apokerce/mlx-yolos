#!/usr/bin/env bash
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
#
# Ported from ultralytics/data/scripts/get_coco.sh — narrowed to the
# subset needed by mlx-yolos' pose evaluator: COCO val2017 images and
# the official person_keypoints annotations (which carry both bbox and
# keypoint labels for the ~2 346 person-keypointed val images;
# Ultralytics' yolo pose val evaluates both metrics against this file).
#
# Usage:
#   bash scripts/get_coco_pose_val.sh [target_dir]
#
# Default target_dir is ./datasets/coco-pose. After download:
#   <target_dir>/images/val2017/000000000139.jpg ...
#   <target_dir>/annotations/person_keypoints_val2017.json
#
# Skips work it has already done — safe to re-run.

set -euo pipefail

usage() {
  sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
  exit 0
}

case "${1:-}" in
  -h|--help) usage ;;
esac

TARGET="${1:-./datasets/coco-pose}"
mkdir -p "${TARGET}/images" "${TARGET}/annotations"

VAL_URL="http://images.cocodataset.org/zips/val2017.zip"           # ~1 GB
ANN_URL="http://images.cocodataset.org/annotations/annotations_trainval2017.zip"  # ~241 MB

cd "$(dirname "${TARGET}")"
TARGET="$(basename "${TARGET}")"

# --- val2017 images ---------------------------------------------------------
if [ ! -d "${TARGET}/images/val2017" ]; then
  echo "downloading ${VAL_URL}"
  curl -L "${VAL_URL}" -o val2017.zip --progress-bar
  echo "unzipping val2017.zip → ${TARGET}/images/"
  unzip -q val2017.zip -d "${TARGET}/images/"
  rm val2017.zip
else
  echo "found ${TARGET}/images/val2017 — skipping image download"
fi

# --- annotations ------------------------------------------------------------
need_ann=0
for f in person_keypoints_val2017.json; do
  if [ ! -f "${TARGET}/annotations/${f}" ]; then
    need_ann=1
    break
  fi
done

if [ "${need_ann}" -eq 1 ]; then
  echo "downloading ${ANN_URL}"
  curl -L "${ANN_URL}" -o annotations.zip --progress-bar
  echo "unzipping annotations.zip → ${TARGET}/"
  # The zip puts files in annotations/ at the archive root, so this lands
  # them at <target>/annotations/<json>.
  unzip -q -o annotations.zip -d "${TARGET}/"
  # Trim every annotation file we don't need to keep the on-disk footprint
  # small. We only use person_keypoints_val2017.json — that's the GT for
  # both bbox and keypoint mAP in the pose-val pipeline.
  rm -f "${TARGET}/annotations/person_keypoints_train2017.json"
  rm -f "${TARGET}/annotations/instances_train2017.json"
  rm -f "${TARGET}/annotations/instances_val2017.json"
  rm -f "${TARGET}/annotations/captions_train2017.json"
  rm -f "${TARGET}/annotations/captions_val2017.json"
  rm annotations.zip
else
  echo "found ${TARGET}/annotations/person_keypoints_val2017.json — skipping"
fi

echo
echo "ready:"
ls -lh "${TARGET}/annotations/" | awk '{print "  "$0}'
echo "  $(ls "${TARGET}/images/val2017" | wc -l) val2017 images"
