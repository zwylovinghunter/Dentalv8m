#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/yolov8

RUN_NAME="exp02_stage2_yolov8m_spdconv2_coordatt_p3p4_from_stage1"
RESULT_DIR="/root/autodl-tmp/yolov8/${RUN_NAME}"
LOG_PATH="/root/autodl-tmp/yolov8/logs/${RUN_NAME}.log"

if [ -e "${RESULT_DIR}" ]; then
  echo "Result directory already exists, refusing to overwrite: ${RESULT_DIR}" >&2
  exit 1
fi

mkdir -p logs
screen -dmS "${RUN_NAME}" bash -lc "cd /root/autodl-tmp/yolov8 && python scripts/train_ablation.py --model configs/yolov8m-spdcoordatt-dental.yaml --weights /root/autodl-tmp/yolov8/yolov8m.pt --data data/dental_lesion_3cls_large.yaml --epochs 100 --imgsz 640 --batch 16 --device 0 --workers 8 --project /root/autodl-tmp/yolov8 --name ${RUN_NAME} --seed 42 --patience 100 --optimizer auto --cos-lr False --amp True --cache False --close-mosaic 10 > ${LOG_PATH} 2>&1"
echo "started ${RUN_NAME}"
