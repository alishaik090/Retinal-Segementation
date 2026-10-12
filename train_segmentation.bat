@echo off
setlocal
cd /d "%~dp0"
python -m retinal_ai.train_segmentation --architecture resnet18_unet --pretrained-encoder --epochs 100 --image-size 256 --batch-size 2
python -m retinal_ai.report_segmentation

