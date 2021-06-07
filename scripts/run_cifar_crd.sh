# sample scripts for running the distillation code
# use resnet32x4 and resnet8x4 as an example

# crd
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill crd --model_s resnet8x4 -a 0 -b 0.8 --trial 1 

