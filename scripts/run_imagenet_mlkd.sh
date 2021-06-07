# sample scripts for running the distillation code
# use resnet32x4 and resnet8x4 as an example

#python train_student_imagenet.py --path_t ./save/models/resnet34-333f7ec4.pth --resume ./model_best.pth.tar --batch-size 128 --workers 3 -r 1 -a 50 -b 1 -d 20

python train_student_imagenet.py --path_t ./save/models/resnet34-333f7ec4.pth --batch-size 256 --workers 12 -r 1 -a 20 -b 0.5 -d 10



