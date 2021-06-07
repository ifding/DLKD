# sample scripts for running the distillation code
# use resnet32x4 and resnet8x4 as an example

#python train_student.py --path_t ./save/models/vgg13_vanilla/vgg13.pth --distill mlkd --model_s vgg8 -r 1 -a 50 -b 1 -d 20 --trial 1

#python train_student.py --path_t ./save/models/wrn_40_2_vanilla/wrn_40_2.pth --distill mlkd --model_s wrn_40_1 -r 1 -a 50 -b 1 -d 20 --trial 1

#python train_student.py --path_t ./save/models/wrn_40_2_vanilla/wrn_40_2.pth --distill mlkd --model_s wrn_16_2 -r 1 -a 50 -b 1 -d 20 --trial 1

#python train_student.py --path_t ./save/models/resnet32x4_vanilla/resnet32x4.pth --distill mlkd --model_s resnet8x4 -r 1 -a 50 -b 1 -d 20 --trial 1

#python train_student.py --path_t ./save/models/resnet56_vanilla/resnet56.pth --distill mlkd --model_s resnet20 -r 1 -a 50 -b 1 -d 20 --trial 1


python train_student.py --path_t ./save/models/resnet32x4_vanilla/resnet32x4.pth --distill mlkd --model_s ShuffleV2 -r 1 -a 50 -b 1 -d 20 --trial 1

#python train_student.py --path_t ./save/models/ResNet50_vanilla/ResNet50.pth --distill mlkd --model_s MobileNetV2 -r 1 -a 50 -b 1 -d 20 --trial 1

#python train_student.py --path_t ./save/models/vgg13_vanilla/vgg13.pth --distill mlkd --model_s MobileNetV2 -r 1 -a 50 -b 1 -d 20 --trial 1

#python train_student.py --path_t ./save/models/wrn_40_2_vanilla/wrn_40_2.pth --distill mlkd --model_s ShuffleV1 -r 1 -a 50 -b 1 -d 20 --trial 1

#python train_student.py --path_t ./save/models/resnet32x4_vanilla/resnet32x4.pth --distill mlkd --model_s ShuffleV1 -r 1 -a 50 -b 1 -d 20 --trial 1

#python train_student.py --path_t ./save/models/ResNet50_vanilla/ResNet50.pth --distill mlkd --model_s vgg8 -r 1 -a 50 -b 1 -d 20 --trial 1


