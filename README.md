# term7cv
This is used for SUTD 50.035 Computer Vision project, inventory management using SKU-110K dataset(dense object detection)

## Model Result:
1. VGG
```
Epoch 1/5
848s 4s/step - loss: 0.3038 - val_loss: 0.3025
Epoch 2/5
851s 5s/step - loss: 0.3052 - val_loss: 0.3025
Epoch 3/5
862s 5s/step - loss: 0.3052 - val_loss: 0.3025
Epoch 4/5
854s 5s/step - loss: 0.3052 - val_loss: 0.3025
Epoch 5/5
854s 5s/step - loss: 0.3051 - val_loss: 0.3025
59s 4s/step - loss: 0.3025
Loss on validation data: 0.3025272488594055
```

2. ResNet50
```
Epoch 1/5
866s 5s/step - loss: 0.3085 - val_loss: 0.3025
Epoch 2/5
818s 4s/step - loss: 0.3044 - val_loss: 0.3025
Epoch 3/5
834s 4s/step - loss: 0.3106 - val_loss: 0.3169
Epoch 4/5
835s 4s/step - loss: 0.3181 - val_loss: 0.3170
Epoch 5/5
835s 4s/step - loss: 0.3184 - val_loss: 0.3170
65s 4s/step - loss: 0.3069
Loss on validation data: 0.3068734407424927
```

3. YoloV8 0.1 of dataset, epoch=5
```
results_dict: {'metrics/precision(B)': 0.8195015171037628, 'metrics/recall(B)': 0.7019714058132889, 'metrics/mAP50(B)': 0.7884318516877316, 'metrics/mAP50-95(B)': 0.4450958072553458, 'fitness': 0.4794294116985844}
save_dir: PosixPath('runs/detect/train33')
speed: {'preprocess': 0.21475759045831086, 'inference': 5.162103422756853, 'loss': 0.0029062402659449085, 'postprocess': 11.559042437323209}
```
