# term7cv
This is used for SUTD 50.035 Computer Vision project, inventory management using SKU-110K dataset(dense object detection)

## Model Results:
| Model | Loss (after 5 epochs)  |
|-------|------------------------|
| VGG16 | 0.3025 | 
| VGG19 | 0.2936 |
| ResNet50 | 0.3069 |
| ResNet50V2 | 0.3187 |
| InceptionV3 | 0.3187 |
| InceptionResNetV2 | 0.3043 |
| Xception | 0.3187 |
| MobileNetV2 | 0.3187 |
| EfficientNetB2 | 0.2871 |

1. VGG16: 59s 4s/step - loss: 0.3025
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
Loss on validation data: 0.3025272488594055
```

2. ResNet50: 65s 4s/step - loss: 0.3069
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
Loss on validation data: 0.3068734407424927
```

3. YoloV8 0.1 of dataset, epoch=5
```
results_dict: {'metrics/precision(B)': 0.8195015171037628, 'metrics/recall(B)': 0.7019714058132889, 'metrics/mAP50(B)': 0.7884318516877316, 'metrics/mAP50-95(B)': 0.4450958072553458, 'fitness': 0.4794294116985844}
save_dir: PosixPath('runs/detect/train33')
speed: {'preprocess': 0.21475759045831086, 'inference': 5.162103422756853, 'loss': 0.0029062402659449085, 'postprocess': 11.559042437323209}
```

4. Xception: 68s 4s/step - loss: 0.3187
```
Epoch 1/5
189/189 - 994s 5s/step - loss: 0.3134 - val_loss: 0.3186
Epoch 2/5
189/189 - 986s 5s/step - loss: 0.3160 - val_loss: 0.3177
Epoch 3/5
189/189 - 991s 5s/step - loss: 0.3167 - val_loss: 0.3182
Epoch 4/5
189/189 - 985s 5s/step - loss: 0.3166 - val_loss: 0.3187
Epoch 5/5
189/189 - 984s 5s/step - loss: 0.3167 - val_loss: 0.3187
Loss on validation data for Xception: 0.31870660185813904
```

4. VGG19: 155s 10s/step - loss: 0.2936
```
Epoch 1/5
189/189 - 2219s 12s/step - loss: 0.3165 - val_loss: 0.3108
Epoch 2/5
189/189 - 2247s 12s/step - loss: 0.3045 - val_loss: 0.2936
Epoch 3/5
189/189 - 2202s 12s/step - loss: 0.2998 - val_loss: 0.2936
Epoch 4/5
189/189 - 2210s 12s/step - loss: 0.2969 - val_loss: 0.2936
Epoch 5/5
189/189 - 2203s 12s/step - loss: 0.2969 - val_loss: 0.2936
Loss on validation data for VGG19: 0.2935792803764343
```

5. ResNet50V2: 62s 4s/step - loss: 0.3187
```
Epoch 1/5
189/189 - 901s 5s/step - loss: 0.3214 - val_loss: 0.3179
Epoch 2/5
189/189 - 906s 5s/step - loss: 0.3184 - val_loss: 0.3187
Epoch 3/5
189/189 - 910s 5s/step - loss: 0.3177 - val_loss: 0.3187
Epoch 4/5
189/189 - 894s 5s/step - loss: 0.3170 - val_loss: 0.3187
Epoch 5/5
189/189 - 895s 5s/step - loss: 0.3166 - val_loss: 0.3187
Loss on validation data for ResNet50V2: 0.31870660185813904
```

6. InceptionV3: 52s 3s/step - loss: 0.3187
```
Epoch 1/5
189/189 - 705s 4s/step - loss: 0.3165 - val_loss: 0.3187
Epoch 2/5
189/189 - 715s 4s/step - loss: 0.3078 - val_loss: 0.3043
Epoch 3/5
189/189 - 718s 4s/step - loss: 0.3068 - val_loss: 0.3187
Epoch 4/5
189/189 - 720s 4s/step - loss: 0.3165 - val_loss: 0.3187
Epoch 5/5
189/189 - 733s 4s/step - loss: 0.3165 - val_loss: 0.3187
Loss on validation data for InceptionV3: 0.31870660185813904
```

7. InceptionResNetV2: 160s 11s/step - loss: 0.3043
```
Epoch 1/5
189/189 - 1218s 6s/step - loss: 0.3051 - val_loss: 0.3043
Epoch 2/5
189/189 - 1232s 7s/step - loss: 0.3039 - val_loss: 0.3043
Epoch 3/5
189/189 - 1241s 7s/step - loss: 0.3047 - val_loss: 0.3043
Epoch 4/5
189/189 - 1280s 7s/step - loss: 0.3046 - val_loss: 0.3043
Epoch 5/5
189/189 - 8975s 48s/step - loss: 0.3049 - val_loss: 0.3043
Loss on validation data for InceptionResNetV2: 0.3042828142642975
```

8. MobileNetV2: 46s 3s/step - loss: 0.3187
```
Epoch 1/5
189/189 - 854s 4s/step - loss: 0.3002 - val_loss: 0.3015
Epoch 2/5
189/189 - 849s 4s/step - loss: 0.3009 - val_loss: 0.3015
Epoch 3/5
189/189 - 694s 4s/step - loss: 0.3152 - val_loss: 0.3187
Epoch 4/5
189/189 - 694s 4s/step - loss: 0.3166 - val_loss: 0.3187
Epoch 5/5
189/189 - 700s 4s/step - loss: 0.3166 - val_loss: 0.3187
Loss on validation data for MobileNetV2: 0.31870660185813904
```

9. EfficientNetB2: 79s 5s/step - loss: 0.2871
```
Epoch 1/5
189/189 - 1068s 6s/step - loss: 0.2889 - val_loss: 0.2871
Epoch 2/5
189/189 - 1054s 6s/step - loss: 0.2877 - val_loss: 0.2871
Epoch 3/5
189/189 - 1083s 6s/step - loss: 0.2877 - val_loss: 0.2871
Epoch 4/5
189/189 - 1100s 6s/step - loss: 0.2877 - val_loss: 0.2871
Epoch 5/5
189/189 - 1122s 6s/step - loss: 0.2876 - val_loss: 0.2871
Loss on validation data for EfficientNetB2: 0.2870558798313141
```