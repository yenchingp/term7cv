# term7cv
This is used for SUTD 50.035 Computer Vision project, inventory management using SKU-110K dataset(dense object detection)

## Model Results:
| Model | Loss (after 5 epochs) | IoU | Time |
|-------|-----------------------|-----|------|
| VGG16 | 0.7038 | 0.4975 | 5180s |
| ResNet50 | 0.7045 | 0.4975 | 1600s |
| Xception | 7.3319 | 0.1314 | 1500s |
| VGG19 | 7.8031 | 0.4975 | 2100s |
| ResNet50V2 | 0.7234 | 0.4968 | 1150s | 
| InceptionV3 | 7.5169 | 0.3852 | 715s |
| InceptionResNetV2 | 0.7073 | 0.4969 | 1200s |
| MobileNetV2 | 0.6951 | 0.4975 | 600s |
| EfficientNetB2 | 0.6948 | 0.4975 | 890s |

1. VGG16: 428s 28s/step - loss: 0.7038 - auc_4: 0.5007 - precision_4: 0.9955 - recall_4: 0.5491
Average IoU for VGG16: 0.4975138008594513
```
Epoch 1/5
5188s 27s/step - loss: 0.8078 - auc_4: 0.5723 - precision_4: 0.9975 - recall_4: 0.5116 - val_loss: 0.7592 - val_auc_4: 0.5737 - val_precision_4: 0.9962 - val_recall_4: 0.7206
Epoch 2/5
5479s 29s/step - loss: 0.7014 - auc_4: 0.6364 - precision_4: 0.9977 - recall_4: 0.5241 - val_loss: 0.6966 - val_auc_4: 0.5162 - val_precision_4: 0.9961 - val_recall_4: 0.6335
Epoch 3/5
5463s 29s/step - loss: 0.6828 - auc_4: 0.7530 - precision_4: 0.9988 - recall_4: 0.5283 - val_loss: 0.6973 - val_auc_4: 0.5595 - val_precision_4: 0.9962 - val_recall_4: 0.5763
Epoch 4/5
5627s 30s/step - loss: 0.6748 - auc_4: 0.7464 - precision_4: 0.9988 - recall_4: 0.5233 - val_loss: 0.7015 - val_auc_4: 0.5492 - val_precision_4: 0.9965 - val_recall_4: 0.6376
Epoch 5/5
6292s 33s/step - loss: 0.6697 - auc_4: 0.7785 - precision_4: 0.9991 - recall_4: 0.5214 - val_loss: 0.7038 - val_auc_4: 0.5007 - val_precision_4: 0.9955 - val_recall_4: 0.5491
```

2. ResNet50: 98s 6s/step - loss: 0.7045 - auc_5: 0.7644 - precision_5: 0.9993 - recall_5: 0.3965
Average IoU for ResNet50: 0.4975138008594513
```
Epoch 1/5
1663s 9s/step - loss: 0.7921 - auc_5: 0.5409 - precision_5: 0.9968 - recall_5: 0.5126 - val_loss: 0.7019 - val_auc_5: 0.7187 - val_precision_5: 0.9981 - val_recall_5: 0.5821
Epoch 2/5
1423s 8s/step - loss: 0.7015 - auc_5: 0.6454 - precision_5: 0.9979 - recall_5: 0.5257 - val_loss: 0.6930 - val_auc_5: 0.6769 - val_precision_5: 0.9982 - val_recall_5: 0.5993
Epoch 3/5
1678s 9s/step - loss: 0.6824 - auc_5: 0.7285 - precision_5: 0.9987 - recall_5: 0.5247 - val_loss: 0.6941 - val_auc_5: 0.7128 - val_precision_5: 0.9984 - val_recall_5: 0.5119
Epoch 4/5
1815s 10s/step - loss: 0.6734 - auc_5: 0.7581 - precision_5: 0.9988 - recall_5: 0.5215 - val_loss: 0.6964 - val_auc_5: 0.6578 - val_precision_5: 0.9980 - val_recall_5: 0.5505
Epoch 5/5
1602s 8s/step - loss: 0.6672 - auc_5: 0.7799 - precision_5: 0.9990 - recall_5: 0.5208 - val_loss: 0.7045 - val_auc_5: 0.7644 - val_precision_5: 0.9993 - val_recall_5: 0.3965
```

3. Xception: 120s 8s/step - loss: 7.3319 - auc: 0.4347 - precision: 0.9948 - recall: 0.7445
Average IoU for Xception: 0.1314603090286255
```
Epoch 1/5
1757s 9s/step - loss: 7.3449 - auc: 0.4481 - precision: 0.9961 - recall: 0.7524 - val_loss: 7.3358 - val_auc: 0.4372 - val_precision: 0.9948 - val_recall: 0.7494
Epoch 2/5
1703s 9s/step - loss: 7.3613 - auc: 0.4533 - precision: 0.9961 - recall: 0.7499 - val_loss: 7.3358 - val_auc: 0.4372 - val_precision: 0.9948 - val_recall: 0.7494
Epoch 3/5
1708s 9s/step - loss: 7.3611 - auc: 0.4532 - precision: 0.9961 - recall: 0.7498 - val_loss: 7.3358 - val_auc: 0.4372 - val_precision: 0.9948 - val_recall: 0.7494
Epoch 4/5
1842s 10s/step - loss: 7.3619 - auc: 0.4533 - precision: 0.9961 - recall: 0.7499 - val_loss: 7.3358 - val_auc: 0.4372 - val_precision: 0.9948 - val_recall: 0.7494
Epoch 5/5
1720s 9s/step - loss: 7.3822 - auc: 0.4591 - precision: 0.9962 - recall: 0.7076 - val_loss: 7.3319 - val_auc: 0.4347 - val_precision: 0.9948 - val_recall: 0.7445
Loss on validation data for Xception: [7.331908702850342, 0.4347253143787384, 0.9948090314865112, 0.7444506287574768]
```

4. VGG19: 152s 10s/step - loss: 7.8031 - auc_1: 0.5000 - precision_1: 0.0000e+00 - recall_1: 0.0000e+00
Average IoU for VGG19: 0.4975138008594513
```
Epoch 1/5
2475s 13s/step - loss: 7.4619 - auc_1: 0.5114 - precision_1: 0.9969 - recall_1: 0.3466 - val_loss: 7.5439 - val_auc_1: 0.6256 - val_precision_1: 1.0000 - val_recall_1: 0.2511
Epoch 2/5
2164s 11s/step - loss: 7.5849 - auc_1: 0.5755 - precision_1: 0.9995 - recall_1: 0.1748 - val_loss: 7.8031 - val_auc_1: 0.5000 - val_precision_1: 0.0000e+00 - val_recall_1: 0.0000e+00
Epoch 3/5
2161s 11s/step - loss: 7.7915 - auc_1: 0.4998 - precision_1: 0.9962 - recall_1: 0.0055 - val_loss: 7.8031 - val_auc_1: 0.5000 - val_precision_1: 0.0000e+00 - val_recall_1: 0.0000e+00
Epoch 4/5
2162s 11s/step - loss: 7.7953 - auc_1: 0.5002 - precision_1: 1.0000 - recall_1: 3.3357e-04 - val_loss: 7.8031 - val_auc_1: 0.5000 - val_precision_1: 0.0000e+00 - val_recall_1: 0.0000e+00
Epoch 5/5
2164s 11s/step - loss: 7.7956 - auc_1: 0.5004 - precision_1: 1.0000 - recall_1: 8.1308e-04 - val_loss: 7.8031 - val_auc_1: 0.5000 - val_precision_1: 0.0000e+00 - val_recall_1: 0.0000e+00
```

5. ResNet50V2: 81s 5s/step - loss: 0.7234 - auc_3: 0.5666 - precision_3: 0.9968 - recall_3: 0.8610
Average IoU for ResNet50V2: 0.49682319164276123
```
Epoch 1/5
1252s 7s/step - loss: 0.7590 - auc_3: 0.5535 - precision_3: 0.9970 - recall_3: 0.5128 - val_loss: 0.7040 - val_auc_3: 0.5506 - val_precision_3: 0.9962 - val_recall_3: 0.5868
Epoch 2/5
1138s 6s/step - loss: 0.7023 - auc_3: 0.6435 - precision_3: 0.9981 - recall_3: 0.5260 - val_loss: 0.7004 - val_auc_3: 0.6032 - val_precision_3: 0.9963 - val_recall_3: 0.7425
Epoch 3/5
1139s 6s/step - loss: 0.6904 - auc_3: 0.6977 - precision_3: 0.9986 - recall_3: 0.5347 - val_loss: 0.7094 - val_auc_3: 0.6031 - val_precision_3: 0.9968 - val_recall_3: 0.6934
Epoch 4/5
1146s 6s/step - loss: 0.6877 - auc_3: 0.6920 - precision_3: 0.9983 - recall_3: 0.5380 - val_loss: 0.7164 - val_auc_3: 0.6470 - val_precision_3: 0.9971 - val_recall_3: 0.7569
Epoch 5/5
1143s 6s/step - loss: 0.6852 - auc_3: 0.7306 - precision_3: 0.9986 - recall_3: 0.5381 - val_loss: 0.7234 - val_auc_3: 0.5666 - val_precision_3: 0.9968 - val_recall_3: 0.8610
```

6. InceptionV3: 50s 3s/step - loss: 7.5169 - auc_3: 0.6256 - precision_3: 1.0000 - recall_3: 0.2511
Average IoU for InceptionV3: 0.3852076530456543
```
Epoch 1/5
725s 4s/step - loss: 7.2585 - auc_3: 0.7418 - precision_3: 0.9997 - recall_3: 0.5255 - val_loss: 7.2577 - val_auc_3: 0.7511 - val_precision_3: 1.0000 - val_recall_3: 0.5022
Epoch 2/5
715s 4s/step - loss: 7.2433 - auc_3: 0.7508 - precision_3: 1.0000 - recall_3: 0.5016 - val_loss: 7.2577 - val_auc_3: 0.7511 - val_precision_3: 1.0000 - val_recall_3: 0.5022
Epoch 3/5
725s 4s/step - loss: 7.2510 - auc_3: 0.7406 - precision_3: 0.9998 - recall_3: 0.5113 - val_loss: 7.2577 - val_auc_3: 0.7511 - val_precision_3: 1.0000 - val_recall_3: 0.5022
Epoch 4/5
724s 4s/step - loss: 7.2560 - auc_3: 0.7283 - precision_3: 0.9996 - recall_3: 0.5168 - val_loss: 7.2577 - val_auc_3: 0.7511 - val_precision_3: 1.0000 - val_recall_3: 0.5022
Epoch 5/5
718s 4s/step - loss: 7.3290 - auc_3: 0.7135 - precision_3: 1.0000 - recall_3: 0.4270 - val_loss: 7.5169 - val_auc_3: 0.6256 - val_precision_3: 1.0000 - val_recall_3: 0.2511
Loss on validation data for InceptionV3: [7.516930103302002, 0.6255549192428589, 1.0, 0.25110986828804016]
```

7. InceptionResNetV2: 86s 6s/step - loss: 0.7073 - auc: 0.7009 - precision: 0.9995 - recall: 0.5602
Average IoU for InceptionResNetV2: 0.4969613254070282
```
Epoch 1/5
1253s 7s/step - loss: 0.7411 - auc: 0.5867 - precision: 0.9975 - recall: 0.5277 - val_loss: 0.7234 - val_auc: 0.7135 - val_precision: 1.0000 - val_recall: 0.0300
Epoch 2/5
1225s 6s/step - loss: 0.7050 - auc: 0.5931 - precision: 0.9975 - recall: 0.5359 - val_loss: 0.7149 - val_auc: 0.7772 - val_precision: 1.0000 - val_recall: 0.1579
Epoch 3/5
1217s 6s/step - loss: 0.6956 - auc: 0.6659 - precision: 0.9984 - recall: 0.5454 - val_loss: 0.7075 - val_auc: 0.7914 - val_precision: 1.0000 - val_recall: 0.5425
Epoch 4/5
1224s 6s/step - loss: 0.6926 - auc: 0.7061 - precision: 0.9992 - recall: 0.5480 - val_loss: 0.7078 - val_auc: 0.7312 - val_precision: 1.0000 - val_recall: 0.4856
Epoch 5/5
1226s 6s/step - loss: 0.6922 - auc: 0.7258 - precision: 0.9992 - recall: 0.5444 - val_loss: 0.7073 - val_auc: 0.7009 - val_precision: 0.9995 - val_recall: 0.5602
```

8. MobileNetV2: 41s 3s/step - loss: 0.6951 - auc_1: 0.6144 - precision_1: 0.9974 - recall_1: 0.5316
Average IoU for MobileNetV2: 0.4975138008594513
```
Epoch 1/5
630s 3s/step - loss: 0.7683 - auc_1: 0.5554 - precision_1: 0.9972 - recall_1: 0.5131 - val_loss: 0.6981 - val_auc_1: 0.5574 - val_precision_1: 0.9966 - val_recall_1: 0.6421
Epoch 2/5
596s 3s/step - loss: 0.7000 - auc_1: 0.6625 - precision_1: 0.9983 - recall_1: 0.5223 - val_loss: 0.6931 - val_auc_1: 0.5431 - val_precision_1: 0.9967 - val_recall_1: 0.6676
Epoch 3/5
592s 3s/step - loss: 0.6872 - auc_1: 0.7092 - precision_1: 0.9985 - recall_1: 0.5313 - val_loss: 0.6911 - val_auc_1: 0.6027 - val_precision_1: 0.9964 - val_recall_1: 0.5441
Epoch 4/5
587s 3s/step - loss: 0.6815 - auc_1: 0.7367 - precision_1: 0.9988 - recall_1: 0.5315 - val_loss: 0.6931 - val_auc_1: 0.5119 - val_precision_1: 0.9954 - val_recall_1: 0.4853
Epoch 5/5
591s 3s/step - loss: 0.6774 - auc_1: 0.7652 - precision_1: 0.9991 - recall_1: 0.5263 - val_loss: 0.6951 - val_auc_1: 0.6144 - val_precision_1: 0.9974 - val_recall_1: 0.5316
```

9. EfficientNetB2: 62s 4s/step - loss: 0.6948 - auc_2: 0.6200 - precision_2: 0.9968 - recall_2: 0.5197
Average IoU for EfficientNetB2: 0.4975138008594513
```
Epoch 1/5
900s 5s/step - loss: 0.7810 - auc_2: 0.5378 - precision_2: 0.9971 - recall_2: 0.5134 - val_loss: 0.7082 - val_auc_2: 0.6278 - val_precision_2: 0.9973 - val_recall_2: 0.6207
Epoch 2/5
893s 5s/step - loss: 0.7026 - auc_2: 0.6385 - precision_2: 0.9979 - recall_2: 0.5201 - val_loss: 0.7160 - val_auc_2: 0.5550 - val_precision_2: 0.9963 - val_recall_2: 0.6004
Epoch 3/5
891s 5s/step - loss: 0.6885 - auc_2: 0.6680 - precision_2: 0.9981 - recall_2: 0.5354 - val_loss: 0.6925 - val_auc_2: 0.6012 - val_precision_2: 0.9975 - val_recall_2: 0.5486
Epoch 4/5
892s 5s/step - loss: 0.6840 - auc_2: 0.7188 - precision_2: 0.9985 - recall_2: 0.5322 - val_loss: 0.7545 - val_auc_2: 0.7090 - val_precision_2: 0.9988 - val_recall_2: 0.4589
Epoch 5/5
894s 5s/step - loss: 0.6801 - auc_2: 0.7280 - precision_2: 0.9985 - recall_2: 0.5283 - val_loss: 0.6948 - val_auc_2: 0.6200 - val_precision_2: 0.9968 - val_recall_2: 0.5197
```