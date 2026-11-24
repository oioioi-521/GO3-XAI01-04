# Fixed-split method selection results

Native gap columns belong to within-task cells only.

## Held-out comparison

| profile | objective | policy | scenarios | hit_rate | mean_rank_regret |
|---|---|---|---|---|---|
| full | efficiency | goal_mode | 574 | 1.000000 | 0.000000 |
| full | efficiency | goal_rank | 574 | 1.000000 | 0.000000 |
| full | efficiency | tree | 574 | 1.000000 | 0.000000 |
| full | faithfulness | goal_mode | 574 | 0.585366 | 0.220557 |
| full | faithfulness | goal_rank | 574 | 0.585366 | 0.220557 |
| full | faithfulness | tree | 574 | 0.585366 | 0.220557 |
| full | stability | goal_mode | 574 | 0.768293 | 0.055749 |
| full | stability | goal_rank | 574 | 0.768293 | 0.055749 |
| full | stability | tree | 574 | 0.768293 | 0.055749 |
| no_grad_logits | efficiency | goal_mode | 574 | 1.000000 | 0.000000 |
| no_grad_logits | efficiency | goal_rank | 574 | 1.000000 | 0.000000 |
| no_grad_logits | efficiency | tree | 574 | 1.000000 | 0.000000 |
| no_grad_logits | faithfulness | goal_mode | 574 | 0.343206 | 0.342044 |
| no_grad_logits | faithfulness | goal_rank | 574 | 0.343206 | 0.342044 |
| no_grad_logits | faithfulness | tree | 574 | 0.400697 | 0.350755 |
| no_grad_logits | stability | goal_mode | 574 | 0.898955 | 0.040070 |
| no_grad_logits | stability | goal_rank | 574 | 0.898955 | 0.040070 |
| no_grad_logits | stability | tree | 574 | 0.898955 | 0.040070 |

## Conditional recommendations

| profile | dataset | model | objective | recommended | candidate_methods | retain_multiple |
|---|---|---|---|---|---|---|
| full | imagenet | densenet121 | efficiency | gradcam | gradcam | False |
| full | imagenet | densenet121 | faithfulness | ig | ig / occlusion / kernelshap | True |
| full | imagenet | densenet121 | stability | rise | gradcam / rise / kernelshap | True |
| full | imagenet | resnet50 | efficiency | gradcam | gradcam | False |
| full | imagenet | resnet50 | faithfulness | ig | ig / kernelshap | True |
| full | imagenet | resnet50 | stability | rise | gradcam / rise / kernelshap | True |
| full | imagenet | vgg16 | efficiency | gradcam | gradcam | False |
| full | imagenet | vgg16 | faithfulness | ig | ig / kernelshap | True |
| full | imagenet | vgg16 | stability | rise | gradcam / rise / kernelshap | True |
| full | voc | densenet121 | efficiency | gradcam | gradcam | False |
| full | voc | densenet121 | faithfulness | ig | ig / occlusion / kernelshap | True |
| full | voc | densenet121 | stability | rise | rise | False |
| full | voc | resnet50 | efficiency | gradcam | gradcam | False |
| full | voc | resnet50 | faithfulness | ig | ig / kernelshap | True |
| full | voc | resnet50 | stability | rise | rise | False |
| full | voc | vgg16 | efficiency | gradcam | gradcam | False |
| full | voc | vgg16 | faithfulness | ig | ig / kernelshap | True |
| full | voc | vgg16 | stability | rise | rise | False |
| no_grad_logits | imagenet | densenet121 | efficiency | occlusion | occlusion | False |
| no_grad_logits | imagenet | densenet121 | faithfulness | lime | rise / kernelshap / lime | True |
| no_grad_logits | imagenet | densenet121 | stability | rise | rise / kernelshap | True |
| no_grad_logits | imagenet | resnet50 | efficiency | occlusion | occlusion | False |
| no_grad_logits | imagenet | resnet50 | faithfulness | lime | rise / kernelshap / lime | True |
| no_grad_logits | imagenet | resnet50 | stability | rise | rise / kernelshap | True |
| no_grad_logits | imagenet | vgg16 | efficiency | occlusion | occlusion | False |
| no_grad_logits | imagenet | vgg16 | faithfulness | kernelshap | occlusion / rise / kernelshap / lime | True |
| no_grad_logits | imagenet | vgg16 | stability | rise | rise / kernelshap | True |
| no_grad_logits | voc | densenet121 | efficiency | occlusion | occlusion | False |
| no_grad_logits | voc | densenet121 | faithfulness | lime | rise / kernelshap / lime | True |
| no_grad_logits | voc | densenet121 | stability | rise | rise | False |
| no_grad_logits | voc | resnet50 | efficiency | occlusion | occlusion | False |
| no_grad_logits | voc | resnet50 | faithfulness | lime | rise / kernelshap / lime | True |
| no_grad_logits | voc | resnet50 | stability | rise | rise | False |
| no_grad_logits | voc | vgg16 | efficiency | occlusion | occlusion | False |
| no_grad_logits | voc | vgg16 | faithfulness | kernelshap | occlusion / rise / kernelshap / lime | True |
| no_grad_logits | voc | vgg16 | stability | rise | rise | False |

## Leaf support

| profile | leaf | recommended | train_scenarios | train_original_images | test_scenarios | train_modal_share | candidate_methods |
|---|---|---|---|---|---|---|---|
| full | 3 | ig | 1170 | 626 | 388 | 0.694017 | ig / kernelshap |
| full | 4 | ig | 573 | 573 | 186 | 0.521815 | ig / occlusion / kernelshap |
| full | 6 | rise | 873 | 323 | 269 | 0.624284 | gradcam / rise / kernelshap |
| full | 7 | rise | 870 | 316 | 305 | 0.925287 | rise |
| full | 8 | gradcam | 1743 | 639 | 574 | 1.000000 | gradcam |
| no_grad_logits | 3 | lime | 1173 | 626 | 381 | 0.359761 | rise / kernelshap / lime |
| no_grad_logits | 4 | kernelshap | 570 | 570 | 193 | 0.417544 | occlusion / rise / kernelshap / lime |
| no_grad_logits | 6 | rise | 873 | 323 | 269 | 0.835052 | rise / kernelshap |
| no_grad_logits | 7 | rise | 870 | 316 | 305 | 0.972414 | rise |
| no_grad_logits | 8 | occlusion | 1743 | 639 | 574 | 1.000000 | occlusion |
