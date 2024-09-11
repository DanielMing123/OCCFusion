# Training and Testing

## NuScenes 3D Semantic Occupancy Prediction Task
**a. Train OCCFusion with 8 GPUs.**
```shell
bash tools/dist_train.sh configs/OccFusion.py 8
```
**b. Test OCCFusion with 8 GPUs.**
```shell
bash tools/dist_test.sh configs/OccFusion.py ${path to corresponding checkpoint}$ 8
```
