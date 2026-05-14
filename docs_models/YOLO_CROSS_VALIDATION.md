- 5-fold cross-validation strategy.
- Trainging (train2) data will be randomly shuffled and divided into five folds, with each fold serving once as the validation set during training and hyperparameter tuning.
- 5 or more different YOLO models will be trained on the training folds and evaluated on the validation fold.
- To compute/predict the final pose class on test data, we select the class label with the highest average output probabilities among the class probabilities cast by a model on test images.

- Reduce the padding size and somehow adapt it based on the visibility of the animal in focus.

- Enable class weighting - cls_pw
- Enable dropout