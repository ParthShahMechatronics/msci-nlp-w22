# Assignment 4 submission
Trained a feedforward network for sentiment analysis. A batch size of 64 was used, Stop words were not removed and the model trained on 10 epochs.
## classification accuracy table
| Activation type | Dropout value | Accuracy(test set) |
| ------------- | ------------- | ------------- |
| sigmoid  | 0.3  | 0.663|
| ReLU  | 0.3  | 0.652|
| tanh  | 0.3  | 0.655 |
| sigmoid  | 0.25  | 0.663|
| ReLU  | 0.25  | 0.650|
| tanh  | 0.25  | 0.654 |
| sigmoid  | 0.2  | 0.662|
| ReLU  | 0.2  | 0.647|
| tanh  | 0.2  | 0.657 |

A Dropout rate of 0.25 was found to provide relatively accurate data
