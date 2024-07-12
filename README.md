# Secure_FL

The codes above were implemented from scratch.

- `FedAvg.py` : FedAvg is the most basic and widely used aggregation algorithm in Federated Learning, which involves a simple averaging of the local model updates from participating clients to create a new global model. Note that the relevant experiments were conducted under highly non-IID (non-Independent and Identically Distributed) data. Here, taking [Adult Census Income Dataset](https://www.kaggle.com/datasets/uciml/adult-census-income) as an example to create non-IID partitions with 8 groups based on the 'income' feature, where 2 devices only hold label "0", 2 devices only hold label "1", and the remaining devices (4 devices) have both labels "0" and "1". This can be considered as label distribution skew. In addition, the number of samples held by the devices varies (i.e., quantity skew or unbalancedness). The experimental results without data poisoning attack are as follows:

  ![The global model accuracy of FedAvg algorithm.](https://github.com/jjzgeeks/Secure_FL/blob/main/readme_pics/FedAvg_global_model_accuracy_30_8.png)
   ![The train loss and test loss of FedAvg algorithm.](https://github.com/jjzgeeks/Secure_FL/blob/main/readme_pics/FedAvg_loss_30_8.png)


- `data_poisoning_FL.py`: 2 devices only hold label "0" intentionally modify the labels associated with training data points, i.e., They both modify 60% of the real training data with label "0" to label "1". The goal is to mislead the model during training, causing it to learn incorrect or biased associations. Byzantine-resilient secure aggregation methods, i.e., Trimmed-mean, Krum, Multi-Krum and Median, are implemented. The experimental results are as follows:
    ![The results of secure aggregation methods.](https://github.com/jjzgeeks/Secure_FL/blob/main/readme_pics/Secure_aggregation_results.png)

| Aggregation methods | Precision | F1-socre | AUC | MCC |
| :---: |:---: |:---: |:---: |:---: |
| FedAvg | $0.233 \pm 0.001$ | $0.378 \pm 0.003$ |  $0.499 \pm 0.003$ |  $-0.005 \pm 0.015$ | 
| Krum |  $0.008 \pm 0.043$ |  $0.013 \pm 0.069$ |   $0.50 \pm 0.0$ |   $0.0 \pm 0.0$ | 
| Multi-Krum |  $0.314 \pm 0.179$ | $0.411 \pm 0.119$ | $0.548 \pm 0.106$ | $0.098 \pm 0.20$ |
| Median |  $0.615 \pm 0.099$ | $0.634 \pm 0.073$ | $0.768 \pm 0.063$ |  $0.522 \pm 0.103$ |
| Trimmed-mean | $0.466 \pm 0.075$ |  $0.614 \pm 0.068$ |  $0.787 \pm 0.074$ |  $0.494 \pm 0.120$ |

- `FedAda.py`: This code successfully implements the Federated Learning Adaptive Aggregation (FedAda) algorithm. The key idea is as follows, the local  models of the devices in each round are aggregated by the server using a dynamic weighted approach based on local model test accuracy. Experiments show that FedAda can not only improve global model generalization and performance but also accelerate the convergence of the FL global model, as shown in the following figure. Note that this experiment does not involve adversary attacks.
  
   ![FedAda vs FedAvg.](https://github.com/jjzgeeks/Secure_FL/blob/main/readme_pics/FedAda_vs_FedAvg.png)
