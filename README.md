# Secure_FL

The codes above were implemented from scratch based on my learning.

- `FedAvg.py` : FedAvg is the most basic and widely used aggregation algorithm in Federated Learning, which involves a simple averaging of the local model updates from participating clients to create a new global model. Note that the relevant experiments were conducted in highly non-IID scenarios. Here, I take [Adult Census Income Dataset](https://www.kaggle.com/datasets/uciml/adult-census-income) as an example to create non-IID partitions with 8 groups based on the 'income' feature, where 2 devices only hold label "0", 2 devices only hold label "1" and the remaining devices (4 devices) have both labels "0" and "1". The experimental results without data poisoning attack are as follows:
  ![The global model accuracy of FedAvg algorithm.](https://github.com/jjzgeeks/Secure_FL/blob/main/readme_pics/FedAvg_global_model_accuracy_30_8.png)
   ![The train loss and test loss of FedAvg algorithm.](https://github.com/jjzgeeks/Secure_FL/blob/main/readme_pics/FedAvg_loss_30_8.png)


- `data_poisoning_FL.py`: 2 devices only hold label "0" intentionally modify the labels associated with training data points, i.e., They both modify 60% of the real training data with label "0" to label "1". The goal is to mislead the model during training, causing it to learn incorrect or biased associations. Byzantine-resilient secure aggregation methods, i.e., Trimmed-mean, Krum, Multi-Krum and Median, are implemented. The experimental results are as follows:
    ![The results of secure aggregation methods.](https://github.com/jjzgeeks/Secure_FL/blob/main/readme_pics/Secure_aggregation_results.png)


  - `FedAda.py`: This code successfully implements the Federated Learning Adaptive Aggregation (FedAda) algorithm. The key idea is as follows, the local  models of the devices in each round are aggregated by the server using a dynamic weighted approach based on local model test accuracy. Experiments show that FedAda can not only improve global model generalization and performance but also accelerate the convergence of the global model.
     ![FedAda vs FedAvg.](https://github.com/jjzgeeks/Secure_FL/blob/main/readme_pics/FedAda_vs_FedAvg.png)
    
