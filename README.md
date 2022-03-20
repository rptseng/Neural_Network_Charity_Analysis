# Neural_Network_Charity_Analysis Challenge



## Model Optimization
The approach for optimizing the model involved the following steps:
- Dropping the "SPECIAL_CONSIDERATIONS_N" column from the list of features. This variable could be adding noise as it is redundant to the "SPECIAL_CONSIDERATIONS_Y" column during one-hot encoding.
- Increasing the number of features by including and encoding the "NAME" column from the original dataframe. After binning names for occurrences < 20, an additional 122 features were added to the model.
- Because additional input features were added, the number of neurons and hidden layers were increased. Three hidden layers with 180, 90, and 30 neurons respectively
- The activation function for the hidden layers was changed from ReLU to tanh, and the output layer remained as a sigmoid function

![model_op1.png](https://github.com/rptseng/Neural_Network_Charity_Analysis/blob/main/Resources/model_op1.png)

Evaluating the model after all these changes, the model accuracy achieved 77% with a loss of 0.48. This is an improvement over the original model which had an accuracy of 73% and a loss of 0.56.

![optimization_1](https://github.com/rptseng/Neural_Network_Charity_Analysis/blob/main/Resources/optimization_1.png)

