# Neural_Network_Charity_Analysis Challenge
Taking the DataFrame from charity_data.csv, the goal of this analysis is to fit a TensorFlow neural network machine learning model to predict whether funding applications will be successful if funded by Alphabet Soup.

## Results
### Processing Dataset and Training Model
The target variable for this model is the IS_SUCCESSFUL parameter for funding applications
The feature variables for the model are:
- NAME
- APPLICATION_TYPE 
- AFFILIATION, 
- CLASSIFICATION
- USE_CASE
- ORGANIZATION
- STATUS
- INCOME_AMT

The EIN identification column is irrelevant for the model because it is a unique identification number per application and is dropped during pre-processing.

The bins are created for the CLASSIFICATION column and APPLICATION_TYPE column.

![classification_bin.png](https://github.com/rptseng/Neural_Network_Charity_Analysis/blob/main/Resources/classification_bin.png)

![application_bin.png](https://github.com/rptseng/Neural_Network_Charity_Analysis/blob/main/Resources/application_bin.png)

Categorical variables are encoded using a one-hot encoder method.

![categorical_encoder.png](https://github.com/rptseng/Neural_Network_Charity_Analysis/blob/main/Resources/categorial_encoder.png)

With 43 input features, the model is defined with two hidden layers with 80 and 30 neurons respectively, using ReLU as the activation function and sigmoid as the output function.

![original_model.png](https://github.com/rptseng/Neural_Network_Charity_Analysis/blob/main/Resources/original_model.png)

The model's weights are checkpointed every five epochs and the results are saved to AlphabetSoupCharity.h5. 

![original_results.png](https://github.com/rptseng/Neural_Network_Charity_Analysis/blob/main/Resources/original_results.png)

### Model Optimization
The approach for optimizing the model involved the following steps:
- Dropping the "SPECIAL_CONSIDERATIONS_N" column from the list of features. This variable could be adding noise as it is redundant to the "SPECIAL_CONSIDERATIONS_Y" column during one-hot encoding.
- Increasing the number of features by including and encoding the "NAME" column from the original dataframe. After binning "NAMES" for occurrences < 20 = "Other", an additional 122 features were added to the model. The total number of features increased to 164 from 43 in the original model.
- Because additional input features were added, the number of neurons and hidden layers were increased. Three hidden layers with 180, 90, and 30 neurons respectively.
- The activation function for the hidden layers was changed from ReLU to tanh, and the output layer remained as a sigmoid function.

![model_op1.png](https://github.com/rptseng/Neural_Network_Charity_Analysis/blob/main/Resources/model_op1.png)

## Summary
The evaluation of the original model with 43 features with two hidden layers gave a result of 73% accuracy and loss of 56%.

Evaluating the model after all of the optimization changes, the model accuracy showed improvement over the original by achieving 77% accuracy with a loss of 0.48.

![optimization_1](https://github.com/rptseng/Neural_Network_Charity_Analysis/blob/main/Resources/optimization_1.png)

Another model that could be used is the Random Forest Classifier because it is suitable for addressing classification problems using tabular data. We would want a supervised model to address the question of whether the funding applications within this data set are likely to be successful or unsuccessful.