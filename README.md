# **Neural Network Charity Analysis**
With knowledge of machine learning and neural networks, this project will use the features in the provided dataset [charity_data.csv](https://github.com/vzhang90/Neural_Network_Charity_Analysis/blob/main/charity_data.csv) to help create a binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup.

><sub>**Charity dataset:** [charity_data.csv](https://github.com/vzhang90/Neural_Network_Charity_Analysis/blob/main/charity_data.csv)</sub>   

Within this dataset are a number of columns that capture metadata about each organization, containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Such as the following:
- **EIN** and **NAME** —> Identification columns
- **APPLICATION_TYPE** —> Alphabet Soup application type
- **AFFILIATION** —> Affiliated sector of industry
- **CLASSIFICATION** -> Government organization classification
- **USE_CASE** —> Use case for funding
- **ORGANIZATION** —> Organization type
- **STATUS** —> Active status
- **INCOME_AMT** —> Income classification
- **SPECIAL_CONSIDERATIONS** —> Special consideration for application
- **ASK_AMT** —> Funding amount requested
- **IS_SUCCESSFUL** —> Was the money used effectively

---
---
## **Overview of Analysis**
> <sub>**AlphabetSoupCharity jupyter notebook file:** [AlphabetSoupCharity.ipynb](https://github.com/vzhang90/Neural_Network_Charity_Analysis/blob/main/AlphabetSoupCharity.ipynb)</sub>   
> <sub>**AlphabetSoupCharity.h5 file:** [AlphabetSoupCharity.h5]()</sub>   
> <sub>**AlphabetSoupCharity_Optimization jupyter notebook file:** [AlphabetSoupCharity_Optimization.ipynb](https://github.com/vzhang90/Neural_Network_Charity_Analysis/blob/main/AlphabetSoupCharity_Optimization.ipynb)</sub>   
> <sub>**AlphabetSoupCharity_Optimization.h5 file:** [AlphabetSoupCharity_Optimization.h5](https://github.com/vzhang90/Neural_Network_Charity_Analysis/blob/main/AlphabetSoupCharity_Optimization.ipynb)</sub> 

**Neural networks** *(also known as artificial neural networks, or ANN)* are a set of algorithms that are modeled after the human brain with an advanced form of machine learning that recognizes patterns and features in input data and provides a clear quantitative output.

One way to use a neural network model is to create a classification algorithm that determines if an input belongs in one category versus another. Alternatively neural network models can behave like a regression model, where a dependent output variable can be predicted from independent input variables. Therefore, neural network models can be an alternative to many of the traditional statistical or machine learning models, such as random forest, logistic regression, or multiple linear regression. 


---
### ***Preprocessing Data for a Neural Network Model***

>> *The process of **model->fit->predict/transform** follows the same general steps across all of data science:*   
>>  <sub>&ensp;&thinsp;&ensp;&thinsp;&ensp;&thinsp;&ensp;&thinsp;*i. Decide on a model, and create a model instance  
>>  &ensp;&thinsp;&ensp;&thinsp;&ensp;&thinsp;&ensp;&thinsp;ii. Split into training and testing sets, and preprocess the data  
>>  &ensp;&thinsp;&ensp;&thinsp;&ensp;&thinsp;&ensp;&thinsp;iii. Train/fit the training data to the model  
>>  &ensp;&thinsp;&ensp;&thinsp;&ensp;&thinsp;&ensp;&thinsp;iv. Use the model for predictions and transformations*</sub>

After importing the necessary dependencies to initialize [AlphabetSoupCharity.ipynb file](https://github.com/vzhang90/Neural_Network_Charity_Analysis/blob/main/AlphabetSoupCharity.ipynb), ***preprocess the data for a neural network model*** by:
1. Read in the [charity_data.csv](https://github.com/vzhang90/Neural_Network_Charity_Analysis/blob/main/charity_data.csv) to a Pandas DataFrame
    - identifies variable(s) considered the target(s)
    - identifies variable(s) considered the feature(s)
2. Drop the `EIN` and `NAME` columns
3. Determine the number of unique values for each column
4. The columns with more than 10 unique values grouped together
5. Create a density plot to determine the distribution of the column values
6. Use the density plot to create a cutoff point to bin "rare" categorical variables together in a new column, `Other`, and then check if the binning was successful
7. Generate a list of categorical variables
8. Encode categorical variables using one-hot encoding, and place the variables in a new DataFrame
9. Merge the one-hot encoding DataFrame with the original DataFrame, and drop the originals
10. Split the preprocessed data into features and target arrays
11. Split the preprocessed data into training and testing datasets
12. Standardize numerical variables using Scikit-Learn’s `StandardScaler` class, then scale the data
---
### ***Compile, Train, and Evaluate the Model***
Using knowledge of **TensorFlow**, this will now design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup–funded organization will be successful based on the features in the dataset. After considering the inputs to determine the number of neurons and layers in this model, this next portion of the code in [AlphabetSoupCharity.ipynb](https://github.com/vzhang90/Neural_Network_Charity_Analysis/blob/main/AlphabetSoupCharity.ipynb) will compile, train, and evaluate your binary classification model to calculate the model’s loss and accuracy.
1. Create a neural network model by assigning the number of input features and nodes for each layer using **Tensorflow Keras**
2. Create the first hidden layer  
    - `Keras` Dense class  
    - number of input features equal to the number of variables in feature DataFrame
    - choose an appropriate activation function
3. Add second hidden layer with `relu activation` function to identify nonlinear characteristics from the input values
4. Create output layer with `sigmoid` activation function
5. Check the structure of the model with `summary()` function
6. Compile the model to define the loss and accuracy metrics
    - using `binary_crossentropy` loss function, `adam` optimizer, and `accuracy` metrics
        - *same parameters as this neural network to use model as a binary classifier*
7. Train the model with `fit` function
8. Create a callback that saves the model's weights every 5 epochs
9. Evaluate the model using the test data with `evaluate()` function 
    - to evaluate the model's performance by testing its predictive capabilities (loss and accuracy) on our testing dataset
10. Save and export your results to an HDF5 file, and name it [AlphabetSoupCharity.h5](https://github.com/vzhang90/Neural_Network_Charity_Analysis/blob/main/AlphabetSoupCharity.h5)
---
### ***Optimize the Model***
Using knowledge of **TensorFlow**, optimize this model in order to achieve a ***target predictive accuracy higher than 75%***

1. Create a new Jupyter Notebook file and name it [AlphabetSoupCharity_Optimization.ipynb](https://github.com/vzhang90/Neural_Network_Charity_Analysis/blob/main/AlphabetSoupCharity_Optimization.ipynb)
2. Import your dependencies, and read in the [charity_data.csv](https://github.com/vzhang90/Neural_Network_Charity_Analysis/blob/main/charity_data.csv) to a Pandas DataFrame
3. Preprocess the dataset like before in [AlphabetSoupCharity.ipynb](https://github.com/vzhang90/Neural_Network_Charity_Analysis/blob/main/AlphabetSoupCharity.ipynb) , taking into account any modifications to optimize the model
4. Design a neural network model, taking into account any modifications that will optimize the model to achieve higher than 75% accuracy by using any or all of the following:
    - *Adjusting the input data to ensure that there are no variables or outliers that are causing confusion in the model by:*
        - *Dropping more or fewer columns*
        - *Creating more bins for rare occurrences in columns*
        - *Increasing or decreasing the number of values for each bin*
    - *Adding more neurons to a hidden layer*
    - *Adding more hidden layers*
    - *Using different activation functions for the hidden layers*
    - *Adding or reducing the number of epochs to the training regimen*
5. Create a callback that saves the model's weights every 5 epochs
6. Save and export your results to an HDF5 file, and name it [AlphabetSoupCharity_Optimization.h5](https://github.com/vzhang90/Neural_Network_Charity_Analysis/blob/main/AlphabetSoupCharity_Optimization.h5)
---
---
## **Results**
***Data Preprocessing***
- The `IS_SUCCESSFUL` column is the variable considered to be the target of this model
- The `APPLICATION_TYPE`, `AFFILIATION`, `CLASSIFICATION`, `USE_CASE`, `ORGANIZATION`, `STATUS`, `INCOME_AMT`, `SPECIAL_CONSIDERATIONS`, and `ASK_AMT` columns are the variables considered to be features of this model
- `EIN` & `NAME` identification columns are neither targets nor variables, so are removed from the input data

***Compiling, Training, and Evaluating the Model***
- The first layer had 8 neurons with a `relu` activation function
- The second layer had 5 neurons with a `relu` activation function
- The outer layer had 1 neuron with a `sigmoid` activation function 

- This model produced a fairly reliable classifier approximately 71.6% of the time.
- To try and increase model performance depending on the complexity of the dataset, I opted to increase the number of neurons and epochs to allow for the deep learning model more opportunities to optimize the weight coefficients.
---
---
## **Summary**
There are a number of ***advantages*** to using a neural network instead of a traditional statistical or machine learning model
- neural networks are effective at detecting complex, nonlinear relationships
- neural networks have greater tolerance for messy data and can learn to ignore noisy characteristics in data

The two biggest ***disadvantages*** to using a neural network model are: 
1. the layers of neurons are often too complex to dissect and understand 
    - creating a black box problem
2. neural networks are prone to overfitting 
    - characterizing the training data so well that it does not generalize to test data effectively   
        - *overfitting occurs when a model gives undue importance to patterns within a particular dataset that are not found in other, similar datasets*
    
However, both of the disadvantages can be mitigated and accounted for.

The neural network deep learning model was able to predict whether applicants will be successful if funded by Alphabet Soup approximately 71.6% of the time. 