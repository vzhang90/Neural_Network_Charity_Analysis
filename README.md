# Neural Network Charity Analysis
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


## Overview of Analysis
Neural networks *(also known as artificial neural networks, or ANN)* are a set of algorithms that are modeled after the human brain with an advanced form of machine learning that recognizes patterns and features in input data and provides a clear quantitative output.

One way to use a neural network model is to create a classification algorithm that determines if an input belongs in one category versus another. Alternatively neural network models can behave like a regression model, where a dependent output variable can be predicted from independent input variables. Therefore, neural network models can be an alternative to many of the traditional statistical or machine learning models, such as random forest, logistic regression, or multiple linear regression. 

> <sub>**AlphabetSoupCharity code:** [AlphabetSoupCharity.ipynb](https://github.com/vzhang90/Neural_Network_Charity_Analysis/blob/main/AlphabetSoupCharity.ipynb)</sub>   

### Preprocessing Data for a Neural Network Model
1. Read in the [charity_data.csv](https://github.com/vzhang90/Neural_Network_Charity_Analysis/blob/main/charity_data.csv) to a Pandas DataFrame
    - identifies variable(s) considered the target(s)
    - identifies variable(s) considered the feature(s)
2. Drop the `EIN` and `NAME` columns
3. Determine the number of unique values for each column
4. For those columns that have more than 10 unique values, determine the number of data points for each unique value
5. Create a density plot to determine the distribution of the column values
6. Use the density plot to create a cutoff point to bin "rare" categorical variables together in a new column, `Other`, and then check if the binning was successful
7. Generate a list of categorical variables
8. Encode categorical variables using one-hot encoding, and place the variables in a new DataFrame
9. Merge the one-hot encoding DataFrame with the original DataFrame, and drop the originals
10. Split the preprocessed data into features and target arrays
11. Split the preprocessed data into training and testing datasets
12. Standardize numerical variables using Scikit-Learn’s StandardScaler class, then scale the data

### Compile, Train, and Evaluate the Model

### Optimize the Model

## Results

## Summary
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