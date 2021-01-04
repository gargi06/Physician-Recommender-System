# To resolve some future warning, upgrade version of sklearn by un-commenting below line:
# pip install --user --upgrade scikit-learn==0.22.2

'''
This file has a function make_prediction(input_string) which takes input string of
symptoms as an argument from the user and returns the risk value of disease as 
1 or 2.'''

'''
We have tested the performance of various classification algorithms and Logistic
Regression model gives the best results for this dataset.
hyper-parameter tuning / alternate model might further increase the accuracy of the model.
'''

'''
TfidfTransformer and CountVectorizer are used to vectorize the input string data,
so that it can be used for training by Classification model.
'''

'''
This file saves 2 files using pickle.
First: CountVectorizer --> We have converted training data to Tfidf Vectors for modelling.
We need to covert the input string from the user using the same Tfidf object used 
for training.

Second: LogisticRegression trained model , to avoid training the model again and again 
every time the program is run.

Be careful with the file names and their paths in your system, as it might lead to an error.
'''

'''
Please note all the libraries required in this file and install them prior to running
this file in your system.
'''
try:
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    import numpy as np
    import pandas as pd
    import pickle
    # from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    f = open('final_model.sav')
    classifier = pickle.load(open('final_model.sav', 'rb'))

except:
    import re

    # importing dataset
    '''Please change the path to the appropriate path in your system'''
    df = pd.read_csv('data/df_diseases.csv')

    # dropping unwanted col's
    df.drop([df.columns[0], df.columns[2]], axis=1, inplace=True)

    # Filling NaN values with empty string
    df.fillna('', inplace=True)

    # some pre-processing
    for i in range(len(df)):

        df.loc[i, 'symptoms'] = re.sub(
            '[^A-Za-z0-9., ]+', '', df.loc[i, 'symptoms'])
        df.loc[i, 'causes'] = re.sub(
            '[^A-Za-z0-9., ]+', '', df.loc[i, 'causes'])
        df.loc[i, 'risk_factor'] = re.sub(
            '[^A-Za-z0-9., ]+', '', df.loc[i, 'risk_factor'])

        df.loc[i, 'overview'] = re.sub(
            '[^A-Za-z0-9., ]+', '', df.loc[i, 'overview'])
        df.loc[i, 'treatment'] = re.sub(
            '[^A-Za-z0-9., ]+', '', df.loc[i, 'treatment'])
        df.loc[i, 'medication'] = re.sub(
            '[^A-Za-z0-9., ]+', '', df.loc[i, 'medication'])

        df.loc[i, 'home_remedies'] = re.sub(
            '[^A-Za-z0-9., ]+', '', df.loc[i, 'home_remedies'])

    # some more preprocessing
    df['name'] = df['name'].str.lower()
    df['symptoms'] = df['symptoms'].str.lower()
    df['causes'] = df['causes'].str.lower()

    df['risk_factor'] = df['risk_factor'].str.lower()
    df['overview'] = df['overview'].str.lower()
    df['treatment'] = df['treatment'].str.lower()

    df['medication'] = df['medication'].str.lower()
    df['home_remedies'] = df['home_remedies'].str.lower()

    # adding a new column named 'class' which will contain the category of the disease
    df['class'] = -1

    # combining classes 1&2 = class 1, 3&4 = class 2
    def create_classes(row):
        if((row['medication'] == '') and (row['home_remedies'] == '')):
            return 1
        elif((row['medication'] != '') and (row['home_remedies'] == '')):
            # return 2
            return 1
        elif((row['medication'] != '') and (row['home_remedies'] != '')):
            return 2
        elif((row['medication'] == '') and (row['home_remedies'] != '')):
            # return 4
            return 2

    df['class'] = df.apply(create_classes, axis=1)

    test_df = df.copy()

    test_df['input'] = ''

    for i in range(len(test_df)):
        test_df.loc[i, 'input'] = test_df.loc[i, 'symptoms']+test_df.loc[i,
                                                                         'causes']+test_df.loc[i, 'risk_factor']+test_df.loc[i, 'overview']

    # Training model on entire dataset
    X = test_df['input'].values

    Y = test_df['class'].values

    # Converting text into tfidf vector

    from sklearn.feature_extraction.text import CountVectorizer

    # saving CountVectorizer object to use it later during prediction
    count_vect = CountVectorizer(decode_error="replace").fit(X)

    file_name_for_vector = 'feature.pkl'

    pickle.dump(count_vect.vocabulary_, open(file_name_for_vector, 'wb'))

    X_counts = count_vect.transform(X)

    # TF-IDF
    from sklearn.feature_extraction.text import TfidfTransformer
    tfidf_transformer = TfidfTransformer().fit(X_counts)

    X_tfidf = tfidf_transformer.transform(X_counts)

    # creating logistic regression classifier and saving it.
    classifier = LogisticRegression(class_weight='balanced').fit(X_tfidf, Y)

    filename = 'final_model.sav'

    pickle.dump(classifier, open(filename, 'wb'))


def make_prediction(input1):

    # because TfidfTransformer.fit() method takes input of np array object, converting the string input to that format.

    input_arr = np.array([input1])
    ser = pd.Series(input_arr)
    new_input = ser.values

    transformer = TfidfTransformer()

    file_name_for_vector = 'feature.pkl'

    # Loading saved model of CountVectorizer

    loaded_vec = CountVectorizer(
        decode_error="replace", vocabulary=pickle.load(open(file_name_for_vector, "rb")))

    input_tfidf = transformer.fit_transform(loaded_vec.transform(new_input))

    return classifier.predict(input_tfidf)[0]


'''For testing:'''
# x = 'fever cough cold'

# print(make_prediction(x))
