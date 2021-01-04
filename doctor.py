'''
This file does not perform any Machine Learning task.
It simply takes the predicted system eg. Digestive System(string) and location eg. Pune(string)
as an input, performs a search on doctor's dataset and returns all the results in the form
of 3 lists:
doctor names - (list)
doctor location - (list)
doctor contact - (list) 
'''

import pandas as pd
import numpy as np

'''
If the doctor's dataset becomes very huge, it is better to save the loaded dataset using 
pickle rather than importing a large dataset everytime this program runs.
'''

# Please change the path to appropriate path:
df = pd.read_csv('data/doctor_dataset.csv')

# the following function takes input a single system and single location as input.
# If you have multiple systems to give as input, please call this function using a loop
# this function outputs 3 lists, first containing the names of doctors, second containing
# the location of the doctor and third containing their contact numbers.
# containing their respective contact numbers.


def get_doctors(system, location):
    # querying the dataset
    x = df[np.logical_and(df.Speciality == system, df.Location == location)]
    # If there are no results found for given location, we return doctors for that system from all other locations.
    if(len(x) == 0):
        x = df[df['Speciality'] == system]

    doctor_names = [name for name in x['Doctor']]
    doctor_location = [loc for loc in x['Location']]
    doctor_contact = [contact for contact in x['Contact']]

    return doctor_names, doctor_location, doctor_contact


# TESTING:

# test_system = 'Muscular System'
# test_location = "Pune"

# names, location, contacts = get_doctors(test_system, test_location)

# print(names)
# print(location)
# print(contacts)
