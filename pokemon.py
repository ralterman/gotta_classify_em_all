import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
import imblearn
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, tree
import xgboost
import pickle

pd.options.display.max_rows = 1000

pokemon = pd.read_csv('pokemon.csv', index_col = 0)
pokemon


# Load model
model = pickle.load(open('model.pkl', 'rb'))

# Set variables for train test split
y = pokemon['Legendary']
x = pokemon[['Pokemon_Name', 'HP', 'Attack', 'Defense', 'Special_Attack', 'Special_Defense', 'Speed', 'Height_M', 'Weight_KG']]
X = pd.DataFrame(x, columns = x.columns)

X_train , X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)


# Fit each model and find accuracy, precision, recall, and F1 score and print out confusion matrix and classification report
y_pred= model.predict(X_test.drop('Pokemon_Name', axis=1))
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("Accuracy of %s is %s"%(model, acc))
print("Precision of %s is %s"%(model, prec))
print("Recall of %s is %s"%(model, recall))
print("F1 Score of %s is %s"%(model, f1))
print("")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("")
print(classification_report(y_test, y_pred))
print("")
print("")



# Make dataframe from testing data
pokemon2 = pd.DataFrame(y_test)

# Add prediction data to the dataframe for comparison (actual vs predicted)
pokemon2 = pokemon2.rename(columns={"Legendary": "y_test"})
pokemon2['y_pred'] = y_pred


# Concatenate that dataframe onto the main dataframe to match with pokémon names
y = pd.concat([X_test, pokemon2], axis=1)
y


# Check if misclassified and return the names of the pokémon that are misclassified to a list but also 'None' if pokémon was not misclassified
def misclassified(row):
    if row['y_test'] != row['y_pred']:
        return row['Pokemon_Name']

misclassifications = []
misclassifications = y.apply(lambda x: misclassified(x), axis=1)

# Get ride of 'None' from list so only those that were misclassified remain
missed = []
for poke in misclassifications:
    if poke != None:
        missed.append(poke)

missed


# Make new dataframe for just the misclassified pokémon with all of their attributes
missed_list = pd.DataFrame()
for item in missed:
    x = pd.DataFrame(pokemon.loc[pokemon['Pokemon_Name'] == item])
    missed_list = pd.concat([missed_list, x], axis=0)

# Manually label whether the pokémon's misclassification was a False Positive or a False Negative
missed_list['Label'] = ['False Neg', 'False Neg', 'False Pos', 'False Neg', 'False Pos', 'False Neg']

missed_list


# Drop misclassified pokémon from original dataframe
b = pokemon.drop([720, 763, 778, 793, 796, 808])

# Get averages for each of the categories in the main dataframe, differentiating between legendary and non-legendary
c = b.groupby(['Legendary']).mean()
c
