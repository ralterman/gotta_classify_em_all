import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import imblearn
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, tree
import xgboost
from xgboost import plot_importance
import pickle

pd.options.display.max_rows = 1000


# Getting and cleaning data:

# Scrape all pokémon and their attributes from pokemon.db
url = 'https://pokemondb.net/pokedex/all'
page = requests.get(url)
data = pd.read_html(page.content)
pokedex = pd.DataFrame(data[0])
pokedex = pokedex.set_index('#') # Change index to pokédex number
pokedex = pokedex.loc[~pokedex.index.duplicated(keep='first')] # Drop the different sprites of pokémon

# Separate types to get Type1 and Type2 of each pokémon
split_types = pokedex['Type'].str.split(' ', n = 1, expand = True)
pokedex['Type1'] = split_types[0]
pokedex['Type2'] = split_types[1]

# Drop unnecessary columns, rename and reorder the rest
pokedex = pokedex.drop(columns = ['Type', 'Total'])
pokedex = pokedex.rename(columns={"Name": "Pokemon_Name", "Sp. Atk": "Special_Attack", "Sp. Def": "Special_Defense"})
pokedex = pokedex.reindex(columns=['Pokemon_Name','Type1', 'Type2', 'HP', 'Attack', 'Defense', 'Special_Attack', 'Special_Defense', 'Speed'])


# If specific sprite label remains after dropping duplicates, get rid of the extra parts of the pokémon name
# Fix special pokémon name cases to allow for API calls to work
def default_name(row):
    if len(row['Pokemon_Name'].split()) > 2:
        return row['Pokemon_Name'].split()[0]
    elif (len(row['Pokemon_Name'].split()) == 2) and (row['Pokemon_Name'].split()[1] == 'Male'):
        return row['Pokemon_Name'].split()[0]
    elif row['Pokemon_Name'][-1] == '♀':
        return row['Pokemon_Name'][:-1] + 'f'
    elif row['Pokemon_Name'][-1] == '♂':
        return row['Pokemon_Name'][:-1] + 'm'
    elif 'é' in row['Pokemon_Name']:
        return row['Pokemon_Name'].replace('é', 'e')
    else:
        return row['Pokemon_Name']

pokedex['Pokemon_Name'] = pokedex.apply(default_name, axis = 1)


# Get url for each pokémon for PokéAPI calls from the PokéAPI
def get_urls(num_of_pokemon):
    url_list = []
    url = 'https://pokeapi.co/api/v2/pokemon/?limit=' + str(num_of_pokemon)
    response = requests.get(url)
    poke_dict = response.json()
    poke_list = poke_dict['results']
    return poke_list

# Get each pokémon's height and weight from PokéAPI
def get_height_weight(poke_url_list):
    info_list = []
    info_list.append(('', '')) # Add empty row to beginning of list so indices match
    for item in poke_url_list:
        response2 = requests.get(item['url'])
        info_dict = response2.json()
        height = info_dict['height']/10
        weight = info_dict['weight']/10
        info_list.append((height, weight))
    return info_list

# hw_list = get_height_weight(get_urls(807))


# Create new dataframe of just heights and weights and concatenate that to the end of the pokédex dataframe
hw = pd.DataFrame(hw_list)
hw = hw.rename(columns={0: 'Height_M', 1: 'Weight_KG'})
pokemon = pd.concat([pokedex, hw], axis=1)
pokemon = pokemon.drop(0, axis=0) # Drop the first empty row
pokemon = pokemon.drop(pokemon.index[807:]) # Drop all pokémon that were not on the PokéAPI


# Get heights and weights for the few pokémon not on the PokéAPI but that you want to look at from serebii
def get_height(row):
    pokemon = row['Pokemon_Name']
    if len(pokemon.split()) == 2: # If pokémon name is two words, make it one for url - always make it all lowercase
        pokemon = row['Pokemon_Name'].replace(' ', '')
    base_url = 'https://www.serebii.net/pokemon/' + pokemon.lower() +'/'
    page = requests.get(base_url)
    soup = BeautifulSoup(page.content, 'html.parser')
    item = soup.find_all('td', class_='fooinfo')
    if item[-3].text.split('\t')[-1][:-1] == '':
        return item[-6].text.split('\t')[-1][:-1]
    else:
        return item[-3].text.split('\t')[-1][:-1]

def get_weight(row):
    pokemon = row['Pokemon_Name']
    if len(pokemon.split()) == 2:
        pokemon = row['Pokemon_Name'].replace(' ', '')
    base_url = 'https://www.serebii.net/pokemon/' + pokemon.lower() +'/'
    page = requests.get(base_url)
    soup = BeautifulSoup(page.content, 'html.parser')
    item = soup.find_all('td', class_='fooinfo')
    if item[-2].text.split('\t')[-1][:-2] == '':
        return item[-5].text.split('\t')[-1][:-2]
    else:
        return item[-2].text.split('\t')[-1][:-2]

# Make new dataframe for just those pokémon and concatenate that to the big dataframe
pokemon2 = pokedex[807:810]
pokemon2['Height_M'] = pokemon2.apply(lambda x: get_height(x), axis=1)
pokemon2['Weight_KG'] = pokemon2.apply(lambda x: get_weight(x), axis=1)


pokemon = pd.concat([pokemon, pokemon2], ignore_index=True)
pokemon = pokemon.shift()[1:]
pokemon = pokemon.rename_axis('Pokedex_Num', axis='columns')


# Find which pokémon are considered 'legendary' from serebii (each pokémon was the third item in the returned list from scraping)
def get_legendary(url):
    idx = 0
    legend_list = []
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    item = soup.find_all('td', align='center')
    while idx < len(item):
        legend_list.append(item[idx].text)
        idx += 3
    return legend_list

legends = get_legendary('https://www.serebii.net/pokemon/legendary.shtml')


# Mark legendary pokémon with a 1 and non-legendary with a 0
def mark_legends(row):
    if row['Pokemon_Name'] in legends:
        return 1
    else:
        return 0

pokemon['Legendary'] = pokemon.apply(lambda x: mark_legends(x), axis=1)

pokemon.to_csv('pokemon.csv')


#-----------------------------------------------------------------------------------------------


# Exploratory Data Analysis:

pokemon = pd.read_csv('pokemon.csv', index_col = 0)
pokemon


# Make correlation heatmap
plt.figure(figsize=(20,10))
ax = sns.heatmap(pokemon.corr(), annot=True, cmap='Greens')
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)


# Make scatter matrix of each variable's distribution
pd.plotting.scatter_matrix(pokemon.iloc[:, 2:], figsize=(25,10))


# Create overlapping histograms to show different distributions of features for legendary vs non-legendary
def multi_hist_plot(df_1, df_2, to_plot):
    fig = plt.figure(figsize=(10,5))
    plt.style.use('seaborn')
    sns.set_palette('bright')
    hist_1 = sns.distplot(df_1[to_plot], kde = False)
    hist_2 = sns.distplot(df_2[to_plot], kde = False)
    plt.show()

df_1 = pokemon[pokemon['Legendary'] == 1]
df_2 = pokemon[pokemon['Legendary'] == 0]


hp_hist = multi_hist_plot(df_1, df_2, 'HP')
attack_hist = multi_hist_plot(df_1, df_2, 'Attack')
defense_hist = multi_hist_plot(df_1, df_2, 'Defense')
spattack_hist = multi_hist_plot(df_1, df_2, 'Special_Attack')
spdefense_hist = multi_hist_plot(df_1, df_2, 'Special_Defense')
speed_hist = multi_hist_plot(df_1, df_2, 'Speed')
height_hist = multi_hist_plot(df_1, df_2, 'Height_M')
weight_hist = multi_hist_plot(df_1, df_2, 'Weight_KG')


#-----------------------------------------------------------------------------------------------


# Modeling

# Set variables for train test split
y = pokemon['Legendary']
x = pokemon[['HP', 'Attack', 'Defense', 'Special_Attack', 'Special_Defense', 'Speed', 'Height_M', 'Weight_KG']]
X = pd.DataFrame(x, columns = x.columns)

X_train , X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

# Due to class imbalance, synthesize fake data for the minority class (legendary) using SMOTE
smote = SMOTE(sampling_strategy='minority')
X_synth, y_synth = smote.fit_sample(X_train, y_train)


# Create list of different classifiers/models you want to run
classifiers = []

model1 = xgboost.XGBClassifier()
classifiers.append(model1)

model2 = svm.SVC()
classifiers.append(model2)

model3 = tree.DecisionTreeClassifier()
classifiers.append(model3)

model4 = RandomForestClassifier()
classifiers.append(model4)


# Fit each model with synthesized training data and find accuracy, precision, recall, and F1 score and print out confusion matrix and classification report
for clf in classifiers:
    clf.fit(X_synth, y_synth)
    y_pred= clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print("Accuracy of %s is %s"%(clf, acc))
    print("Precision of %s is %s"%(clf, prec))
    print("Recall of %s is %s"%(clf, recall))
    print("F1 Score of %s is %s"%(clf, f1))
    print("")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("")
    print(classification_report(y_test, y_pred))
    print("")
    print("")


# Since SVM performed best, run grid search on that model to find optimal parameters
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],  'kernel': ['rbf', 'linear', 'poly', 'sigmoid']}
grid = GridSearchCV(svm.SVC(), param_grid)
grid_search = grid.fit(X_synth, y_synth)

# Optimal SVM parameters:
print (grid_search.best_params_)
print (grid_search.best_estimator_)

grid_predictions = grid_search.predict(X_test)

# Print new confusion matrix and classification report post grid search
print(confusion_matrix(y_test, grid_predictions))
print(classification_report(y_test, grid_predictions))

# Dump model to separately reverse engineer to find names of misclassified pokémon in pokemon.py file
model = svm.SVC(C=1000, kernel='rbf')
model.fit(X_train, y_train)
pickle.dump(model, open('model.pkl', 'wb'))


# Run grid search on XGBoost model
param_grid2 = {
    'learning_rate': [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    'max_depth': [2, 4, 6, 8],
    'min_child_weight': [1, 3, 5, 7],
    'subsample': [0.3, 0.5, 0.7],
    'n_estimators': [50, 100, 150],
}

grid_clf = GridSearchCV(xgboost.XGBClassifier(), param_grid2, scoring='f1', cv=None, n_jobs=1)
xgb_gridsearch = grid_clf.fit(X_synth, y_synth)

best_parameters = grid_clf.best_params_

# Optimal XGBoost parameters:
print('Grid Search found the following optimal parameters: ')
for param_name in sorted(best_parameters.keys()):
    print('%s: %r' % (param_name, best_parameters[param_name]))

xgb_model = xgboost.XGBClassifier(learning_rate=0.1, max_depth=6, min_child_weight=1, n_estimators=100, subsample=0.7)


test_preds = grid_clf.predict(X_test)

# Print new confusion matrix and classification report post grid search on XGBoost model
print(confusion_matrix(y_test, test_preds))
print(classification_report(y_test, test_preds))

xgb_model.fit(X_synth, y_synth)

# Find feature importances for XGBoost model
print(xgb_model.feature_importances_)
# Plot feature importances
def plot_feature_importances(model):
    n_features = X_synth.shape[1]
    plt.figure(figsize=(15,10))
    plt.barh(range(n_features), model.feature_importances_, align='center', color='green')
    plt.yticks(np.arange(n_features), X_train.columns.values)
    plt.xlabel('Feature importance')
    plt.ylabel('Feature')

plot_feature_importances(xgb_model)
