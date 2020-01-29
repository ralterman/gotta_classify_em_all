import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import seaborn

pd.options.display.max_rows = 1000


url = 'https://pokemondb.net/pokedex/all'
page = requests.get(url)
data = pd.read_html(page.content)
pokedex = pd.DataFrame(data[0])
pokedex = pokedex.set_index('#')
pokedex = pokedex.loc[~pokedex.index.duplicated(keep='first')]

split_types = pokedex['Type'].str.split(' ', n = 1, expand = True)
pokedex['Type1'] = split_types[0]
pokedex['Type2'] = split_types[1]
pokedex = pokedex.drop(columns = ['Type', 'Total'])
pokedex = pokedex.rename(columns={"Name": "Pokemon_Name", "Sp. Atk": "Special_Attack", "Sp. Def": "Special_Defense"})
pokedex = pokedex.reindex(columns=['Pokemon_Name','Type1', 'Type2', 'HP', 'Attack', 'Defense', 'Special_Attack', 'Special_Defense', 'Speed'])


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


pokedex['height_m'] = ""
pokedex['weight_kg'] = ""

def get_height(row):
    pokemon = row['Pokemon_Name']
    if len(pokemon.split()) == 2:
        pokemon = row['Pokemon_Name'].replace(' ', '')
    base_url = 'https://www.serebii.net/pokemon/' + pokemon.lower() +'/'
    page = requests.get(base_url)
    soup = BeautifulSoup(page.content, 'html.parser')
    item = soup.find_all('td', class_='fooinfo')
    return item[-3].text.split('\t')[-1][:-1]

def get_weight(row):
    pokemon = row['Pokemon_Name']
    if len(pokemon.split()) == 2:
        pokemon = row['Pokemon_Name'].replace(' ', '')
    base_url = 'https://www.serebii.net/pokemon/' + pokemon.lower() +'/'
    page = requests.get(base_url)
    soup = BeautifulSoup(page.content, 'html.parser')
    item = soup.find_all('td', class_='fooinfo')
    print (pokemon)
    return item[-2].text.split('\t')[-1][:-2]

pokedex['height_m'] = pokedex.apply(lambda x: get_height(x), axis=1)
pokedex['weight_kg'] = pokedex.apply(lambda x: get_weight(x), axis=1)

pokedex
