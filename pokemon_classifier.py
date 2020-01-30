import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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


def get_urls(num_of_pokemon):
    url_list = []
    url = 'https://pokeapi.co/api/v2/pokemon/?limit=' + str(num_of_pokemon)
    response = requests.get(url)
    poke_dict = response.json()
    poke_list = poke_dict['results']
    return poke_list

def get_height_weight(poke_url_list):
    info_list = []
    info_list.append(('', ''))
    for item in poke_url_list:
        response2 = requests.get(item['url'])
        info_dict = response2.json()
        height = info_dict['height']/10
        weight = info_dict['weight']/10
        info_list.append((height, weight))
    return info_list

# hw_list = get_height_weight(get_urls(807))


hw = pd.DataFrame(hw_list)
hw = hw.rename(columns={0: 'Height_M', 1: 'Weight_KG'})
pokemon = pd.concat([pokedex, hw], axis=1)
pokemon = pokemon.drop(0, axis=0)
pokemon = pokemon.drop(pokemon.index[807:])


def get_height(row):
    pokemon = row['Pokemon_Name']
    if len(pokemon.split()) == 2:
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

pokemon2 = pokedex[807:810]
pokemon2['Height_M'] = pokemon2.apply(lambda x: get_height(x), axis=1)
pokemon2['Weight_KG'] = pokemon2.apply(lambda x: get_weight(x), axis=1)


pokemon = pd.concat([pokemon, pokemon2], ignore_index=True)
pokemon = pokemon.shift()[1:]
pokemon = pokemon.rename_axis('Pokedex_Num', axis='columns')


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


def mark_legends(row):
    if row['Pokemon_Name'] in legends:
        return 1
    else:
        return 0

pokemon['Legendary'] = pokemon.apply(lambda x: mark_legends(x), axis=1)

pokemon.to_csv('pokemon.csv')

#-----------------------------------------------------------------------------------------------

pokemon = pd.read_csv('pokemon.csv', index_col = 0)
pokemon

plt.figure(figsize=(20,10))
ax = sns.heatmap(pokemon.corr(), annot=True, cmap='Greens')
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)

pd.plotting.scatter_matrix(pokemon.iloc[:, 2:], figsize=(25,8))
