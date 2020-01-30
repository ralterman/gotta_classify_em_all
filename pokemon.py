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
pokedex

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

pokemon2 = pokedex[807:809]
pokemon2['Weight_KG'] = pokemon2.apply(lambda x: get_weight(x), axis=1)
pokemon2['Height_KG'] = pokemon2.apply(lambda x: get_height(x), axis=1)
pokemon2


pokemon = 'Melmetal'
if len(pokemon.split()) == 2:
    pokemon = row['Pokemon_Name'].replace(' ', '')
base_url = 'https://www.serebii.net/pokemon/' + pokemon.lower() +'/'
page = requests.get(base_url)
soup = BeautifulSoup(page.content, 'html.parser')
item = soup.find_all('td', class_='fooinfo')
item
for thing in item:
    if 'kg' in thing.text:
        print(thing.text)
x = item[-6].text.split('\t')[-1][:-1]
x




# def get_urls(num_of_pokemon):
#     url_list = []
#     url = 'https://pokeapi.co/api/v2/pokemon/?limit=' + str(num_of_pokemon)
#     response = requests.get(url)
#     poke_dict = response.json()
#     poke_list = poke_dict['results']
#     return poke_list
#
# def get_height_weight(poke_url_list):
#     info_list = []
#     info_list.append(('', ''))
#     for item in poke_url_list:
#         response2 = requests.get(item['url'])
#         info_dict = response2.json()
#         height = info_dict['height']/10
#         weight = info_dict['weight']/10
#         info_list.append((height, weight))
#     return info_list
#
# # hw_list = get_height_weight(get_urls(807))
#
#
# hw = pd.DataFrame(hw_list)
# hw = hw.rename(columns={0: 'Height_M', 1: 'Weight_KG'})
# pokemon = pd.concat([pokedex, hw], axis=1)
# pokemon = pokemon.drop(0, axis=0)
# pokemon = pokemon.drop(pokemon.index[807:])
#
# pokemon.to_csv('pokemon.csv')
#
#
# pokemon = pd.read_csv('pokemon.csv', index_col = 0)
# pokemon


# import requests
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn
#
#
# def get_urls(num_of_pokemon):
#     url_list = []
#     url = 'https://pokeapi.co/api/v2/pokemon/?limit=' + str(num_of_pokemon)
#     response = requests.get(url)
#     poke_dict = response.json()
#     poke_list = poke_dict['results']
#     return poke_list
#
# x = get_urls(807)
# x
#
# def get_basic_info(poke_url_list):
#     info_list = []
#     for item in poke_url_list:
#         response2 = requests.get(item['url'])
#         info_dict = response2.json()
#         del info_dict['moves']
#         del info_dict['game_indices']
#         del info_dict['abilities'][0]['ability']['url']
#         info_list.append(info_dict)
#         return info_list
#
# y = get_basic_info(get_urls(1))
# y[0]['weight']
# y[0]['height']
#
#
#

# def get_height(row):
#     pokemon = row['Pokemon_Name']
#     if len(pokemon.split()) == 2:
#         pokemon = row['Pokemon_Name'].replace(' ', '')
#     base_url = 'https://www.serebii.net/pokemon/' + pokemon.lower() +'/'
#     page = requests.get(base_url)
#     soup = BeautifulSoup(page.content, 'html.parser')
#     item = soup.find_all('td', class_='fooinfo')
#     # if (item[-3].text.split('\t')[-1][:-2] == '') and (item[-6].text.split('\t')[-1][:-2] == ''):
#     #     return item[-10].text.split('\t')[-1][:-1]
#     # elif item[-3].text.split('\t')[-1][:-1] == '':
#     #     return item[-6].text.split('\t')[-1][:-1]
#     # elif '\n' in item[-3].text.split('\t')[-1][:-2]:
#     #     return item[-4].text.split('\t')[-1][:-5]
#     # else:
#     return item[-3].text.split('\t')[-1][:-1]
#
# def get_weight(row):
#     pokemon = row['Pokemon_Name']
#     if len(pokemon.split()) == 2:
#         pokemon = row['Pokemon_Name'].replace(' ', '')
#     base_url = 'https://www.serebii.net/pokemon/' + pokemon.lower() +'/'
#     page = requests.get(base_url)
#     soup = BeautifulSoup(page.content, 'html.parser')
#     item = soup.find_all('td', class_='fooinfo')
#     print (pokemon)
#     # if (item[-2].text.split('\t')[-1][:-2] == '') and (item[-5].text.split('\t')[-1][:-2] == ''):
#     #     return item[-9].text.split('\t')[-1][:-2]
#     # elif item[-2].text.split('\t')[-1][:-2] == '':
#     #     return item[-5].text.split('\t')[-1][:-2]
#     # elif '\n' in item[-2].text.split('\t')[-1][:-2]:
#     #     return item[-3].text.split('\t')[-1][:-7]
#     # else:
#     return item[-2].text.split('\t')[-1][:-2]
#
# pokemon2 = pokemon[808:]
# # poke['height_m'] = poke.apply(lambda x: get_height(x), axis=1)
# pokemon2['weight_kg'] = pokemon2.apply(lambda x: get_weight(x), axis=1)
# pokemon2
#
#
# # pokemon = 'Charizard'
# # if len(pokemon.split()) == 2:
# #     pokemon = row['Pokemon_Name'].replace(' ', '')
# # base_url = 'https://www.serebii.net/pokemon/' + pokemon.lower() +'/'
# # page = requests.get(base_url)
# # soup = BeautifulSoup(page.content, 'html.parser')
# # item = soup.find_all('td', class_='fooinfo')
# # item
# # for thing in item:
# #     if 'kg' in thing.text:
# #         print(thing.text)
# # x = item[-5].text
# # x
