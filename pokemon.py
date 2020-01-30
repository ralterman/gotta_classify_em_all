import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn


def get_urls(num_of_pokemon):
    url_list = []
    url = 'https://pokeapi.co/api/v2/pokemon/?limit=' + str(num_of_pokemon)
    response = requests.get(url)
    poke_dict = response.json()
    poke_list = poke_dict['results']
    return poke_list

x = get_urls(807)
x

def get_basic_info(poke_url_list):
    info_list = []
    for item in poke_url_list:
        response2 = requests.get(item['url'])
        info_dict = response2.json()
        del info_dict['moves']
        del info_dict['game_indices']
        del info_dict['abilities'][0]['ability']['url']
        info_list.append(info_dict)
        return info_list

y = get_basic_info(get_urls(1))
y[0]['weight']
y[0]['height']




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
