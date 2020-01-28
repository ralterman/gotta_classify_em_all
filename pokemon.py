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

# x = get_urls(807)
# x

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
y
