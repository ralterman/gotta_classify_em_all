import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import seaborn

pd.options.display.max_rows = 2000

url = 'https://pokemondb.net/pokedex/all'
page = requests.get(url)
data = pd.read_html(page.content)
pokedex = pd.DataFrame(data[0])
pokedex = pokedex.set_index('#')
pokedex = pokedex.loc[~pokedex.index.duplicated(keep='first')]

split_types = pokedex['Type'].str.split(' ', n = 1, expand = True)
pokedex['Type1'] = split_types[0]
pokedex['Type2'] = split_types[1]
pokedex = pokedex.drop(columns = 'Type')
pokedex = pokedex.rename(columns={"Name": "Pokemon_Name", "Sp. Atk": "Special_Attack", "Sp. Def": "Special_Defense"})
pokedex = pokedex.reindex(columns=['Pokemon_Name','Type1', 'Type2', 'HP', 'Attack', 'Defense', 'Special_Attack', 'Special_Defense', 'Speed', 'Total'])
pokedex
