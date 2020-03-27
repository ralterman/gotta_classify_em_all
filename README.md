# Gotta Classify 'Em All
### _Classification of legendary vs. non-legendary Pokémon_

## Goal
To see if legendary Pokémon can be differentiated from non-legendary, and discover the underlying features that separate legendary Pokémon from their less esteemed counterparts.

## Data Cleaning/Preprocessing
__809 unique Pokémon (generations 1-7), 80 of them considered legendary__

__Features: HP, Attack, Defense, Special Attack, Special Defense, Speed, Height, Weight__
1. Used Pandas read_html function to scrape table of Pokémon and their attributes from the [Pokémon Database](https://pokemondb.net/pokedex/all "Pokémon Database")
  * Reset dataframe index to be the Pokédex number and dropped different sprites of the same Pokémon
  * Separated 'Type' category into 'Type 1' and 'Type 2' when necessary, and dropped 'Total' category
  * Fixed special Pokémon name cases for API calls to work
2. Leveraged the [PokéAPI](https://pokeapi.co/docs/v2.html "PokéAPI") for additional attributes
  * Got the URLs for all of the Pokémon via API call
  * Used the API to grab each Pokémon's height and weight and converted them to meters and kilograms, respectively
  * Concatenated those values to the Pokédex dataframe and dropped all Pokémon that were not included in the PokéAPI
3. Utilized BeautifulSoup library to scrape heights and weights of any remaining Pokémon of interest from [Serebii](https://www.serebii.net/pokemon/ "Serebii")
4. Got list of legendary Pokémon using BeautifulSoup library to scrape [Serebii](https://www.serebii.net/pokemon/legendary.shtml "Serebii Legendaries")
* Marked all Pokémon in the dataframe as either legendary (1) or non-legendary (0)
