# Gotta Classify 'Em All
### _Classification of legendary vs. non-legendary Pokémon_

## Goal
To see if legendary Pokémon can be differentiated from non-legendary, and discover the underlying features that separate legendary Pokémon from their less esteemed counterparts.

## Data Cleaning
__809 unique Pokémon (generations 1-7), 80 of them considered legendary__
1. Used Pandas read_html function to scrape table of Pokémon and their attributes from the [Pokémon Database](https://pokemondb.net/pokedex/all "Pokémon Database")
  * Reset dataframe index to be the Pokédex number and dropped different sprites of the same Pokémon
  * Separated 'Type' category into 'Type 1' and 'Type 2' when necessary, and dropped 'Total' category
  * Fixed special Pokémon name cases for API calls to work
2. Leveraged the [PokéAPI](https://pokeapi.co/docs/v2.html "PokéAPI") for additional attributes
  * Got the URLs for all of the Pokémon via API call
  * Used the API to grab each Pokémon's height and weight and converted them to meters and kilograms, respectively
  * Concatenate these values to the Pokédex dataframe and drop all Pokémon that were not included in the PokéAPI
3. 
