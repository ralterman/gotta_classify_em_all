# Gotta Classify 'Em All
### _Classification model separating legendary from non-legendary Pokémon_

## Goal
To see if legendary Pokémon can be differentiated from non-legendary, and discover the underlying features that separate legendary Pokémon from their less esteemed counterparts.

## Data Cleaning/Preprocessing
__809 unique Pokémon (generations 1-7), 80 of them considered legendary__

__Features: HP, Attack, Defense, Special Attack, Special Defense, Speed, Height, Weight__
1. Used Pandas read_html function to scrape table of Pokémon and their attributes from the [Pokémon Database](https://pokemondb.net/pokedex/all "Pokémon Database")
   * Reset dataframe index to be the Pokédex number and dropped different sprites of the same Pokémon
   * Separated 'Type' category into 'Type 1' and 'Type 2' when necessary, and dropped 'Total' category (althought 'Type' categories            ultimately not used for modeling)
   * Fixed special Pokémon name cases for API calls to work
2. Leveraged the [PokéAPI](https://pokeapi.co/docs/v2.html "PokéAPI") for additional attributes
   * Got the URLs for all of the Pokémon via API call
   * Used the API to grab each Pokémon's height and weight and converted them to meters and kilograms, respectively
   * Concatenated those values to the Pokédex dataframe and dropped all Pokémon that were not included in the PokéAPI
3. Utilized BeautifulSoup library to scrape heights and weights of any remaining Pokémon of interest from [Serebii](https://www.serebii.net/pokemon/ "Serebii")

4. Got list of legendary Pokémon from scraping [Serebii](https://www.serebii.net/pokemon/legendary.shtml "Serebii Legendaries") using        BeautifulSoup library
   * Marked all Pokémon in the dataframe as either legendary (1) or non-legendary (0)

<p align="center"><img src="https://github.com/ralterman/pokemon_classifier/blob/master/images/data.png"></p>

## Exploratory Data Analysis
1. Correlation heatmap of Pokémon features
  <p align="center"><img src="https://github.com/ralterman/pokemon_classifier/blob/master/images/heatmap.png"></p>

2. Overlapping histograms showing the distributions of all features for non-legendary vs. legendary Pokémon
  <p align="center"><img src="https://github.com/ralterman/pokemon_classifier/blob/master/images/histograms1.png"></p>
  <p align="center"><img src="https://github.com/ralterman/pokemon_classifier/blob/master/images/histograms2.png"></p>

## Modeling
* Explanatory variables (aforementioned 8 features) vs. response variable of legendary or not
* Performed SMOTE on training data to account for class imbalance since only about 10% of all Pokémon in the data were legendary —           synthesized more elements for the legendary class
* Trained and tested 4 classification machine learning models — Decision Tree, Random Forest, XGBoost, and SVM:

    | Model         | Accurary | Precision | Recall | F1-Score |
    |:-------------:|:--------:|:---------:|:------:|:--------:|
    | Decision Tree | 0.90     | 0.50      | 0.62   | 0.56     |
    | Random Forest | 0.95     | 0.72      | 0.81   | 0.76     |
    | XGBoost       | 0.96     | 0.74      | 0.88   | 0.80     |
    | _SVM_         | _0.96_   | _0.78_    | _0.88_ | _0.82_   |


* Ran Grid Search on top-performing SVM model with C values ranging from 0.001 to 1000 (moving one decimal place to the right for each       value in that range) and for all possible kernel values
* Optimal parameters were C = 1000 and kernel = rbf (radial)
* Accuracy, Precision, and F1-Score all increased, while Recall dropped:

    | Model         | Accurary | Precision | Recall | F1-Score |
    |:-------------:|:--------:|:---------:|:------:|:--------:|
    | SVM (before)  | 0.96     | 0.78      | 0.88   | 0.82     |
    | _SVM (after)_ | _0.97_   | _0.92_    | _0.75_ | _0.83_   |

<p align="center"><img src="https://github.com/ralterman/pokemon_classifier/blob/master/images/confusion_matrix.png"></p>

__Final model overall displayed 97% accuracy__

## Feature Importance
* Because SVM does not have feature importance with the rbf kernel, I ran a grid search on my XGBoost model, since it was also very good,   and calculated the feature importance on that model
* Feature importance seems to match correlations with ‘legendary’ from heatmap, with 'Special Attack' appearing to have the most influence

<p align="center"><img src="https://github.com/ralterman/pokemon_classifier/blob/master/images/feature_importance.png"></p>

## Insight into Misclassifications - [pokemon.py](https://github.com/ralterman/pokemon_classifier/blob/master/pokemon.py                      "Misclassifications")

<p align="center"><img src="https://github.com/ralterman/pokemon_classifier/blob/master/images/misclassifications.png"></p>

* Misclassified Pokémon evidently displayed characteristics similar to the classes they were identified as being, i.e. non-legendary Pokémon having traits more similar to the average legendary Pokémon and vice versa
