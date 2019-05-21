from flask import Flask, request, jsonify, send_file
import json

import pandas as pd
import numpy as np
import pickle

pokemons = pd.read_pickle("./pokemons_with_dummies.pkl")
model  = pickle.load(open('gboost.pkl', 'rb'))
app = Flask(__name__)


def get_pokemon_by_name(pokemon_name):
    return pokemons[pokemons['Name'] == pokemon_name]


def predict_winner(pokemon1_name, pokemon2_name):
    pokemon1 = get_pokemon_by_name(pokemon1_name).reset_index(drop=True)
    pokemon2 = get_pokemon_by_name(pokemon2_name).reset_index(drop=True)
    pokemon1 = pokemon1.add_prefix('pokemon1_')
    pokemon2 = pokemon2.add_prefix('pokemon2_')
    test_data = pd.concat([pokemon1, pokemon2], axis=1, sort=False).drop(columns=['pokemon1_Name','pokemon2_Name'])
    test_data['HP_diff'] = test_data['pokemon1_HP'].sub(test_data['pokemon2_HP'])
    test_data = test_data.drop(columns=['pokemon1_HP', 'pokemon2_HP'])
    test_data['Attack_diff'] = test_data['pokemon1_Attack'].sub(test_data['pokemon2_Attack'])
    test_data = test_data.drop(columns=['pokemon1_Attack', 'pokemon2_Attack'])
    test_data['Defence_diff'] = test_data['pokemon1_Defense'].sub(test_data['pokemon2_Defense'])
    test_data = test_data.drop(columns=['pokemon1_Defense', 'pokemon2_Defense'])
    test_data['sp_atk_diff'] = test_data['pokemon1_Sp. Atk'].sub(test_data['pokemon2_Sp. Atk'])
    test_data = test_data.drop(columns=['pokemon1_Sp. Atk', 'pokemon2_Sp. Atk'])
    test_data['sp_def_diff'] = test_data['pokemon1_Sp. Def'].sub(test_data['pokemon2_Sp. Def'])
    test_data = test_data.drop(columns=['pokemon1_Sp. Def', 'pokemon2_Sp. Def'])
    test_data['speed_diff'] = test_data['pokemon1_Speed'].sub(test_data['pokemon2_Speed'])
    test_data = test_data.drop(columns=['pokemon1_Speed', 'pokemon2_Speed'])
    result = model.predict_proba(test_data)
    return result

def get_pretty_output_for_winner(pokemon1_name, pokemon2_name):
    result_probs = predict_winner(pokemon1_name, pokemon2_name)
    winner_1_probs = result_probs[0][0]
    winner_2_probs = result_probs[0][1]
    if winner_1_probs > winner_2_probs:
        return {'name': pokemon1_name, "probability": "{:.3%}".format(winner_1_probs)}
    else:
        return {'name': pokemon2_name, "probability": "{:.3%}".format(winner_2_probs)}


@app.route('/check_winner', methods=['POST'])
def check_winner():
    json = request.get_json()
    pokemon1_name = json['pokemon_1']
    pokemon2_name = json['pokemon_2']
    if pokemon1_name not in list(pokemons['Name']):
        return jsonify({'error': 'No such pokemon: \'{}\''.format(pokemon1_name)})
    if pokemon2_name not in list(pokemons['Name']):
        return jsonify({'error': 'No such pokemon: \'{}\''.format(pokemon2_name)})
    res = get_pretty_output_for_winner(pokemon1_name, pokemon2_name)
    return jsonify(res)

if __name__ == '__main__':
    app.run()