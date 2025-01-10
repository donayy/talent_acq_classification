# Talent Acqusition Classification
Prediction of which class (average, highlighted) a player belongs to based on the ratings of players' attributes observed by scouts.

The dataset contains information about players' attributes observed during matches and the ratings given by scouts for those attributes. 
It includes the attributes scored during the match and their respective ratings.

attributes.csv
8 Variables - 10,730 Observations
task_response_id: A set of evaluations for all players in a team's roster for a match by a scout
match_id: The ID of the relevant match
evaluator_id: The ID of the evaluator (scout)
player_id: The ID of the relevant player
position_id: The ID of the position the player played in the match
1: Goalkeeper
2: Center-back
3: Right-back
4: Left-back
5: Defensive midfielder
6: Central midfielder
7: Right wing
8: Left wing
9: Attacking midfielder
10: Forward
analysis_id: A set of evaluations for a player’s attributes by a scout during a match
attribute_id: The ID of each attribute the players are rated on
attribute_value: The score a scout gives to a player's attribute

potential_labels.csv
5 Variables - 322 Observations
task_response_id: A set of evaluations for all players in a team's roster for a match by a scout
match_id: The ID of the relevant match
evaluator_id: The ID of the evaluator (scout)
player_id: The ID of the relevant player
potential_label: The final label indicating the scout’s decision about a player during a match (target variable)
