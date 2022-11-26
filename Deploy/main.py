import pickle
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, url_for



app = Flask(__name__)

le = pickle.load(open('labelencoder.pkl', 'rb'))
data = pd.read_csv('model_prepped_dataset_modified.csv')


def extract_data_outome_prediction(home_team_name, away_team_name, home_elo, away_elo, data):

    Last_5_Home_Team_avgGoal = 0
    df = data[(data['Home_Team'] == home_team_name) |
              (data['Away_Team'] == home_team_name)]
    df = df[len(df) - 5:]
    for i in range(len(df)):
        Last_5_Home_Team_avgGoal += df.iloc[i]['Home_Team_Goal']
    Last_5_Home_Team_avgGoal /= 5

    Last_5_Away_Team_avgGoal = 0
    df = data[(data['Home_Team'] == away_team_name) |
              (data['Away_Team'] == away_team_name)]
    df = df[len(df) - 5:]
    for i in range(len(df)):
        Last_5_Away_Team_avgGoal += df.iloc[i]['Away_Team_Goal']
    Last_5_Away_Team_avgGoal /= 5

    Last_5_Home_Team_Home_avgGoal = 0
    df = data[data['Home_Team'] == home_team_name]
    df = df[len(df) - 5:]
    for i in range(len(df)):
        Last_5_Home_Team_Home_avgGoal += df.iloc[i]['Home_Team_Goal']
    Last_5_Home_Team_Home_avgGoal /= 5

    Last_5_Away_Team_Away_avgGoal = 0
    df = data[data['Away_Team'] == away_team_name]
    df = df[len(df) - 5:]
    for i in range(len(df)):
        Last_5_Away_Team_Away_avgGoal += df.iloc[i]['Away_Team_Goal']
    Last_5_Away_Team_Away_avgGoal /= 5

    Last_5_Home_Team_All_Streak = 0
    df = data[(data['Home_Team'] == home_team_name) |
              (data['Away_Team'] == home_team_name)]
    df = df[len(df) - 5:]
    for i in range(len(df)):
        Last_5_Home_Team_All_Streak += df.iloc[i]['Outcome']

    Last_5_Away_Team_All_Streak = 0
    df = data[(data['Home_Team'] == away_team_name) |
              (data['Away_Team'] == away_team_name)]
    df = df[len(df) - 5:]
    for i in range(len(df)):
        Last_5_Away_Team_All_Streak += df.iloc[i]['Outcome']

    Last_5_Home_Team_Home_Streak = 0
    df = data[(data['Home_Team'] == home_team_name)]
    df = df[len(df) - 5:]
    for i in range(len(df)):
        Last_5_Home_Team_Home_Streak += df.iloc[i]['Outcome']

    Last_5_Away_Team_Away_Streak = 0
    df = data[(data['Away_Team'] == away_team_name)]
    df = df[len(df) - 5:]
    for i in range(len(df)):
        Last_5_Away_Team_Away_Streak += df.iloc[i]['Outcome']

    Last_3_same_team_home_goal = 0
    df = data[(data['Home_Team'] == home_team_name) &
              (data['Away_Team'] == away_team_name)]
    df = df[len(df) - 3:]
    for i in range(len(df)):
        Last_3_same_team_home_goal += df.iloc[i]['Home_Team_Goal']
    Last_3_same_team_home_goal /= 3

    Last_3_same_team_away_goal = 0
    df = data[(data['Home_Team'] == home_team_name) &
              (data['Away_Team'] == away_team_name)]
    df = df[len(df) - 3:]
    for i in range(len(df)):
        Last_3_same_team_away_goal += df.iloc[i]['Away_Team_Goal']
    Last_3_same_team_away_goal /= 3

    Last_3_same_team_outcome = 0
    df = data[(data['Home_Team'] == home_team_name) &
              (data['Away_Team'] == away_team_name)]
    df = df[len(df) - 3:]
    for i in range(len(df)):
        Last_3_same_team_outcome += df.iloc[i]['Outcome']

    Home_Team_Points = 0
    df = data[data['Home_Team'] == home_team_name]
    Home_Team_Points = df.iloc[-1]['Home_Team_Points']

    Away_Team_Points = 0
    df = data[data['Away_Team'] == away_team_name]
    Away_Team_Points = df.iloc[-1]['Away_Team_Points']

    return [
        home_team_name, away_team_name, home_elo, away_elo,
        Last_5_Home_Team_avgGoal, Last_5_Away_Team_avgGoal, Last_5_Home_Team_Home_avgGoal, Last_5_Away_Team_Away_avgGoal,
        Last_5_Home_Team_All_Streak, Last_5_Away_Team_All_Streak, Last_5_Home_Team_Home_Streak, Last_5_Away_Team_Away_Streak,
        Last_3_same_team_home_goal, Last_3_same_team_away_goal, Last_3_same_team_outcome,
        Home_Team_Points, Away_Team_Points
    ]


def extract_data_GD_prediction(home_team_name, away_team_name, home_elo, away_elo, data):

    Last_5_Home_Team_avgGoal = 0
    df = data[(data['Home_Team'] == home_team_name) |
              (data['Away_Team'] == home_team_name)]
    df = df[len(df) - 5:]
    for i in range(len(df)):
        Last_5_Home_Team_avgGoal += df.iloc[i]['Home_Team_Goal']
    Last_5_Home_Team_avgGoal /= 5

    Last_5_Away_Team_avgGoal = 0
    df = data[(data['Home_Team'] == away_team_name) |
              (data['Away_Team'] == away_team_name)]
    df = df[len(df) - 5:]
    for i in range(len(df)):
        Last_5_Away_Team_avgGoal += df.iloc[i]['Away_Team_Goal']
    Last_5_Away_Team_avgGoal /= 5

    Last_5_Home_Team_Home_avgGoal = 0
    df = data[data['Home_Team'] == home_team_name]
    df = df[len(df) - 5:]
    for i in range(len(df)):
        Last_5_Home_Team_Home_avgGoal += df.iloc[i]['Home_Team_Goal']
    Last_5_Home_Team_Home_avgGoal /= 5

    Last_5_Away_Team_Away_avgGoal = 0
    df = data[data['Away_Team'] == away_team_name]
    df = df[len(df) - 5:]
    for i in range(len(df)):
        Last_5_Away_Team_Away_avgGoal += df.iloc[i]['Away_Team_Goal']
    Last_5_Away_Team_Away_avgGoal /= 5

    Last_5_Home_Team_All_avgGD = 0
    df = data[(data['Home_Team'] == home_team_name) |
              (data['Away_Team'] == home_team_name)]
    df = df[len(df) - 5:]
    for i in range(len(df)):
        Last_5_Home_Team_All_avgGD += df.iloc[i]['Home_Team_Goal'] - \
            df.iloc[i]['Away_Team_Goal']
    Last_5_Home_Team_All_avgGD /= 5

    Last_5_Away_Team_All_avgGD = 0
    df = data[(data['Home_Team'] == away_team_name) |
              (data['Away_Team'] == away_team_name)]
    df = df[len(df) - 5:]
    for i in range(len(df)):
        Last_5_Away_Team_All_avgGD += df.iloc[i]['Home_Team_Goal'] - \
            df.iloc[i]['Away_Team_Goal']
    Last_5_Away_Team_All_avgGD /= 5

    Last_5_Home_Team_Home_avgGD = 0
    df = data[data['Home_Team'] == home_team_name]
    df = df[len(df) - 5:]
    for i in range(len(df)):
        Last_5_Home_Team_Home_avgGD += df.iloc[i]['Home_Team_Goal'] - \
            df.iloc[i]['Away_Team_Goal']
    Last_5_Home_Team_Home_avgGD /= 5

    Last_5_Away_Team_Away_avgGD = 0
    df = data[data['Away_Team'] == away_team_name]
    df = df[len(df) - 5:]
    for i in range(len(df)):
        Last_5_Away_Team_Away_avgGD += df.iloc[i]['Home_Team_Goal'] - \
            df.iloc[i]['Away_Team_Goal']

    Last_3_same_team_home_avgGD = 0
    df = data[(data['Home_Team'] == home_team_name) &
              (data['Away_Team'] == away_team_name)]
    df = df[len(df) - 3:]
    for i in range(len(df)):
        Last_3_same_team_home_avgGD += df.iloc[i]['Home_Team_Goal'] - \
            df.iloc[i]['Away_Team_Goal']
    Last_3_same_team_home_avgGD /= 3

    Last_3_same_team_away_avgGD = 0
    df = data[(data['Home_Team'] == away_team_name) &
              (data['Away_Team'] == home_team_name)]
    df = df[len(df) - 3:]
    for i in range(len(df)):
        Last_3_same_team_away_avgGD += df.iloc[i]['Home_Team_Goal'] - \
            df.iloc[i]['Away_Team_Goal']
    Last_3_same_team_away_avgGD /= 3

    Last_3_same_team_avgGD = 0
    df = data[(data['Home_Team'] == home_team_name) & (data['Away_Team'] == away_team_name) | (
        data['Home_Team'] == away_team_name) & (data['Away_Team'] == home_team_name)]
    df = df[len(df) - 3:]
    for i in range(len(df)):
        Last_3_same_team_avgGD += df.iloc[i]['Home_Team_Goal'] - \
            df.iloc[i]['Away_Team_Goal']
    Last_3_same_team_avgGD /= 3

    Home_Team_Points = 0
    df = data[data['Home_Team'] == home_team_name]
    Home_Team_Points = df.iloc[-1]['Home_Team_Points']

    Away_Team_Points = 0
    df = data[data['Away_Team'] == away_team_name]
    Away_Team_Points = df.iloc[-1]['Away_Team_Points']

    return [
        home_team_name, away_team_name, home_elo, away_elo,
        Last_5_Home_Team_avgGoal, Last_5_Away_Team_avgGoal, Last_5_Home_Team_Home_avgGoal, Last_5_Away_Team_Away_avgGoal,
        Last_5_Home_Team_All_avgGD, Last_5_Away_Team_All_avgGD, Last_5_Home_Team_Home_avgGD, Last_5_Away_Team_Away_avgGD,
        Last_3_same_team_home_avgGD, Last_3_same_team_away_avgGD, Last_3_same_team_avgGD,
        Home_Team_Points, Away_Team_Points
    ]


def extract_data_goal_prediction(home_team_name, away_team_name, home_elo, away_elo, data):

    Last_5_Home_Team_avgGoal = 0
    df = data[(data['Home_Team'] == home_team_name) |
              (data['Away_Team'] == home_team_name)]
    df = df[len(df) - 5:]
    for i in range(len(df)):
        Last_5_Home_Team_avgGoal += df.iloc[i]['Home_Team_Goal']
    Last_5_Home_Team_avgGoal /= 5

    Last_5_Away_Team_avgGoal = 0
    df = data[(data['Home_Team'] == away_team_name) |
              (data['Away_Team'] == away_team_name)]
    df = df[len(df) - 5:]
    for i in range(len(df)):
        Last_5_Away_Team_avgGoal += df.iloc[i]['Away_Team_Goal']
    Last_5_Away_Team_avgGoal /= 5

    Last_5_Home_Team_Home_avgGoal = 0
    df = data[data['Home_Team'] == home_team_name]
    df = df[len(df) - 5:]
    for i in range(len(df)):
        Last_5_Home_Team_Home_avgGoal += df.iloc[i]['Home_Team_Goal']
    Last_5_Home_Team_Home_avgGoal /= 5

    Last_5_Away_Team_Away_avgGoal = 0
    df = data[data['Away_Team'] == away_team_name]
    df = df[len(df) - 5:]
    for i in range(len(df)):
        Last_5_Away_Team_Away_avgGoal += df.iloc[i]['Away_Team_Goal']
    Last_5_Away_Team_Away_avgGoal /= 5

    Last_5_Home_Team_All_Streak = 0
    df = data[(data['Home_Team'] == home_team_name) |
              (data['Away_Team'] == home_team_name)]
    df = df[len(df) - 5:]
    for i in range(len(df)):
        Last_5_Home_Team_All_Streak += df.iloc[i]['Outcome']

    Last_5_Away_Team_All_Streak = 0
    df = data[(data['Home_Team'] == away_team_name) |
              (data['Away_Team'] == away_team_name)]
    df = df[len(df) - 5:]
    for i in range(len(df)):
        Last_5_Away_Team_All_Streak += df.iloc[i]['Outcome']

    Last_5_Home_Team_Home_Streak = 0
    df = data[(data['Home_Team'] == home_team_name)]
    df = df[len(df) - 5:]
    for i in range(len(df)):
        Last_5_Home_Team_Home_Streak += df.iloc[i]['Outcome']

    Last_5_Away_Team_Away_Streak = 0
    df = data[(data['Away_Team'] == away_team_name)]
    df = df[len(df) - 5:]
    for i in range(len(df)):
        Last_5_Away_Team_Away_Streak += df.iloc[i]['Outcome']

    Last_3_same_team_home_goal = 0
    df = data[(data['Home_Team'] == home_team_name) &
              (data['Away_Team'] == away_team_name)]
    df = df[len(df) - 3:]
    for i in range(len(df)):
        Last_3_same_team_home_goal += df.iloc[i]['Home_Team_Goal']
    Last_3_same_team_home_goal /= 3

    Last_3_same_team_away_goal = 0
    df = data[(data['Home_Team'] == home_team_name) &
              (data['Away_Team'] == away_team_name)]
    df = df[len(df) - 3:]
    for i in range(len(df)):
        Last_3_same_team_away_goal += df.iloc[i]['Away_Team_Goal']
    Last_3_same_team_away_goal /= 3

    Last_3_same_team_outcome = 0
    df = data[(data['Home_Team'] == home_team_name) &
              (data['Away_Team'] == away_team_name)]
    df = df[len(df) - 3:]
    for i in range(len(df)):
        Last_3_same_team_outcome += df.iloc[i]['Outcome']

    Home_Team_Points = 0
    df = data[data['Home_Team'] == home_team_name]
    Home_Team_Points = df.iloc[-1]['Home_Team_Points']

    Away_Team_Points = 0
    df = data[data['Away_Team'] == away_team_name]
    Away_Team_Points = df.iloc[-1]['Away_Team_Points']

    return [
        home_team_name, away_team_name, home_elo, away_elo,
        Last_5_Home_Team_avgGoal, Last_5_Away_Team_avgGoal, Last_5_Home_Team_Home_avgGoal, Last_5_Away_Team_Away_avgGoal,
        Last_5_Home_Team_All_Streak, Last_5_Away_Team_All_Streak, Last_5_Home_Team_Home_Streak, Last_5_Away_Team_Away_Streak,
        Last_3_same_team_home_goal, Last_3_same_team_away_goal, Last_3_same_team_outcome,
        Home_Team_Points, Away_Team_Points
    ]


@app.route('/')
def index():
    return render_template('index.html', data="HELLO")


@app.route("/predict", methods=['POST'])
def predict():
    Home_Team_Name = request.form['Home_Team_Name']
    Away_Team_Name = request.form['Away_Team_Name']
    Home_Team_ELO = int(request.form['Home_Team_ELO'])
    Away_Team_ELO = int(request.form['Away_Team_ELO'])
    Capital = float(request.form['Capital'])
    choice = int(request.form['Choice'])

    if Home_Team_Name == Away_Team_Name:
        return render_template('prediction.html', data="Please enter different teams")

    if choice == 1:
        model = pickle.load(
            open('Bset_Model_Outcome_Prediction_SVC.pkl', 'rb'))
        d = extract_data_outome_prediction(
            Home_Team_Name, Away_Team_Name, Home_Team_ELO, Away_Team_ELO, data)
        d[0] = le.transform([Home_Team_Name])[0]
        d[1] = le.transform([Away_Team_Name])[0]
        outcome = model.predict([d])[0]
        if outcome == 1:
            outcome = f"win"
        elif outcome == -1:
            outcome = f"lose"
        else:
            outcome = f"Draw"

        probability = model.predict_proba([d])[0]
        probability = [round(x, 2) for x in probability]
        odds = []
        for i in range(len(probability)):
            odds.append(round(probability[i] / (1 - probability[i]), 2))

        odds_of_lose = odds[0]
        odds_of_draw = odds[1]
        odds_of_win = odds[2]

        betting_amount = abs(round(
            Capital * (max(probability) * max(odds) - (1 - max(probability))) / max(odds), 2))

        profit = abs(round(betting_amount * max(odds) - betting_amount, 2))

        return render_template(
            "prediction1.html",
            data=outcome,
            Home_Team_Name=Home_Team_Name,
            Away_Team_Name=Away_Team_Name,
            proba_of_lose=probability[0],
            proba_of_draw=probability[1],
            proba_of_win=probability[2],
            odds_of_lose=odds_of_lose,
            odds_of_draw=odds_of_draw,
            odds_of_win=odds_of_win,
            betting_amount=betting_amount,
            profit=profit
        )

    if choice == 2:
        model = pickle.load(
            open('Best_Model_GD_Prediction_AdaBoost.pkl', 'rb'))
        d = extract_data_GD_prediction(
            Home_Team_Name, Away_Team_Name, Home_Team_ELO, Away_Team_ELO, data)
        d[0] = le.transform([Home_Team_Name])[0]
        d[1] = le.transform([Away_Team_Name])[0]
        gd = model.predict([d])[0]
        probability = model.predict_proba([d])[0]
        probability = [round(x, 2) for x in probability]
        odds = []
        for i in range(len(probability)):
            odds.append(round(probability[i] / (1 - probability[i]), 2))

        betting_amount = abs(round(
            Capital * (max(probability) * max(odds) - (1 - max(probability))) / max(odds), 2))

        profit = abs(round(betting_amount * max(odds) - betting_amount, 2))

        return render_template(
            'prediction2.html',
            Home_Team_Name=Home_Team_Name,
            Away_Team_Name=Away_Team_Name,
            gd=gd,
            proba_of_neg3=probability[0],
            proba_of_neg2=probability[1],
            proba_of_neg1=probability[2],
            proba_of_0=probability[3],
            proba_of_1=probability[4],
            proba_of_2=probability[5],
            proba_of_3=probability[6],
            proba_of_4=probability[7],
            proba_of_5=probability[8],
            odds_of_neg3=odds[0],
            odds_of_neg2=odds[1],
            odds_of_neg1=odds[2],
            odds_of_0=odds[3],
            odds_of_1=odds[4],
            odds_of_2=odds[5],
            odds_of_3=odds[6],
            odds_of_4=odds[7],
            odds_of_5=odds[8],
            betting_amount=betting_amount,
            profit=profit
        )

    if choice == 3:
        model_home = pickle.load(
            open('Best_Model_Goal_Prediction_LogisticRegression_Home.pkl', 'rb'))
        model_away = pickle.load(
            open('Best_Model_Goal_Prediction_LogisticRegression_Away.pkl', 'rb'))

        d = extract_data_goal_prediction(
            Home_Team_Name, Away_Team_Name, Home_Team_ELO, Away_Team_ELO, data)
        d[0] = le.transform([Home_Team_Name])[0]
        d[1] = le.transform([Away_Team_Name])[0]

        home_team_goal = model_home.predict([d])[0]
        away_team_goal = model_away.predict([d])[0]

        home_team_goals_proba = model_home.predict_proba([d])[0]
        away_team_goals_proba = model_away.predict_proba([d])[0]

        home_team_goals_proba = [round(x, 2) for x in home_team_goals_proba]
        away_team_goals_proba = [round(x, 2) for x in away_team_goals_proba]

        home_team_goals_odds = []
        for i in range(len(home_team_goals_proba)):
            home_team_goals_odds.append(
                round(home_team_goals_proba[i] / (1 - home_team_goals_proba[i]), 2))

        away_team_goals_odds = []
        for i in range(len(away_team_goals_proba)):
            away_team_goals_odds.append(
                round(away_team_goals_proba[i] / (1 - away_team_goals_proba[i]), 2))

        mat = np.zeros((6, 6))

        for i in range(len(home_team_goals_proba)):
            for j in range(len(away_team_goals_proba)):
                p = home_team_goals_proba[i] * away_team_goals_proba[j]
                mat[i][j] = round(p, 2)

        betting_amount_home = abs(round(Capital * (max(home_team_goals_proba) * max(
            home_team_goals_odds) - (1 - max(home_team_goals_proba))) / max(home_team_goals_odds), 2))
        betting_amount_away = abs(round(Capital * (max(away_team_goals_proba) * max(
            away_team_goals_odds) - (1 - max(away_team_goals_proba))) / max(away_team_goals_odds), 2))

        profit_home = abs(round(betting_amount_home *
                          max(home_team_goals_odds) - betting_amount_home, 2))
        profit_away = abs(round(betting_amount_away *
                          max(away_team_goals_odds) - betting_amount_away, 2))

        return render_template(
            'prediction3.html',
            Home_Team_Name=Home_Team_Name,
            Away_Team_Name=Away_Team_Name,
            home_team_goal=home_team_goal,
            away_team_goal=away_team_goal,

            home_proba_of_0goal=home_team_goals_proba[0],
            home_proba_of_1goal=home_team_goals_proba[1],
            home_proba_of_2goal=home_team_goals_proba[2],
            home_proba_of_3goal=home_team_goals_proba[3],
            home_proba_of_4goal=home_team_goals_proba[4],
            home_proba_of_5goal=home_team_goals_proba[5],

            away_proba_of_0goal=away_team_goals_proba[0],
            away_proba_of_1goal=away_team_goals_proba[1],
            away_proba_of_2goal=away_team_goals_proba[2],
            away_proba_of_3goal=away_team_goals_proba[3],
            away_proba_of_4goal=away_team_goals_proba[4],
            away_proba_of_5goal=away_team_goals_proba[5],

            home_odds_of_0goal=home_team_goals_odds[0],
            home_odds_of_1goal=home_team_goals_odds[1],
            home_odds_of_2goal=home_team_goals_odds[2],
            home_odds_of_3goal=home_team_goals_odds[3],
            home_odds_of_4goal=home_team_goals_odds[4],
            home_odds_of_5goal=home_team_goals_odds[5],

            away_odds_of_0goal=away_team_goals_odds[0],
            away_odds_of_1goal=away_team_goals_odds[1],
            away_odds_of_2goal=away_team_goals_odds[2],
            away_odds_of_3goal=away_team_goals_odds[3],
            away_odds_of_4goal=away_team_goals_odds[4],
            away_odds_of_5goal=away_team_goals_odds[5],

            goal_00=mat[0][0],
            goal_01=mat[0][1],
            goal_02=mat[0][2],
            goal_03=mat[0][3],
            goal_04=mat[0][4],
            goal_05=mat[0][5],
            goal_10=mat[1][0],
            goal_11=mat[1][1],
            goal_12=mat[1][2],
            goal_13=mat[1][3],
            goal_14=mat[1][4],
            goal_15=mat[1][5],
            goal_20=mat[2][0],
            goal_21=mat[2][1],
            goal_22=mat[2][2],
            goal_23=mat[2][3],
            goal_24=mat[2][4],
            goal_25=mat[2][5],
            goal_30=mat[3][0],
            goal_31=mat[3][1],
            goal_32=mat[3][2],
            goal_33=mat[3][3],
            goal_34=mat[3][4],
            goal_35=mat[3][5],
            goal_40=mat[4][0],
            goal_41=mat[4][1],
            goal_42=mat[4][2],
            goal_43=mat[4][3],
            goal_44=mat[4][4],
            goal_45=mat[4][5],
            goal_50=mat[5][0],
            goal_51=mat[5][1],
            goal_52=mat[5][2],
            goal_53=mat[5][3],
            goal_54=mat[5][4],
            goal_55=mat[5][5],

            betting_amount_home=betting_amount_home,
            betting_amount_away=betting_amount_away,

            profit_home=profit_home,
            profit_away=profit_away
        )


if __name__ == '__main__':
    app.run(debug=True)
