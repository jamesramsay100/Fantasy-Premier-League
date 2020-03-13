from tqdm import tqdm
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import glob

from catboost import CatBoostRegressor

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import (train_test_split, GridSearchCV)
from sklearn.model_selection import cross_val_score


class UpdateTeam:
    def __init__(self, prev_team):
        PATH = 'C:/Users/rajxxk/OneDrive - BP/Documents/Personal/Projects/Fantasy-Premier-League/'
        self.prev_team = pd.read_csv(PATH + prev_team)
        self.fixtures = pd.read_csv(PATH + 'data/2019-20/fixtures.csv')
        self.teams = pd.read_csv(PATH + 'data/2019-20/teams.csv')
        self.players_raw = pd.read_csv(PATH + 'data/2019-20/players_raw.csv')
        self.next_team = self.prev_team

        self.gw_all_players = pd.DataFrame()
        print('Loading latest fixtures and player data...')
        for filename in tqdm(glob.glob(PATH + 'data/2019-20/players/**/gw.csv', recursive=True)):
            gw = pd.read_csv(filename)
            gw['player_name'] = filename.split("\\")[-2].split("_")[0] + " " + filename.split("\\")[-2].split("_")[1]
            gw['player_ID'] = int(filename.split("\\")[-2].split("_")[-1])
            self.gw_all_players = self.gw_all_players.append(gw)

        cols = ['id', 'chance_of_playing_next_round', 'team', 'points_per_game', 'element_type']
        self.gw_all_players = self.gw_all_players.merge(self.players_raw[cols],
                                                        how='left',
                                                        left_on='player_ID',
                                                        right_on='id')
        # change change of playing to numeric
        self.gw_all_players['chance_of_playing_next_round'] = \
            self.gw_all_players['chance_of_playing_next_round'].replace({'None': '0'}).astype('float')/100

        self.next_gw = self.gw_all_players['round'].max() + 1
        print('Next game week : {}'.format(self.next_gw))

        self.model = None
        self.gw_predictions = pd.DataFrame()

        self.best_features = ['points_per_game(t-5)', 'threat(t-1)',
                              'transfers_in(t-1)', 'transfers_balance(t-1)', 'was_home(t)',
                              'round(t)', 'ict_index(t-1)', 'player_name(t)',
                              'influence(t-1)', 'opponent_team(t)', 'points_per_game(t-3)',
                              'points_per_game(t-2)', 'round(t-1)', 'minutes(t-1)',
                              'total_points(t-1)', 'total_points(t-2)', 'total_points(t-3)', 'total_points(t-4)',
                              'points_per_game(t-1)', 'element_type(t)', 'value(t)', 'team(t)']

        self.transfer_options = pd.DataFrame()

    def update_model(self, max_iter=100):
        # create lag features
        lag_features = series_to_supervised(self.gw_all_players, 5)

        # change floats to ints
        lag_features['opponent_team(t)'] = lag_features['opponent_team(t)'].astype('int')
        lag_features['element_type(t)'] = lag_features['element_type(t)'].astype('int')
        lag_features['team(t)'] = lag_features['team(t)'].astype('int')

        best_features = self.best_features

        # create training and test datasets
        n = self.next_gw - 1
        X_train = lag_features[lag_features['round(t)'] < (n)][best_features]
        X_test = lag_features[lag_features['round(t)'] == (n)][best_features]
        y_train = lag_features[lag_features['round(t)'] < (n)]['total_points(t)']
        y_test = lag_features[lag_features['round(t)'] == (n)]['total_points(t)']

        cat_features = [7, 9, 19, 21]

        self.model = CatBoostRegressor(iterations=max_iter,
                                       learning_rate=0.05,
                                       depth=6)

        self.model.fit(
            X_train,
            y_train,
            cat_features,
            eval_set=[(X_test, y_test)],
            plot=True
        )

        print('Model accuracy : {}'.format(mean_absolute_error(self.model.predict(X_test), y_test)))

    def make_gw_preds(self):

        gw_target = self.next_gw

        lag_only = series_to_supervised(self.gw_all_players, 5, 0)
        lag_only = lag_only[lag_only['round(t-1)'] == (gw_target - 1)]

        lag_only['player_name(t)'] = lag_only['player_name(t-1)']
        lag_only['element_type(t)'] = lag_only['element_type(t-1)']
        lag_only['value(t)'] = lag_only['value(t-1)']
        lag_only['team(t)'] = lag_only['team(t-1)']
        lag_only['round(t)'] = gw_target

        lag_only['opponent_team(t)'] = 0
        lag_only['was_home(t)'] = False

        for index, row in lag_only.iterrows():
            team = row['team(t)']
            try:
                a = \
                self.fixtures[(self.fixtures['event'] == gw_target) & (self.fixtures['team_h'] == team)]['team_a'].iloc[
                    0]
                lag_only.loc[index, 'opponent_team(t)'] = a
                lag_only.loc[index, 'was_home(t)'] = True
            except:
                pass

            try:
                a = \
                self.fixtures[(self.fixtures['event'] == gw_target) & (self.fixtures['team_a'] == team)]['team_h'].iloc[
                    0]
                lag_only.loc[index, 'opponent_team(t)'] = a
            except:
                pass

        # change floats to ints so models accepts as categorical features
        lag_only['opponent_team(t)'] = lag_only['opponent_team(t)'].astype('int')
        lag_only['element_type(t)'] = lag_only['element_type(t)'].astype('int')
        lag_only['team(t)'] = lag_only['team(t)'].astype('int')

        pred_input = lag_only[self.best_features].copy()

        self.gw_predictions = pd.DataFrame()
        self.gw_predictions['names'] = pred_input['player_name(t)'].values
        self.gw_predictions['prices'] = (pred_input['value(t)'] / 10).values
        self.gw_predictions['positions'] = pred_input['element_type(t)'].values
        self.gw_predictions['clubs'] = pred_input['team(t)'].values

        self.gw_predictions['model'] = self.model.predict(pred_input).astype(int)  # predicted points for GW12
        self.gw_predictions['5_wk_avg'] = lag_only[
            ['total_points(t-1)', 'total_points(t-2)', 'total_points(t-3)', 'total_points(t-4)']].mean(
            axis=1).reset_index(drop=True)
        self.gw_predictions = self.gw_predictions.merge(right=lag_only[['chance_of_playing_next_round(t-1)', 'player_name(t)']],
                                                        left_on='names',
                                                        right_on='player_name(t)')

        self.gw_predictions['prediction'] = (self.gw_predictions['model'] + 2 * self.gw_predictions[
            '5_wk_avg']) / 3
        self.gw_predictions['prediction'] = self.gw_predictions['prediction'] * self.gw_predictions['chance_of_playing_next_round(t-1)']



        # print(self.gw_predictions.sort_values(by='avg_pred', ascending=False).head(10))

    def gen_transfer_options(self, in_bank=0):

        self.prev_team = self.prev_team.merge(self.gw_predictions[['names', 'prediction']],
                                              left_on='Names',
                                              right_on='names')

        for index, row in self.prev_team.iterrows():
            top_row = self.gw_predictions[(self.gw_predictions['positions'] == row['element_type']) &
                                          (self.gw_predictions['prices'] <= row['value'] + in_bank) &
                                          # (self.gw_predictions['names'] != row['Names'])
                                          (~self.gw_predictions['names'].isin(self.prev_team['Names']))
                                          ].sort_values(by=['prediction'], ascending=False).head(1)
            self.transfer_options = self.transfer_options.append(top_row)

        self.transfer_options.reset_index(drop=True, inplace=True)
        self.transfer_options['original player'] = self.prev_team['Names'].reset_index(drop=True)
        self.transfer_options['original pts pred'] = self.prev_team['prediction'].reset_index(drop=True)
        self.transfer_options['original price'] = self.prev_team['value'].reset_index(drop=True)
        self.transfer_options['pts - price'] = (self.transfer_options['prediction'] - self.transfer_options['original pts pred']) - \
                                               (self.transfer_options['prices'] - self.transfer_options['original price'] / 10)

        self.transfer_options = self.transfer_options[['original player',
                                                       'original pts pred',
                                                       'original price',
                                                       'names',
                                                       'prices',
                                                       'prediction',
                                                       'pts - price'
                                                       ]].copy()


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        # names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        names += [(str(data.columns[j]) + '(t-%d)' % (i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [(str(data.columns[j]) + '(t)') for j in range(n_vars)]
        else:
            names += [(str(data.columns[j]) + '(t-%d)' % (i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


if __name__ == '__main__':
    test = UpdateTeam('my_teams/team_gw27.csv')
    test.update_model(max_iter=200)
    test.make_gw_preds()
    test.gen_transfer_options(in_bank=1.6)
