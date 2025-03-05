import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import ast
from datetime import datetime, timedelta

    """
    This script processes a dataset of user positions and interactions to compute weighted edges between users based on their proximity and parcel scores. It generates dictionaries of edges and users with their respective weights and active times, and saves the results to JSON files.
    
    """

DIR = '/Users/HP/Desktop/UNI/LM_1/ACN/ACN_project/data/'

def calculate_distance_vectorized(positions):
    """
    Calculates pairwise distances for a set of 3D positions.
    Returns a matrix where entry [i, j] is the distance between positions[i] and positions[j].
    """
    diff = positions[:, None, :] - positions[None, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=2))
    return dist_matrix

def get_edges_optimized(df):
    edges_dict = defaultdict(dict)
    
    for date in sorted(df['date'].unique()):
        print(f'Processing date: {date}')
        df_date = df[df['date'] == date]

        # Recalculate parcel scores for the current date
        parcel_score = df_date['parcel'].value_counts().to_dict()
        N = np.sum(list(parcel_score.values()))

        for rounded_time in df_date['rounded_time'].unique():
            df_time = df_date[df_date['rounded_time'] == rounded_time]
            
            # Extract positions, parcels, and addresses as NumPy arrays
            addresses = df_time['address'].values
            parcels = df_time['parcel'].values
            positions = np.array([ast.literal_eval(pos) for pos in df_time['position']])
            
            # Skip if less than 3 users
            if len(positions) <= 2:
                continue

            # Calculate pairwise distances
            dist_matrix = calculate_distance_vectorized(positions)
            
            # Find pairs within distance threshold or at distance 0 (excluding self-loops)
            within_threshold = np.argwhere((dist_matrix <= 10) & (dist_matrix > 0))
            zero_distance_pairs = np.argwhere((dist_matrix == 0) & (np.arange(len(positions))[:, None] != np.arange(len(positions))[None, :]))

            # Combine both conditions
            valid_edges = np.vstack([within_threshold, zero_distance_pairs])
            
            # Sort valid edges to ensure i < j
            valid_edges = valid_edges[valid_edges[:, 0] < valid_edges[:, 1]]

            # Calculate edge weights and store them
            edges_time = []
            for i, j in valid_edges:
                user1, user2 = addresses[i], addresses[j]
                parcel1, parcel2 = parcels[i], parcels[j]
                weight = 2 * N / (parcel_score[parcel1] + parcel_score[parcel2])
                edges_time.append([user1, user2, weight])
            
            # Store results
            edges_dict[date][rounded_time] = edges_time

    return edges_dict


def count_consecutive_groups(windows):
    # Convert string timestamps to datetime objects
    datetime_windows = [datetime.strptime(ts, '%Y-%m-%d-%H:%M') for ts in sorted(windows)]
    
    # Count groups of consecutive timestamps
    num_windows = 1  # At least one group exists
    for i in range(1, len(datetime_windows)):
        if datetime_windows[i] - datetime_windows[i - 1] != timedelta(minutes=1):
            num_windows += 1
    
    return num_windows

def weight_function(T_ij, T_i, F_ij, F_i, parcel_weight):
    return (T_ij/T_i) * (F_ij/F_i) * parcel_weight


if __name__ == '__main__':

    # Load data
    df = pd.read_csv(f'{DIR}final_cleaned_df.csv')

    # Compute edges dictionary
    edges = get_edges_optimized(df)

    edges_dict = {}
    users_dict = {}

    for date, rounded_times in edges.items():
        print(f"Processing date {date}...")
        for rounded_time, edge_list in rounded_times.items():
            active_users = set()
            window_start = f'{date}-{rounded_time}'
            
            for edge in edge_list:
                new_edge = (edge[0], edge[1])
                active_users.add(edge[0])
                active_users.add(edge[1])
                
                # Update edges_dict
                if new_edge in edges_dict:
                    edges_dict[new_edge]['count'] += 1
                    edges_dict[new_edge]['weight'] += edge[2]
                    edges_dict[new_edge]['windows'].append(window_start)
                else:
                    edges_dict[new_edge] = {'count': 1,
                                            'weight': edge[2],
                                            'windows': [window_start]}

            # Update users_dict
            for user in active_users:
                if user in users_dict:
                    users_dict[user]['active_time'] += 1
                    users_dict[user]['windows'].append(window_start)
                else:
                    users_dict[user] = {'active_time': 1,
                                        'windows': [window_start]}

    # Compute number of consecutive windows for edges and users
    for edge, data_dict in edges_dict.items():
        data_dict['windows'] = count_consecutive_groups(data_dict['windows'])

    for user, data_dict in users_dict.items():
        data_dict['windows'] = count_consecutive_groups(data_dict['windows'])

    
    edges_dict_str_keys = {str(key): value for key, value in edges_dict.items()}


    with open(f'{DIR}edges_dict_new.json', "w") as json_file:
        json.dump(edges_dict_str_keys, json_file)

    with open(f'{DIR}users_dict.json', "w") as json_file:
        json.dump(users_dict, json_file)

    for edge, edge_dict in edges_dict.items():
        T_ij = edge_dict.pop('count')
        F_ij = edge_dict.pop('windows')
        parcel_weight = edge_dict.pop('weight')
        T_i, T_j = users_dict[edge[0]]['active_time'], users_dict[edge[1]]['active_time']
        F_i, F_j = users_dict[edge[0]]['windows'], users_dict[edge[1]]['windows']
        edge_dict['weight_user1'] = weight_function(T_ij, T_i, F_ij, F_i, parcel_weight)
        edge_dict['weight_user2'] = weight_function(T_ij, T_j, F_ij, F_j, parcel_weight)
        edge_dict['simmetric_weight'] = T_ij * F_ij * parcel_weight


    edges_dict_weight = {str(key): value for key, value in edges_dict.items()}  

    with open(f'{DIR}edges_dict_weight.json', "w") as json_file:
        json.dump(edges_dict_weight, json_file)

    