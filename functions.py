import pandas as pd
import json
import numpy as np
import os
import re
import time
from matplotlib import pyplot as plt
import ast
from PIL import Image
import datetime

def get_distance(position1, position2):
    """
    Calculates the 2D Euclidean distance between two points (position1 and position2) in the x-z plane.

    Parameters:
    - position1, position2: Coordinates as lists or tuples [x, y, z].

    Returns:
    - float: The Euclidean distance between the two positions in the x-z plane, ignoring the y-axis.
    """
    
    return np.sqrt((position1[0] - position2[0])**2 + (position1[2] - position2[2])**2)


def get_edges(df, date, radius):

    """
    Identifies pairs of users ("edges") within a specified distance radius at each time step for a given date.

    Parameters:
    - df: DataFrame with columns 'id', 'date', 'rounded_time', 'position'.
    - date: Date to filter data.
    - radius: Maximum distance between users to create an edge.

    Returns:
    - Dictionary where keys are time steps and values are lists of user pairs within the radius.
    """

    # Filter the DataFrame for rows matching the specified date
    df = df[df['date'] == date]

    # Get a list of unique rounded times for the given date
    rounded_times = df['rounded_time'].unique()
    edges_dict = {}  # Initialize an empty dictionary to store edges for each time

    # Iterate through each unique time to evaluate connections at that time
    for time in rounded_times:
        print(f'Time = {time}')
        
        # Filter the DataFrame to include only entries for the current time
        df_tmp = df[df['rounded_time'] == time]
        
        # Extract unique player IDs active at the current time
        unique_ids = df_tmp['id'].unique()
        print(f'Active players: {len(unique_ids)}')
        
        edges_tmp = []  # Temporary list to store edges for the current time

        # Skip if less than 2 players are active (no possible edges)
        if len(unique_ids) < 2:
            pass
        
        # Iterate over pairs of unique player IDs
        for i in range(len(unique_ids)):
            for j in range(i + 1, len(unique_ids)):
                
                # Get positions for the two players and calculate their distance
                pos1 = df_tmp[df_tmp['id'] == unique_ids[i]]['position'].tolist()[0]
                pos2 = df_tmp[df_tmp['id'] == unique_ids[j]]['position'].tolist()[0]
                
                # If distance is below the specified radius, create an edge
                if get_distance(pos1, pos2) < radius:
                    edges_tmp.append([unique_ids[i], unique_ids[j]])

        print(f'Edges created: {len(edges_tmp)}\n')
        
        # Add the edges list for the current time to the dictionary
        edges_dict[time] = edges_tmp

    # Return dictionary containing all edges organized by time
    return edges_dict

import requests
import time

def fetch_wearables_from_user(user_id):
    """
    Fetches the list of wearables owned by a user in Decentraland using the specified user ID.

    Parameters:
    - user_id: The ID of the user whose wearables are being fetched.

    Returns:
    - JSON response containing wearable data if successful, otherwise None.
    """

    # Define the endpoint URL with the user ID and includeDefinitions parameter
    url = f"https://peer.decentraland.org/lambdas/collections/wearables-by-owner/{user_id}?includeDefinitions"
    
    # Set authorization headers if required for access
    headers = {
        'Authorization': 'Bearer YOUR_ACCESS_TOKEN'
    }
    
    # Send a GET request to the endpoint
    response = requests.get(url, headers=headers)
    
    # If request is successful, parse the JSON response
    if response.status_code == 200:
        return response.json()
    else:
        # Print error message if request fails
        print(f"Error fetching data: {response.status_code}")
        return None

    
def fetch_wearables(addresses):
    """
    Fetches wearable data for multiple blockchain addresses from Decentraland.

    Parameters:
    - addresses: A list of blockchain addresses to fetch wearable data for.

    Returns:
    - Dictionary where each address maps to a list of wearables with details like ID, name, description, etc.
    """

    # Initialize dictionary to store wearables data by address
    wearables = {}
    
    # Iterate over each address in the provided list
    for address in addresses:
        print(f'Fetching address {address} wearables:', end='')
        t0 = time.time()  # Track start time
        
        # Fetch wearables for the current address
        tmp = fetch_wearables_from_user(address)
        tmp_wearables = []
        
        # If the fetch is successful, process each wearable item
        if tmp is not None:
            for elem in tmp:
                elem_definition = elem.get('definition', {})
                tmp_wearables.append({
                    'id': elem_definition.get('id', ''),
                    'name': elem_definition.get('name', ''),
                    'description': elem_definition.get('description', ''),
                    'collectionAddress': elem_definition.get('collectionAddress', None),
                    'rarity': elem_definition.get('rarity', ''),
                    'tags': elem_definition.get('data', {}).get('tags', []),
                    'category': elem_definition.get('data', {}).get('category', '')
                })
        
        # Store the fetched wearables data under the current address
        wearables[address] = tmp_wearables
        print(f' {time.time() - t0:.2f} s')  # Print time taken for the current fetch
    
    return wearables  # Return the complete wearables dictionary
