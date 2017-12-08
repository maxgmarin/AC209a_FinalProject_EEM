
### Define functions to extract Audio Features
https://github.com/juandes/spotify-audio-features-data-experiment/blob/master/get_data.py
    
    

# Audio Features Parsing

I am inspired by this blog post: https://towardsdatascience.com/is-my-spotify-music-boring-an-analysis-involving-music-data-and-machine-learning-47550ae931de  

https://github.com/juandes/spotify-audio-features-data-experiment/blob/master/get_data.py

I am going to employ a similar analysis strategy.
        
  


```python
# Import statements
import numpy as np
import pandas as pd
import matplotlib
import spotipy
import spotipy.util as util
import sys
import matplotlib.pyplot as plt
from time import sleep

import seaborn as sns

%matplotlib inline

import ast

import time

from sklearn.manifold import TSNE

from tqdm import tqdm
```


```python
# Create spotify object for API calls
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

client_credentials_manager = SpotifyClientCredentials(client_id='edecee7ae47c4187a736c1abe70f79fa', 
                                                      client_secret='b3ca8746bdef4cf6810c6ac8e4efb1ff')
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


```


```python
targetUsername = 'spotify'

playlists = sp.user_playlists(targetUsername, limit=50)

```


```python
def getPlayListIDs(username, sp):
    playlists = sp.user_playlists(targetUsername, limit=50)
    allPlaylist_ids = []

    while playlists:
        for i, playlist in enumerate(playlists['items']):

            if playlist['name'] == None:
                #print(playlist)
                continue

            #print("%4d %s %s" % (i + 1 + playlists['offset'], playlist['id'],  playlist['name']))

            allPlaylist_ids.append(playlist["id"])

        if playlists['next']:
            playlists = sp.next(playlists)
        else:
            playlists = None
            
    return allPlaylist_ids
```


```python
allPlaylist_ids = getPlayListIDs('spotify', sp)
```

    {'owner': {'type': 'user', 'display_name': 'Spotify', 'uri': 'spotify:user:spotify', 'id': 'spotify', 'href': 'https://api.spotify.com/v1/users/spotify', 'external_urls': {'spotify': 'http://open.spotify.com/user/spotify'}}, 'public': True, 'images': [], 'href': 'https://api.spotify.com/v1/users/spotify/playlists/37i9dQZF1DX5JP3TnliPkl', 'uri': 'spotify:user:spotify:playlist:37i9dQZF1DX5JP3TnliPkl', 'snapshot_id': None, 'type': 'playlist', 'collaborative': False, 'tracks': {'total': 0, 'href': 'https://api.spotify.com/v1/users/spotify/playlists/37i9dQZF1DX5JP3TnliPkl/tracks'}, 'name': None, 'id': '37i9dQZF1DX5JP3TnliPkl', 'external_urls': {'spotify': 'http://open.spotify.com/user/spotify/playlist/37i9dQZF1DX5JP3TnliPkl'}}
    {'owner': {'type': 'user', 'display_name': 'Spotify', 'uri': 'spotify:user:spotify', 'id': 'spotify', 'href': 'https://api.spotify.com/v1/users/spotify', 'external_urls': {'spotify': 'http://open.spotify.com/user/spotify'}}, 'public': True, 'images': [], 'href': 'https://api.spotify.com/v1/users/spotify/playlists/37i9dQZF1DWSDKFhPJcdgc', 'uri': 'spotify:user:spotify:playlist:37i9dQZF1DWSDKFhPJcdgc', 'snapshot_id': None, 'type': 'playlist', 'collaborative': False, 'tracks': {'total': 0, 'href': 'https://api.spotify.com/v1/users/spotify/playlists/37i9dQZF1DWSDKFhPJcdgc/tracks'}, 'name': None, 'id': '37i9dQZF1DWSDKFhPJcdgc', 'external_urls': {'spotify': 'http://open.spotify.com/user/spotify/playlist/37i9dQZF1DWSDKFhPJcdgc'}}
    {'owner': {'type': 'user', 'display_name': 'Spotify', 'uri': 'spotify:user:spotify', 'id': 'spotify', 'href': 'https://api.spotify.com/v1/users/spotify', 'external_urls': {'spotify': 'http://open.spotify.com/user/spotify'}}, 'public': True, 'images': [], 'href': 'https://api.spotify.com/v1/users/spotify/playlists/37i9dQZF1DWTs4di2L4PuA', 'uri': 'spotify:user:spotify:playlist:37i9dQZF1DWTs4di2L4PuA', 'snapshot_id': None, 'type': 'playlist', 'collaborative': False, 'tracks': {'total': 0, 'href': 'https://api.spotify.com/v1/users/spotify/playlists/37i9dQZF1DWTs4di2L4PuA/tracks'}, 'name': None, 'id': '37i9dQZF1DWTs4di2L4PuA', 'external_urls': {'spotify': 'http://open.spotify.com/user/spotify/playlist/37i9dQZF1DWTs4di2L4PuA'}}
    {'owner': {'type': 'user', 'display_name': 'Spotify', 'uri': 'spotify:user:spotify', 'id': 'spotify', 'href': 'https://api.spotify.com/v1/users/spotify', 'external_urls': {'spotify': 'http://open.spotify.com/user/spotify'}}, 'public': True, 'images': [], 'href': 'https://api.spotify.com/v1/users/spotify/playlists/37i9dQZF1DWZ78O1iz38e7', 'uri': 'spotify:user:spotify:playlist:37i9dQZF1DWZ78O1iz38e7', 'snapshot_id': None, 'type': 'playlist', 'collaborative': False, 'tracks': {'total': 0, 'href': 'https://api.spotify.com/v1/users/spotify/playlists/37i9dQZF1DWZ78O1iz38e7/tracks'}, 'name': None, 'id': '37i9dQZF1DWZ78O1iz38e7', 'external_urls': {'spotify': 'http://open.spotify.com/user/spotify/playlist/37i9dQZF1DWZ78O1iz38e7'}}
    {'owner': {'type': 'user', 'display_name': 'Spotify', 'uri': 'spotify:user:spotify', 'id': 'spotify', 'href': 'https://api.spotify.com/v1/users/spotify', 'external_urls': {'spotify': 'http://open.spotify.com/user/spotify'}}, 'public': True, 'images': [], 'href': 'https://api.spotify.com/v1/users/spotify/playlists/37i9dQZF1DX4AUZ5NxnCT2', 'uri': 'spotify:user:spotify:playlist:37i9dQZF1DX4AUZ5NxnCT2', 'snapshot_id': None, 'type': 'playlist', 'collaborative': False, 'tracks': {'total': 0, 'href': 'https://api.spotify.com/v1/users/spotify/playlists/37i9dQZF1DX4AUZ5NxnCT2/tracks'}, 'name': None, 'id': '37i9dQZF1DX4AUZ5NxnCT2', 'external_urls': {'spotify': 'http://open.spotify.com/user/spotify/playlist/37i9dQZF1DX4AUZ5NxnCT2'}}
    {'owner': {'type': 'user', 'display_name': 'Spotify', 'uri': 'spotify:user:spotify', 'id': 'spotify', 'href': 'https://api.spotify.com/v1/users/spotify', 'external_urls': {'spotify': 'http://open.spotify.com/user/spotify'}}, 'public': True, 'images': [], 'href': 'https://api.spotify.com/v1/users/spotify/playlists/37i9dQZF1DX6G7ZCoxfhIh', 'uri': 'spotify:user:spotify:playlist:37i9dQZF1DX6G7ZCoxfhIh', 'snapshot_id': None, 'type': 'playlist', 'collaborative': False, 'tracks': {'total': 0, 'href': 'https://api.spotify.com/v1/users/spotify/playlists/37i9dQZF1DX6G7ZCoxfhIh/tracks'}, 'name': None, 'id': '37i9dQZF1DX6G7ZCoxfhIh', 'external_urls': {'spotify': 'http://open.spotify.com/user/spotify/playlist/37i9dQZF1DX6G7ZCoxfhIh'}}
    {'owner': {'type': 'user', 'display_name': 'Spotify', 'uri': 'spotify:user:spotify', 'id': 'spotify', 'href': 'https://api.spotify.com/v1/users/spotify', 'external_urls': {'spotify': 'http://open.spotify.com/user/spotify'}}, 'public': True, 'images': [], 'href': 'https://api.spotify.com/v1/users/spotify/playlists/37i9dQZF1DX3SpoFfWPxDm', 'uri': 'spotify:user:spotify:playlist:37i9dQZF1DX3SpoFfWPxDm', 'snapshot_id': None, 'type': 'playlist', 'collaborative': False, 'tracks': {'total': 0, 'href': 'https://api.spotify.com/v1/users/spotify/playlists/37i9dQZF1DX3SpoFfWPxDm/tracks'}, 'name': None, 'id': '37i9dQZF1DX3SpoFfWPxDm', 'external_urls': {'spotify': 'http://open.spotify.com/user/spotify/playlist/37i9dQZF1DX3SpoFfWPxDm'}}
    {'owner': {'type': 'user', 'display_name': 'Spotify', 'uri': 'spotify:user:spotify', 'id': 'spotify', 'href': 'https://api.spotify.com/v1/users/spotify', 'external_urls': {'spotify': 'http://open.spotify.com/user/spotify'}}, 'public': True, 'images': [], 'href': 'https://api.spotify.com/v1/users/spotify/playlists/37i9dQZF1DX6KbQTswZ5Ul', 'uri': 'spotify:user:spotify:playlist:37i9dQZF1DX6KbQTswZ5Ul', 'snapshot_id': None, 'type': 'playlist', 'collaborative': False, 'tracks': {'total': 0, 'href': 'https://api.spotify.com/v1/users/spotify/playlists/37i9dQZF1DX6KbQTswZ5Ul/tracks'}, 'name': None, 'id': '37i9dQZF1DX6KbQTswZ5Ul', 'external_urls': {'spotify': 'http://open.spotify.com/user/spotify/playlist/37i9dQZF1DX6KbQTswZ5Ul'}}
    {'owner': {'type': 'user', 'display_name': 'Spotify', 'uri': 'spotify:user:spotify', 'id': 'spotify', 'href': 'https://api.spotify.com/v1/users/spotify', 'external_urls': {'spotify': 'http://open.spotify.com/user/spotify'}}, 'public': True, 'images': [], 'href': 'https://api.spotify.com/v1/users/spotify/playlists/37i9dQZF1DX00Cpqy2v7i0', 'uri': 'spotify:user:spotify:playlist:37i9dQZF1DX00Cpqy2v7i0', 'snapshot_id': None, 'type': 'playlist', 'collaborative': False, 'tracks': {'total': 0, 'href': 'https://api.spotify.com/v1/users/spotify/playlists/37i9dQZF1DX00Cpqy2v7i0/tracks'}, 'name': None, 'id': '37i9dQZF1DX00Cpqy2v7i0', 'external_urls': {'spotify': 'http://open.spotify.com/user/spotify/playlist/37i9dQZF1DX00Cpqy2v7i0'}}
    {'owner': {'type': 'user', 'display_name': 'Spotify', 'uri': 'spotify:user:spotify', 'id': 'spotify', 'href': 'https://api.spotify.com/v1/users/spotify', 'external_urls': {'spotify': 'http://open.spotify.com/user/spotify'}}, 'public': True, 'images': [], 'href': 'https://api.spotify.com/v1/users/spotify/playlists/37i9dQZF1DX3BaWfSusILe', 'uri': 'spotify:user:spotify:playlist:37i9dQZF1DX3BaWfSusILe', 'snapshot_id': None, 'type': 'playlist', 'collaborative': False, 'tracks': {'total': 0, 'href': 'https://api.spotify.com/v1/users/spotify/playlists/37i9dQZF1DX3BaWfSusILe/tracks'}, 'name': None, 'id': '37i9dQZF1DX3BaWfSusILe', 'external_urls': {'spotify': 'http://open.spotify.com/user/spotify/playlist/37i9dQZF1DX3BaWfSusILe'}}
    {'owner': {'type': 'user', 'display_name': 'Spotify', 'uri': 'spotify:user:spotify', 'id': 'spotify', 'href': 'https://api.spotify.com/v1/users/spotify', 'external_urls': {'spotify': 'http://open.spotify.com/user/spotify'}}, 'public': True, 'images': [], 'href': 'https://api.spotify.com/v1/users/spotify/playlists/37i9dQZF1DWUu7JsNp46yZ', 'uri': 'spotify:user:spotify:playlist:37i9dQZF1DWUu7JsNp46yZ', 'snapshot_id': None, 'type': 'playlist', 'collaborative': False, 'tracks': {'total': 0, 'href': 'https://api.spotify.com/v1/users/spotify/playlists/37i9dQZF1DWUu7JsNp46yZ/tracks'}, 'name': None, 'id': '37i9dQZF1DWUu7JsNp46yZ', 'external_urls': {'spotify': 'http://open.spotify.com/user/spotify/playlist/37i9dQZF1DWUu7JsNp46yZ'}}
    {'owner': {'type': 'user', 'display_name': 'Spotify', 'uri': 'spotify:user:spotify', 'id': 'spotify', 'href': 'https://api.spotify.com/v1/users/spotify', 'external_urls': {'spotify': 'http://open.spotify.com/user/spotify'}}, 'public': True, 'images': [], 'href': 'https://api.spotify.com/v1/users/spotify/playlists/37i9dQZF1DWSnydmzQrJyU', 'uri': 'spotify:user:spotify:playlist:37i9dQZF1DWSnydmzQrJyU', 'snapshot_id': None, 'type': 'playlist', 'collaborative': False, 'tracks': {'total': 0, 'href': 'https://api.spotify.com/v1/users/spotify/playlists/37i9dQZF1DWSnydmzQrJyU/tracks'}, 'name': None, 'id': '37i9dQZF1DWSnydmzQrJyU', 'external_urls': {'spotify': 'http://open.spotify.com/user/spotify/playlist/37i9dQZF1DWSnydmzQrJyU'}}
    {'owner': {'type': 'user', 'display_name': 'Spotify', 'uri': 'spotify:user:spotify', 'id': 'spotify', 'href': 'https://api.spotify.com/v1/users/spotify', 'external_urls': {'spotify': 'http://open.spotify.com/user/spotify'}}, 'public': True, 'images': [], 'href': 'https://api.spotify.com/v1/users/spotify/playlists/37i9dQZF1DX5Wyy7I2Qrxv', 'uri': 'spotify:user:spotify:playlist:37i9dQZF1DX5Wyy7I2Qrxv', 'snapshot_id': None, 'type': 'playlist', 'collaborative': False, 'tracks': {'total': 0, 'href': 'https://api.spotify.com/v1/users/spotify/playlists/37i9dQZF1DX5Wyy7I2Qrxv/tracks'}, 'name': None, 'id': '37i9dQZF1DX5Wyy7I2Qrxv', 'external_urls': {'spotify': 'http://open.spotify.com/user/spotify/playlist/37i9dQZF1DX5Wyy7I2Qrxv'}}
    {'owner': {'type': 'user', 'display_name': 'Spotify', 'uri': 'spotify:user:spotify', 'id': 'spotify', 'href': 'https://api.spotify.com/v1/users/spotify', 'external_urls': {'spotify': 'http://open.spotify.com/user/spotify'}}, 'public': True, 'images': [], 'href': 'https://api.spotify.com/v1/users/spotify/playlists/37i9dQZF1DWVlEwW9gqgAE', 'uri': 'spotify:user:spotify:playlist:37i9dQZF1DWVlEwW9gqgAE', 'snapshot_id': None, 'type': 'playlist', 'collaborative': False, 'tracks': {'total': 0, 'href': 'https://api.spotify.com/v1/users/spotify/playlists/37i9dQZF1DWVlEwW9gqgAE/tracks'}, 'name': None, 'id': '37i9dQZF1DWVlEwW9gqgAE', 'external_urls': {'spotify': 'http://open.spotify.com/user/spotify/playlist/37i9dQZF1DWVlEwW9gqgAE'}}
    {'owner': {'type': 'user', 'display_name': 'Spotify', 'uri': 'spotify:user:spotify', 'id': 'spotify', 'href': 'https://api.spotify.com/v1/users/spotify', 'external_urls': {'spotify': 'http://open.spotify.com/user/spotify'}}, 'public': True, 'images': [], 'href': 'https://api.spotify.com/v1/users/spotify/playlists/37i9dQZF1DXa03UPQL42me', 'uri': 'spotify:user:spotify:playlist:37i9dQZF1DXa03UPQL42me', 'snapshot_id': None, 'type': 'playlist', 'collaborative': False, 'tracks': {'total': 0, 'href': 'https://api.spotify.com/v1/users/spotify/playlists/37i9dQZF1DXa03UPQL42me/tracks'}, 'name': None, 'id': '37i9dQZF1DXa03UPQL42me', 'external_urls': {'spotify': 'http://open.spotify.com/user/spotify/playlist/37i9dQZF1DXa03UPQL42me'}}
    {'owner': {'type': 'user', 'display_name': 'Spotify', 'uri': 'spotify:user:spotify', 'id': 'spotify', 'href': 'https://api.spotify.com/v1/users/spotify', 'external_urls': {'spotify': 'http://open.spotify.com/user/spotify'}}, 'public': True, 'images': [], 'href': 'https://api.spotify.com/v1/users/spotify/playlists/37i9dQZF1DX74QLyrlKNWh', 'uri': 'spotify:user:spotify:playlist:37i9dQZF1DX74QLyrlKNWh', 'snapshot_id': None, 'type': 'playlist', 'collaborative': False, 'tracks': {'total': 0, 'href': 'https://api.spotify.com/v1/users/spotify/playlists/37i9dQZF1DX74QLyrlKNWh/tracks'}, 'name': None, 'id': '37i9dQZF1DX74QLyrlKNWh', 'external_urls': {'spotify': 'http://open.spotify.com/user/spotify/playlist/37i9dQZF1DX74QLyrlKNWh'}}
    {'owner': {'type': 'user', 'display_name': 'Spotify', 'uri': 'spotify:user:spotify', 'id': 'spotify', 'href': 'https://api.spotify.com/v1/users/spotify', 'external_urls': {'spotify': 'http://open.spotify.com/user/spotify'}}, 'public': True, 'images': [], 'href': 'https://api.spotify.com/v1/users/spotify/playlists/37i9dQZF1DWT6G0ByJ91Sr', 'uri': 'spotify:user:spotify:playlist:37i9dQZF1DWT6G0ByJ91Sr', 'snapshot_id': None, 'type': 'playlist', 'collaborative': False, 'tracks': {'total': 0, 'href': 'https://api.spotify.com/v1/users/spotify/playlists/37i9dQZF1DWT6G0ByJ91Sr/tracks'}, 'name': None, 'id': '37i9dQZF1DWT6G0ByJ91Sr', 'external_urls': {'spotify': 'http://open.spotify.com/user/spotify/playlist/37i9dQZF1DWT6G0ByJ91Sr'}}
    {'owner': {'type': 'user', 'display_name': 'Spotify', 'uri': 'spotify:user:spotify', 'id': 'spotify', 'href': 'https://api.spotify.com/v1/users/spotify', 'external_urls': {'spotify': 'http://open.spotify.com/user/spotify'}}, 'public': True, 'images': [], 'href': 'https://api.spotify.com/v1/users/spotify/playlists/37i9dQZF1DXaajrW2mmprj', 'uri': 'spotify:user:spotify:playlist:37i9dQZF1DXaajrW2mmprj', 'snapshot_id': None, 'type': 'playlist', 'collaborative': False, 'tracks': {'total': 0, 'href': 'https://api.spotify.com/v1/users/spotify/playlists/37i9dQZF1DXaajrW2mmprj/tracks'}, 'name': None, 'id': '37i9dQZF1DXaajrW2mmprj', 'external_urls': {'spotify': 'http://open.spotify.com/user/spotify/playlist/37i9dQZF1DXaajrW2mmprj'}}
    {'owner': {'type': 'user', 'display_name': 'Spotify', 'uri': 'spotify:user:spotify', 'id': 'spotify', 'href': 'https://api.spotify.com/v1/users/spotify', 'external_urls': {'spotify': 'http://open.spotify.com/user/spotify'}}, 'public': True, 'images': [], 'href': 'https://api.spotify.com/v1/users/spotify/playlists/37i9dQZF1DWStfH1EhH9ta', 'uri': 'spotify:user:spotify:playlist:37i9dQZF1DWStfH1EhH9ta', 'snapshot_id': None, 'type': 'playlist', 'collaborative': False, 'tracks': {'total': 0, 'href': 'https://api.spotify.com/v1/users/spotify/playlists/37i9dQZF1DWStfH1EhH9ta/tracks'}, 'name': None, 'id': '37i9dQZF1DWStfH1EhH9ta', 'external_urls': {'spotify': 'http://open.spotify.com/user/spotify/playlist/37i9dQZF1DWStfH1EhH9ta'}}
    {'owner': {'type': 'user', 'display_name': 'Spotify', 'uri': 'spotify:user:spotify', 'id': 'spotify', 'href': 'https://api.spotify.com/v1/users/spotify', 'external_urls': {'spotify': 'http://open.spotify.com/user/spotify'}}, 'public': True, 'images': [], 'href': 'https://api.spotify.com/v1/users/spotify/playlists/37i9dQZF1DXb5Fp2OFgslq', 'uri': 'spotify:user:spotify:playlist:37i9dQZF1DXb5Fp2OFgslq', 'snapshot_id': None, 'type': 'playlist', 'collaborative': False, 'tracks': {'total': 0, 'href': 'https://api.spotify.com/v1/users/spotify/playlists/37i9dQZF1DXb5Fp2OFgslq/tracks'}, 'name': None, 'id': '37i9dQZF1DXb5Fp2OFgslq', 'external_urls': {'spotify': 'http://open.spotify.com/user/spotify/playlist/37i9dQZF1DXb5Fp2OFgslq'}}
    {'owner': {'type': 'user', 'display_name': 'Spotify', 'uri': 'spotify:user:spotify', 'id': 'spotify', 'href': 'https://api.spotify.com/v1/users/spotify', 'external_urls': {'spotify': 'http://open.spotify.com/user/spotify'}}, 'public': True, 'images': [], 'href': 'https://api.spotify.com/v1/users/spotify/playlists/37i9dQZF1DWZ4eE8tkPZac', 'uri': 'spotify:user:spotify:playlist:37i9dQZF1DWZ4eE8tkPZac', 'snapshot_id': None, 'type': 'playlist', 'collaborative': False, 'tracks': {'total': 0, 'href': 'https://api.spotify.com/v1/users/spotify/playlists/37i9dQZF1DWZ4eE8tkPZac/tracks'}, 'name': None, 'id': '37i9dQZF1DWZ4eE8tkPZac', 'external_urls': {'spotify': 'http://open.spotify.com/user/spotify/playlist/37i9dQZF1DWZ4eE8tkPZac'}}
    {'owner': {'uri': 'spotify:user:spotifycharts', 'type': 'user', 'id': 'spotifycharts', 'href': 'https://api.spotify.com/v1/users/spotifycharts', 'external_urls': {'spotify': 'http://open.spotify.com/user/spotifycharts'}}, 'public': True, 'images': [], 'href': 'https://api.spotify.com/v1/users/spotifycharts/playlists/37i9dQZEVXbKAM9DojWfAF', 'uri': 'spotify:user:spotifycharts:playlist:37i9dQZEVXbKAM9DojWfAF', 'snapshot_id': None, 'type': 'playlist', 'collaborative': False, 'tracks': {'total': 0, 'href': 'https://api.spotify.com/v1/users/spotifycharts/playlists/37i9dQZEVXbKAM9DojWfAF/tracks'}, 'name': None, 'id': '37i9dQZEVXbKAM9DojWfAF', 'external_urls': {'spotify': 'http://open.spotify.com/user/spotifycharts/playlist/37i9dQZEVXbKAM9DojWfAF'}}
    {'owner': {'uri': 'spotify:user:spotifycharts', 'type': 'user', 'id': 'spotifycharts', 'href': 'https://api.spotify.com/v1/users/spotifycharts', 'external_urls': {'spotify': 'http://open.spotify.com/user/spotifycharts'}}, 'public': True, 'images': [], 'href': 'https://api.spotify.com/v1/users/spotifycharts/playlists/37i9dQZEVXbKGcyg6TFGx6', 'uri': 'spotify:user:spotifycharts:playlist:37i9dQZEVXbKGcyg6TFGx6', 'snapshot_id': None, 'type': 'playlist', 'collaborative': False, 'tracks': {'total': 0, 'href': 'https://api.spotify.com/v1/users/spotifycharts/playlists/37i9dQZEVXbKGcyg6TFGx6/tracks'}, 'name': None, 'id': '37i9dQZEVXbKGcyg6TFGx6', 'external_urls': {'spotify': 'http://open.spotify.com/user/spotifycharts/playlist/37i9dQZEVXbKGcyg6TFGx6'}}
    {'owner': {'uri': 'spotify:user:spotifycharts', 'type': 'user', 'id': 'spotifycharts', 'href': 'https://api.spotify.com/v1/users/spotifycharts', 'external_urls': {'spotify': 'http://open.spotify.com/user/spotifycharts'}}, 'public': True, 'images': [], 'href': 'https://api.spotify.com/v1/users/spotifycharts/playlists/37i9dQZEVXbLpFfBvgJdDL', 'uri': 'spotify:user:spotifycharts:playlist:37i9dQZEVXbLpFfBvgJdDL', 'snapshot_id': None, 'type': 'playlist', 'collaborative': False, 'tracks': {'total': 0, 'href': 'https://api.spotify.com/v1/users/spotifycharts/playlists/37i9dQZEVXbLpFfBvgJdDL/tracks'}, 'name': None, 'id': '37i9dQZEVXbLpFfBvgJdDL', 'external_urls': {'spotify': 'http://open.spotify.com/user/spotifycharts/playlist/37i9dQZEVXbLpFfBvgJdDL'}}
    {'owner': {'uri': 'spotify:user:spotifycharts', 'type': 'user', 'id': 'spotifycharts', 'href': 'https://api.spotify.com/v1/users/spotifycharts', 'external_urls': {'spotify': 'http://open.spotify.com/user/spotifycharts'}}, 'public': True, 'images': [], 'href': 'https://api.spotify.com/v1/users/spotifycharts/playlists/37i9dQZEVXbO8SSnS9T1th', 'uri': 'spotify:user:spotifycharts:playlist:37i9dQZEVXbO8SSnS9T1th', 'snapshot_id': None, 'type': 'playlist', 'collaborative': False, 'tracks': {'total': 0, 'href': 'https://api.spotify.com/v1/users/spotifycharts/playlists/37i9dQZEVXbO8SSnS9T1th/tracks'}, 'name': None, 'id': '37i9dQZEVXbO8SSnS9T1th', 'external_urls': {'spotify': 'http://open.spotify.com/user/spotifycharts/playlist/37i9dQZEVXbO8SSnS9T1th'}}



```python
len(allPlaylist_ids)
```




    1673




```python
# Importing data (playlist, track, and artist info)
DF_TrackAlbumInfo = pd.read_csv("/Users/max/Documents/AP209a/FinalProject/df_without_albumtracks.csv").drop(["Unnamed: 0"], axis = 1)

```


```python
# Get all unique track IDs in dataframe 
# Link: (https://stackoverflow.com/questions/7961363/removing-duplicates-in-lists)

All_TrackIDs_InAnalysis = list(dict.fromkeys(DF_TrackAlbumInfo["track_id_y"]))
```


```python
len(All_TrackIDs_InAnalysis)
```




    63902




```python
import time
```


```python
listOfTrack_AudioFeatures = []

IDs_AlreadyUsed = []

#i = 1
for track_ID in tqdm(All_TrackIDs_InAnalysis):
    #sp.audio_analysis("2xmrfQpmS2iJExTlklLoAL")
    
    #track = sp.track(track_ID)
    #print(track_SP)
    
    #track_Name = track["name"]     

    #track_explicit = track['explicit']
    
    
    
    features = sp.audio_features(track_ID)[0] # Get audio features for this specific track

    if features != None:
    
        track_AudioFeatures = [features['energy'], features['liveness'],
                               features['tempo'], features['speechiness'],
                               features['acousticness'], features['instrumentalness'],
                               features['time_signature'], features['danceability'],
                               features['key'], features['duration_ms'],
                               features['loudness'], features['valence'],
                               features['mode'], features['uri']]
    else:
        track_AudioFeatures = [None]*14


    #trackData = ( [ track_Name, track_ID, track_explicit] + track_AudioFeatures )
    trackData = ( [track_ID] + track_AudioFeatures )

    listOfTrack_AudioFeatures.append(trackData)

    #break
    #print(features)

    IDs_AlreadyUsed.append(track_ID)

    time.sleep(0.05)
    #print(i, i/63902)
    
    #i += 1
```

     15%|█▌        | 9675/63902 [49:33<4:37:45,  3.25it/s]


    ---------------------------------------------------------------------------

    TimeoutError                              Traceback (most recent call last)

    ~/anaconda3/lib/python3.5/site-packages/urllib3/connection.py in _new_conn(self)
        140             conn = connection.create_connection(
    --> 141                 (self.host, self.port), self.timeout, **extra_kw)
        142 


    ~/anaconda3/lib/python3.5/site-packages/urllib3/util/connection.py in create_connection(address, timeout, source_address, socket_options)
         82     if err is not None:
    ---> 83         raise err
         84 


    ~/anaconda3/lib/python3.5/site-packages/urllib3/util/connection.py in create_connection(address, timeout, source_address, socket_options)
         72                 sock.bind(source_address)
    ---> 73             sock.connect(sa)
         74             return sock


    TimeoutError: [Errno 60] Operation timed out

    
    During handling of the above exception, another exception occurred:


    NewConnectionError                        Traceback (most recent call last)

    ~/anaconda3/lib/python3.5/site-packages/urllib3/connectionpool.py in urlopen(self, method, url, body, headers, retries, redirect, assert_same_host, timeout, pool_timeout, release_conn, chunked, body_pos, **response_kw)
        600                                                   body=body, headers=headers,
    --> 601                                                   chunked=chunked)
        602 


    ~/anaconda3/lib/python3.5/site-packages/urllib3/connectionpool.py in _make_request(self, conn, method, url, timeout, chunked, **httplib_request_kw)
        345         try:
    --> 346             self._validate_conn(conn)
        347         except (SocketTimeout, BaseSSLError) as e:


    ~/anaconda3/lib/python3.5/site-packages/urllib3/connectionpool.py in _validate_conn(self, conn)
        849         if not getattr(conn, 'sock', None):  # AppEngine might not have  `.sock`
    --> 850             conn.connect()
        851 


    ~/anaconda3/lib/python3.5/site-packages/urllib3/connection.py in connect(self)
        283         # Add certificate verification
    --> 284         conn = self._new_conn()
        285 


    ~/anaconda3/lib/python3.5/site-packages/urllib3/connection.py in _new_conn(self)
        149             raise NewConnectionError(
    --> 150                 self, "Failed to establish a new connection: %s" % e)
        151 


    NewConnectionError: <urllib3.connection.VerifiedHTTPSConnection object at 0x114ba9240>: Failed to establish a new connection: [Errno 60] Operation timed out

    
    During handling of the above exception, another exception occurred:


    MaxRetryError                             Traceback (most recent call last)

    ~/anaconda3/lib/python3.5/site-packages/requests/adapters.py in send(self, request, stream, timeout, verify, cert, proxies)
        439                     retries=self.max_retries,
    --> 440                     timeout=timeout
        441                 )


    ~/anaconda3/lib/python3.5/site-packages/urllib3/connectionpool.py in urlopen(self, method, url, body, headers, retries, redirect, assert_same_host, timeout, pool_timeout, release_conn, chunked, body_pos, **response_kw)
        638             retries = retries.increment(method, url, error=e, _pool=self,
    --> 639                                         _stacktrace=sys.exc_info()[2])
        640             retries.sleep()


    ~/anaconda3/lib/python3.5/site-packages/urllib3/util/retry.py in increment(self, method, url, response, error, _pool, _stacktrace)
        387         if new_retry.is_exhausted():
    --> 388             raise MaxRetryError(_pool, url, error or ResponseError(cause))
        389 


    MaxRetryError: HTTPSConnectionPool(host='api.spotify.com', port=443): Max retries exceeded with url: /v1/audio-features/?ids=04Mh4OWSBUBn5Vpna4DXrA (Caused by NewConnectionError('<urllib3.connection.VerifiedHTTPSConnection object at 0x114ba9240>: Failed to establish a new connection: [Errno 60] Operation timed out',))

    
    During handling of the above exception, another exception occurred:


    ConnectionError                           Traceback (most recent call last)

    <ipython-input-56-9f64de8a1e06> in <module>()
         16 
         17 
    ---> 18     features = sp.audio_features(track_ID)[0] # Get audio features for this specific track
         19 
         20     if features != None:


    ~/anaconda3/lib/python3.5/site-packages/spotipy/client.py in audio_features(self, tracks)
        824         if isinstance(tracks, str):
        825             trackid = self._get_id('track', tracks)
    --> 826             results = self._get('audio-features/?ids=' + trackid)
        827         else:
        828             tlist = [self._get_id('track', t) for t in tracks]


    ~/anaconda3/lib/python3.5/site-packages/spotipy/client.py in _get(self, url, args, payload, **kwargs)
        144         while retries > 0:
        145             try:
    --> 146                 return self._internal_call('GET', url, payload, kwargs)
        147             except SpotifyException as e:
        148                 retries -= 1


    ~/anaconda3/lib/python3.5/site-packages/spotipy/client.py in _internal_call(self, method, url, payload, params)
        106         if self.trace_out:
        107             print(url)
    --> 108         r = self._session.request(method, url, headers=headers, proxies=self.proxies, **args)
        109 
        110         if self.trace:  # pragma: no cover


    ~/anaconda3/lib/python3.5/site-packages/requests/sessions.py in request(self, method, url, params, data, headers, cookies, files, auth, timeout, allow_redirects, proxies, hooks, stream, verify, cert, json)
        506         }
        507         send_kwargs.update(settings)
    --> 508         resp = self.send(prep, **send_kwargs)
        509 
        510         return resp


    ~/anaconda3/lib/python3.5/site-packages/requests/sessions.py in send(self, request, **kwargs)
        616 
        617         # Send the request
    --> 618         r = adapter.send(request, **kwargs)
        619 
        620         # Total elapsed time of the request (approximately)


    ~/anaconda3/lib/python3.5/site-packages/requests/adapters.py in send(self, request, stream, timeout, verify, cert, proxies)
        506                 raise SSLError(e, request=request)
        507 
    --> 508             raise ConnectionError(e, request=request)
        509 
        510         except ClosedPoolError as e:


    ConnectionError: HTTPSConnectionPool(host='api.spotify.com', port=443): Max retries exceeded with url: /v1/audio-features/?ids=04Mh4OWSBUBn5Vpna4DXrA (Caused by NewConnectionError('<urllib3.connection.VerifiedHTTPSConnection object at 0x114ba9240>: Failed to establish a new connection: [Errno 60] Operation timed out',))



```python
l1 = All_TrackIDs_InAnalysis

l2 = IDs_AlreadyUsed

TrackIDs_StillNotParsed = [x for x in l1 if x not in l2]
```


```python
TrackIDs_StillNotParsed [:10]
```




    ['04Mh4OWSBUBn5Vpna4DXrA',
     '3zVOXpIE9A1Bp92BHiKiX0',
     '0hi3WkgcFlsMKdkic9KZp1',
     '4u3aErjHkwVqqdyfNDMyJ1',
     '14xZtVbG1Ue1Bv8fbVVSl5',
     '6x8j5V85ZJHtvOSjSamcVc',
     '5fbeWfU4fnLobWibtLTzdQ',
     '5AlI3P7UjjNPo4ltpYXpj0',
     '7GRM1FIKDLnfVcrqyV4OM9',
     '6UFUBoYk4c1bvVFFKfa1Fs']




```python
#listOfTrack_AudioFeatures = []

#IDs_AlreadyUsed = []

l1 = All_TrackIDs_InAnalysis

l2 = IDs_AlreadyUsed

TrackIDs_StillNotParsed = [x for x in l1 if x not in l2]
for track_ID in tqdm(TrackIDs_StillNotParsed):
    #sp.audio_analysis("2xmrfQpmS2iJExTlklLoAL")
    
    #track = sp.track(track_ID)
    #print(track_SP)
    
    #track_Name = track["name"]     

    #track_explicit = track['explicit']
    
    
    
    features = sp.audio_features(track_ID)[0] # Get audio features for this specific track

    if features != None:
    
        track_AudioFeatures = [features['energy'], features['liveness'],
                               features['tempo'], features['speechiness'],
                               features['acousticness'], features['instrumentalness'],
                               features['time_signature'], features['danceability'],
                               features['key'], features['duration_ms'],
                               features['loudness'], features['valence'],
                               features['mode'], features['uri']]
    else:
        track_AudioFeatures = [None]*14


    #trackData = ( [ track_Name, track_ID, track_explicit] + track_AudioFeatures )
    trackData = ( [track_ID] + track_AudioFeatures )

    listOfTrack_AudioFeatures.append(trackData)

    #break
    #print(features)

    IDs_AlreadyUsed.append(track_ID)

    time.sleep(0.15)
    #print(i, i/63902)
    
    #i += 1
```

    
      0%|          | 0/54227 [00:00<?, ?it/s][A
      0%|          | 1/54227 [00:00<11:16:39,  1.34it/s][A
      0%|          | 2/54227 [00:01<9:57:13,  1.51it/s] [A
      0%|          | 3/54227 [00:01<8:35:43,  1.75it/s][A
      0%|          | 4/54227 [00:02<7:52:26,  1.91it/s][A
      0%|          | 5/54227 [00:02<7:31:03,  2.00it/s][A
      0%|          | 6/54227 [00:02<7:18:57,  2.06it/s][A
      0%|          | 7/54227 [00:03<7:07:01,  2.12it/s][A
      0%|          | 8/54227 [00:03<6:58:00,  2.16it/s][A
      0%|          | 9/54227 [00:04<6:58:06,  2.16it/s][A
      0%|          | 10/54227 [00:04<6:52:43,  2.19it/s][A
      0%|          | 11/54227 [00:04<6:47:24,  2.22it/s][A
      0%|          | 12/54227 [00:05<6:42:47,  2.24it/s][A
      0%|          | 13/54227 [00:05<6:38:25,  2.27it/s][A
      0%|          | 14/54227 [00:06<6:34:43,  2.29it/s][A
      0%|          | 15/54227 [00:06<6:34:14,  2.29it/s][A
      0%|          | 16/54227 [00:06<6:31:37,  2.31it/s][A
      0%|          | 17/54227 [00:07<6:28:34,  2.33it/s][A
      0%|          | 18/54227 [00:07<6:26:33,  2.34it/s][A
      0%|          | 19/54227 [00:08<6:24:43,  2.35it/s][A
      0%|          | 20/54227 [00:08<6:22:48,  2.36it/s][A
    Exception in thread Thread-11:
    Traceback (most recent call last):
      File "/Users/max/anaconda3/lib/python3.5/threading.py", line 914, in _bootstrap_inner
        self.run()
      File "/Users/max/anaconda3/lib/python3.5/site-packages/tqdm/_tqdm.py", line 144, in run
        for instance in self.tqdm_cls._instances:
      File "/Users/max/anaconda3/lib/python3.5/_weakrefset.py", line 60, in __iter__
        for itemref in self.data:
    RuntimeError: Set changed size during iteration
    
     28%|██▊       | 15093/54227 [1:39:49<4:18:50,  2.52it/s]



    ---------------------------------------------------------------------------

    TimeoutError                              Traceback (most recent call last)

    ~/anaconda3/lib/python3.5/site-packages/urllib3/connection.py in _new_conn(self)
        140             conn = connection.create_connection(
    --> 141                 (self.host, self.port), self.timeout, **extra_kw)
        142 


    ~/anaconda3/lib/python3.5/site-packages/urllib3/util/connection.py in create_connection(address, timeout, source_address, socket_options)
         82     if err is not None:
    ---> 83         raise err
         84 


    ~/anaconda3/lib/python3.5/site-packages/urllib3/util/connection.py in create_connection(address, timeout, source_address, socket_options)
         72                 sock.bind(source_address)
    ---> 73             sock.connect(sa)
         74             return sock


    TimeoutError: [Errno 60] Operation timed out

    
    During handling of the above exception, another exception occurred:


    NewConnectionError                        Traceback (most recent call last)

    ~/anaconda3/lib/python3.5/site-packages/urllib3/connectionpool.py in urlopen(self, method, url, body, headers, retries, redirect, assert_same_host, timeout, pool_timeout, release_conn, chunked, body_pos, **response_kw)
        600                                                   body=body, headers=headers,
    --> 601                                                   chunked=chunked)
        602 


    ~/anaconda3/lib/python3.5/site-packages/urllib3/connectionpool.py in _make_request(self, conn, method, url, timeout, chunked, **httplib_request_kw)
        345         try:
    --> 346             self._validate_conn(conn)
        347         except (SocketTimeout, BaseSSLError) as e:


    ~/anaconda3/lib/python3.5/site-packages/urllib3/connectionpool.py in _validate_conn(self, conn)
        849         if not getattr(conn, 'sock', None):  # AppEngine might not have  `.sock`
    --> 850             conn.connect()
        851 


    ~/anaconda3/lib/python3.5/site-packages/urllib3/connection.py in connect(self)
        283         # Add certificate verification
    --> 284         conn = self._new_conn()
        285 


    ~/anaconda3/lib/python3.5/site-packages/urllib3/connection.py in _new_conn(self)
        149             raise NewConnectionError(
    --> 150                 self, "Failed to establish a new connection: %s" % e)
        151 


    NewConnectionError: <urllib3.connection.VerifiedHTTPSConnection object at 0x1163806a0>: Failed to establish a new connection: [Errno 60] Operation timed out

    
    During handling of the above exception, another exception occurred:


    MaxRetryError                             Traceback (most recent call last)

    ~/anaconda3/lib/python3.5/site-packages/requests/adapters.py in send(self, request, stream, timeout, verify, cert, proxies)
        439                     retries=self.max_retries,
    --> 440                     timeout=timeout
        441                 )


    ~/anaconda3/lib/python3.5/site-packages/urllib3/connectionpool.py in urlopen(self, method, url, body, headers, retries, redirect, assert_same_host, timeout, pool_timeout, release_conn, chunked, body_pos, **response_kw)
        638             retries = retries.increment(method, url, error=e, _pool=self,
    --> 639                                         _stacktrace=sys.exc_info()[2])
        640             retries.sleep()


    ~/anaconda3/lib/python3.5/site-packages/urllib3/util/retry.py in increment(self, method, url, response, error, _pool, _stacktrace)
        387         if new_retry.is_exhausted():
    --> 388             raise MaxRetryError(_pool, url, error or ResponseError(cause))
        389 


    MaxRetryError: HTTPSConnectionPool(host='api.spotify.com', port=443): Max retries exceeded with url: /v1/audio-features/?ids=1PqICsBgqq8k2FrAdGUGfq (Caused by NewConnectionError('<urllib3.connection.VerifiedHTTPSConnection object at 0x1163806a0>: Failed to establish a new connection: [Errno 60] Operation timed out',))

    
    During handling of the above exception, another exception occurred:


    ConnectionError                           Traceback (most recent call last)

    <ipython-input-63-4a727bbe24b1> in <module>()
         20 
         21 
    ---> 22     features = sp.audio_features(track_ID)[0] # Get audio features for this specific track
         23 
         24     if features != None:


    ~/anaconda3/lib/python3.5/site-packages/spotipy/client.py in audio_features(self, tracks)
        824         if isinstance(tracks, str):
        825             trackid = self._get_id('track', tracks)
    --> 826             results = self._get('audio-features/?ids=' + trackid)
        827         else:
        828             tlist = [self._get_id('track', t) for t in tracks]


    ~/anaconda3/lib/python3.5/site-packages/spotipy/client.py in _get(self, url, args, payload, **kwargs)
        144         while retries > 0:
        145             try:
    --> 146                 return self._internal_call('GET', url, payload, kwargs)
        147             except SpotifyException as e:
        148                 retries -= 1


    ~/anaconda3/lib/python3.5/site-packages/spotipy/client.py in _internal_call(self, method, url, payload, params)
        106         if self.trace_out:
        107             print(url)
    --> 108         r = self._session.request(method, url, headers=headers, proxies=self.proxies, **args)
        109 
        110         if self.trace:  # pragma: no cover


    ~/anaconda3/lib/python3.5/site-packages/requests/sessions.py in request(self, method, url, params, data, headers, cookies, files, auth, timeout, allow_redirects, proxies, hooks, stream, verify, cert, json)
        506         }
        507         send_kwargs.update(settings)
    --> 508         resp = self.send(prep, **send_kwargs)
        509 
        510         return resp


    ~/anaconda3/lib/python3.5/site-packages/requests/sessions.py in send(self, request, **kwargs)
        616 
        617         # Send the request
    --> 618         r = adapter.send(request, **kwargs)
        619 
        620         # Total elapsed time of the request (approximately)


    ~/anaconda3/lib/python3.5/site-packages/requests/adapters.py in send(self, request, stream, timeout, verify, cert, proxies)
        506                 raise SSLError(e, request=request)
        507 
    --> 508             raise ConnectionError(e, request=request)
        509 
        510         except ClosedPoolError as e:


    ConnectionError: HTTPSConnectionPool(host='api.spotify.com', port=443): Max retries exceeded with url: /v1/audio-features/?ids=1PqICsBgqq8k2FrAdGUGfq (Caused by NewConnectionError('<urllib3.connection.VerifiedHTTPSConnection object at 0x1163806a0>: Failed to establish a new connection: [Errno 60] Operation timed out',))



```python
#listOfTrack_AudioFeatures = []

#IDs_AlreadyUsed = []

l1 = All_TrackIDs_InAnalysis

l2 = IDs_AlreadyUsed

TrackIDs_StillNotParsed = [x for x in l1 if x not in l2]
for track_ID in tqdm(TrackIDs_StillNotParsed):
    #sp.audio_analysis("2xmrfQpmS2iJExTlklLoAL")
    
    #track = sp.track(track_ID)
    #print(track_SP)
    
    #track_Name = track["name"]     

    #track_explicit = track['explicit']
    
    
    
    features = sp.audio_features(track_ID)[0] # Get audio features for this specific track

    if features != None:
    
        track_AudioFeatures = [features['energy'], features['liveness'],
                               features['tempo'], features['speechiness'],
                               features['acousticness'], features['instrumentalness'],
                               features['time_signature'], features['danceability'],
                               features['key'], features['duration_ms'],
                               features['loudness'], features['valence'],
                               features['mode'], features['uri']]
    else:
        track_AudioFeatures = [None]*14


    #trackData = ( [ track_Name, track_ID, track_explicit] + track_AudioFeatures )
    trackData = ( [track_ID] + track_AudioFeatures )

    listOfTrack_AudioFeatures.append(trackData)

    #break
    #print(features)

    IDs_AlreadyUsed.append(track_ID)

    time.sleep(0.15)
    #print(i, i/63902)
    
    #i += 1
```

     20%|██        | 7875/39134 [52:31<3:28:28,  2.50it/s]


    ---------------------------------------------------------------------------

    TimeoutError                              Traceback (most recent call last)

    ~/anaconda3/lib/python3.5/site-packages/urllib3/connection.py in _new_conn(self)
        140             conn = connection.create_connection(
    --> 141                 (self.host, self.port), self.timeout, **extra_kw)
        142 


    ~/anaconda3/lib/python3.5/site-packages/urllib3/util/connection.py in create_connection(address, timeout, source_address, socket_options)
         82     if err is not None:
    ---> 83         raise err
         84 


    ~/anaconda3/lib/python3.5/site-packages/urllib3/util/connection.py in create_connection(address, timeout, source_address, socket_options)
         72                 sock.bind(source_address)
    ---> 73             sock.connect(sa)
         74             return sock


    TimeoutError: [Errno 60] Operation timed out

    
    During handling of the above exception, another exception occurred:


    NewConnectionError                        Traceback (most recent call last)

    ~/anaconda3/lib/python3.5/site-packages/urllib3/connectionpool.py in urlopen(self, method, url, body, headers, retries, redirect, assert_same_host, timeout, pool_timeout, release_conn, chunked, body_pos, **response_kw)
        600                                                   body=body, headers=headers,
    --> 601                                                   chunked=chunked)
        602 


    ~/anaconda3/lib/python3.5/site-packages/urllib3/connectionpool.py in _make_request(self, conn, method, url, timeout, chunked, **httplib_request_kw)
        345         try:
    --> 346             self._validate_conn(conn)
        347         except (SocketTimeout, BaseSSLError) as e:


    ~/anaconda3/lib/python3.5/site-packages/urllib3/connectionpool.py in _validate_conn(self, conn)
        849         if not getattr(conn, 'sock', None):  # AppEngine might not have  `.sock`
    --> 850             conn.connect()
        851 


    ~/anaconda3/lib/python3.5/site-packages/urllib3/connection.py in connect(self)
        283         # Add certificate verification
    --> 284         conn = self._new_conn()
        285 


    ~/anaconda3/lib/python3.5/site-packages/urllib3/connection.py in _new_conn(self)
        149             raise NewConnectionError(
    --> 150                 self, "Failed to establish a new connection: %s" % e)
        151 


    NewConnectionError: <urllib3.connection.VerifiedHTTPSConnection object at 0x1163739b0>: Failed to establish a new connection: [Errno 60] Operation timed out

    
    During handling of the above exception, another exception occurred:


    MaxRetryError                             Traceback (most recent call last)

    ~/anaconda3/lib/python3.5/site-packages/requests/adapters.py in send(self, request, stream, timeout, verify, cert, proxies)
        439                     retries=self.max_retries,
    --> 440                     timeout=timeout
        441                 )


    ~/anaconda3/lib/python3.5/site-packages/urllib3/connectionpool.py in urlopen(self, method, url, body, headers, retries, redirect, assert_same_host, timeout, pool_timeout, release_conn, chunked, body_pos, **response_kw)
        638             retries = retries.increment(method, url, error=e, _pool=self,
    --> 639                                         _stacktrace=sys.exc_info()[2])
        640             retries.sleep()


    ~/anaconda3/lib/python3.5/site-packages/urllib3/util/retry.py in increment(self, method, url, response, error, _pool, _stacktrace)
        387         if new_retry.is_exhausted():
    --> 388             raise MaxRetryError(_pool, url, error or ResponseError(cause))
        389 


    MaxRetryError: HTTPSConnectionPool(host='api.spotify.com', port=443): Max retries exceeded with url: /v1/audio-features/?ids=65wDjaVcrCCtlbqzP4xtzS (Caused by NewConnectionError('<urllib3.connection.VerifiedHTTPSConnection object at 0x1163739b0>: Failed to establish a new connection: [Errno 60] Operation timed out',))

    
    During handling of the above exception, another exception occurred:


    ConnectionError                           Traceback (most recent call last)

    <ipython-input-64-4a727bbe24b1> in <module>()
         20 
         21 
    ---> 22     features = sp.audio_features(track_ID)[0] # Get audio features for this specific track
         23 
         24     if features != None:


    ~/anaconda3/lib/python3.5/site-packages/spotipy/client.py in audio_features(self, tracks)
        824         if isinstance(tracks, str):
        825             trackid = self._get_id('track', tracks)
    --> 826             results = self._get('audio-features/?ids=' + trackid)
        827         else:
        828             tlist = [self._get_id('track', t) for t in tracks]


    ~/anaconda3/lib/python3.5/site-packages/spotipy/client.py in _get(self, url, args, payload, **kwargs)
        144         while retries > 0:
        145             try:
    --> 146                 return self._internal_call('GET', url, payload, kwargs)
        147             except SpotifyException as e:
        148                 retries -= 1


    ~/anaconda3/lib/python3.5/site-packages/spotipy/client.py in _internal_call(self, method, url, payload, params)
        106         if self.trace_out:
        107             print(url)
    --> 108         r = self._session.request(method, url, headers=headers, proxies=self.proxies, **args)
        109 
        110         if self.trace:  # pragma: no cover


    ~/anaconda3/lib/python3.5/site-packages/requests/sessions.py in request(self, method, url, params, data, headers, cookies, files, auth, timeout, allow_redirects, proxies, hooks, stream, verify, cert, json)
        506         }
        507         send_kwargs.update(settings)
    --> 508         resp = self.send(prep, **send_kwargs)
        509 
        510         return resp


    ~/anaconda3/lib/python3.5/site-packages/requests/sessions.py in send(self, request, **kwargs)
        616 
        617         # Send the request
    --> 618         r = adapter.send(request, **kwargs)
        619 
        620         # Total elapsed time of the request (approximately)


    ~/anaconda3/lib/python3.5/site-packages/requests/adapters.py in send(self, request, stream, timeout, verify, cert, proxies)
        506                 raise SSLError(e, request=request)
        507 
    --> 508             raise ConnectionError(e, request=request)
        509 
        510         except ClosedPoolError as e:


    ConnectionError: HTTPSConnectionPool(host='api.spotify.com', port=443): Max retries exceeded with url: /v1/audio-features/?ids=65wDjaVcrCCtlbqzP4xtzS (Caused by NewConnectionError('<urllib3.connection.VerifiedHTTPSConnection object at 0x1163739b0>: Failed to establish a new connection: [Errno 60] Operation timed out',))



```python
#listOfTrack_AudioFeatures = []

#IDs_AlreadyUsed = []

l1 = All_TrackIDs_InAnalysis

l2 = IDs_AlreadyUsed

TrackIDs_StillNotParsed = [x for x in l1 if x not in l2]
for track_ID in tqdm(TrackIDs_StillNotParsed):
    #sp.audio_analysis("2xmrfQpmS2iJExTlklLoAL")
    
    #track = sp.track(track_ID)
    #print(track_SP)
    
    #track_Name = track["name"]     

    #track_explicit = track['explicit']
    
    
    
    features = sp.audio_features(track_ID)[0] # Get audio features for this specific track

    if features != None:
    
        track_AudioFeatures = [features['energy'], features['liveness'],
                               features['tempo'], features['speechiness'],
                               features['acousticness'], features['instrumentalness'],
                               features['time_signature'], features['danceability'],
                               features['key'], features['duration_ms'],
                               features['loudness'], features['valence'],
                               features['mode'], features['uri']]
    else:
        track_AudioFeatures = [None]*14


    #trackData = ( [ track_Name, track_ID, track_explicit] + track_AudioFeatures )
    trackData = ( [track_ID] + track_AudioFeatures )

    listOfTrack_AudioFeatures.append(trackData)

    #break
    #print(features)

    IDs_AlreadyUsed.append(track_ID)

    time.sleep(0.15)
    #print(i, i/63902)
    
    #i += 1
```

    
      0%|          | 0/31259 [00:00<?, ?it/s][A
      0%|          | 1/31259 [00:00<6:42:01,  1.30it/s][A
      0%|          | 2/31259 [00:01<5:05:50,  1.70it/s][A
      0%|          | 3/31259 [00:01<4:32:31,  1.91it/s][A
      0%|          | 4/31259 [00:01<4:16:21,  2.03it/s][A
      0%|          | 5/31259 [00:02<4:06:46,  2.11it/s][A
      0%|          | 6/31259 [00:02<3:59:43,  2.17it/s][A
      0%|          | 7/31259 [00:03<3:59:56,  2.17it/s][A
      0%|          | 8/31259 [00:03<3:57:06,  2.20it/s][A
    Exception in thread Thread-12:
    Traceback (most recent call last):
      File "/Users/max/anaconda3/lib/python3.5/threading.py", line 914, in _bootstrap_inner
        self.run()
      File "/Users/max/anaconda3/lib/python3.5/site-packages/tqdm/_tqdm.py", line 144, in run
        for instance in self.tqdm_cls._instances:
      File "/Users/max/anaconda3/lib/python3.5/_weakrefset.py", line 60, in __iter__
        for itemref in self.data:
    RuntimeError: Set changed size during iteration
    
      1%|          | 344/31259 [02:20<3:30:02,  2.45it/s]



    ---------------------------------------------------------------------------

    TimeoutError                              Traceback (most recent call last)

    ~/anaconda3/lib/python3.5/site-packages/urllib3/connection.py in _new_conn(self)
        140             conn = connection.create_connection(
    --> 141                 (self.host, self.port), self.timeout, **extra_kw)
        142 


    ~/anaconda3/lib/python3.5/site-packages/urllib3/util/connection.py in create_connection(address, timeout, source_address, socket_options)
         82     if err is not None:
    ---> 83         raise err
         84 


    ~/anaconda3/lib/python3.5/site-packages/urllib3/util/connection.py in create_connection(address, timeout, source_address, socket_options)
         72                 sock.bind(source_address)
    ---> 73             sock.connect(sa)
         74             return sock


    TimeoutError: [Errno 60] Operation timed out

    
    During handling of the above exception, another exception occurred:


    NewConnectionError                        Traceback (most recent call last)

    ~/anaconda3/lib/python3.5/site-packages/urllib3/connectionpool.py in urlopen(self, method, url, body, headers, retries, redirect, assert_same_host, timeout, pool_timeout, release_conn, chunked, body_pos, **response_kw)
        600                                                   body=body, headers=headers,
    --> 601                                                   chunked=chunked)
        602 


    ~/anaconda3/lib/python3.5/site-packages/urllib3/connectionpool.py in _make_request(self, conn, method, url, timeout, chunked, **httplib_request_kw)
        345         try:
    --> 346             self._validate_conn(conn)
        347         except (SocketTimeout, BaseSSLError) as e:


    ~/anaconda3/lib/python3.5/site-packages/urllib3/connectionpool.py in _validate_conn(self, conn)
        849         if not getattr(conn, 'sock', None):  # AppEngine might not have  `.sock`
    --> 850             conn.connect()
        851 


    ~/anaconda3/lib/python3.5/site-packages/urllib3/connection.py in connect(self)
        283         # Add certificate verification
    --> 284         conn = self._new_conn()
        285 


    ~/anaconda3/lib/python3.5/site-packages/urllib3/connection.py in _new_conn(self)
        149             raise NewConnectionError(
    --> 150                 self, "Failed to establish a new connection: %s" % e)
        151 


    NewConnectionError: <urllib3.connection.VerifiedHTTPSConnection object at 0x114ba9c88>: Failed to establish a new connection: [Errno 60] Operation timed out

    
    During handling of the above exception, another exception occurred:


    MaxRetryError                             Traceback (most recent call last)

    ~/anaconda3/lib/python3.5/site-packages/requests/adapters.py in send(self, request, stream, timeout, verify, cert, proxies)
        439                     retries=self.max_retries,
    --> 440                     timeout=timeout
        441                 )


    ~/anaconda3/lib/python3.5/site-packages/urllib3/connectionpool.py in urlopen(self, method, url, body, headers, retries, redirect, assert_same_host, timeout, pool_timeout, release_conn, chunked, body_pos, **response_kw)
        638             retries = retries.increment(method, url, error=e, _pool=self,
    --> 639                                         _stacktrace=sys.exc_info()[2])
        640             retries.sleep()


    ~/anaconda3/lib/python3.5/site-packages/urllib3/util/retry.py in increment(self, method, url, response, error, _pool, _stacktrace)
        387         if new_retry.is_exhausted():
    --> 388             raise MaxRetryError(_pool, url, error or ResponseError(cause))
        389 


    MaxRetryError: HTTPSConnectionPool(host='api.spotify.com', port=443): Max retries exceeded with url: /v1/audio-features/?ids=4kK14radw0XfwxJDPt9tnP (Caused by NewConnectionError('<urllib3.connection.VerifiedHTTPSConnection object at 0x114ba9c88>: Failed to establish a new connection: [Errno 60] Operation timed out',))

    
    During handling of the above exception, another exception occurred:


    ConnectionError                           Traceback (most recent call last)

    <ipython-input-65-4a727bbe24b1> in <module>()
         20 
         21 
    ---> 22     features = sp.audio_features(track_ID)[0] # Get audio features for this specific track
         23 
         24     if features != None:


    ~/anaconda3/lib/python3.5/site-packages/spotipy/client.py in audio_features(self, tracks)
        824         if isinstance(tracks, str):
        825             trackid = self._get_id('track', tracks)
    --> 826             results = self._get('audio-features/?ids=' + trackid)
        827         else:
        828             tlist = [self._get_id('track', t) for t in tracks]


    ~/anaconda3/lib/python3.5/site-packages/spotipy/client.py in _get(self, url, args, payload, **kwargs)
        144         while retries > 0:
        145             try:
    --> 146                 return self._internal_call('GET', url, payload, kwargs)
        147             except SpotifyException as e:
        148                 retries -= 1


    ~/anaconda3/lib/python3.5/site-packages/spotipy/client.py in _internal_call(self, method, url, payload, params)
        106         if self.trace_out:
        107             print(url)
    --> 108         r = self._session.request(method, url, headers=headers, proxies=self.proxies, **args)
        109 
        110         if self.trace:  # pragma: no cover


    ~/anaconda3/lib/python3.5/site-packages/requests/sessions.py in request(self, method, url, params, data, headers, cookies, files, auth, timeout, allow_redirects, proxies, hooks, stream, verify, cert, json)
        506         }
        507         send_kwargs.update(settings)
    --> 508         resp = self.send(prep, **send_kwargs)
        509 
        510         return resp


    ~/anaconda3/lib/python3.5/site-packages/requests/sessions.py in send(self, request, **kwargs)
        616 
        617         # Send the request
    --> 618         r = adapter.send(request, **kwargs)
        619 
        620         # Total elapsed time of the request (approximately)


    ~/anaconda3/lib/python3.5/site-packages/requests/adapters.py in send(self, request, stream, timeout, verify, cert, proxies)
        506                 raise SSLError(e, request=request)
        507 
    --> 508             raise ConnectionError(e, request=request)
        509 
        510         except ClosedPoolError as e:


    ConnectionError: HTTPSConnectionPool(host='api.spotify.com', port=443): Max retries exceeded with url: /v1/audio-features/?ids=4kK14radw0XfwxJDPt9tnP (Caused by NewConnectionError('<urllib3.connection.VerifiedHTTPSConnection object at 0x114ba9c88>: Failed to establish a new connection: [Errno 60] Operation timed out',))



```python
#listOfTrack_AudioFeatures = []

#IDs_AlreadyUsed = []

l1 = All_TrackIDs_InAnalysis

l2 = IDs_AlreadyUsed

TrackIDs_StillNotParsed = [x for x in l1 if x not in l2]
for track_ID in tqdm(TrackIDs_StillNotParsed):
    #sp.audio_analysis("2xmrfQpmS2iJExTlklLoAL")
    
    #track = sp.track(track_ID)
    #print(track_SP)
    
    #track_Name = track["name"]     

    #track_explicit = track['explicit']
    
    
    
    features = sp.audio_features(track_ID)[0] # Get audio features for this specific track

    if features != None:
    
        track_AudioFeatures = [features['energy'], features['liveness'],
                               features['tempo'], features['speechiness'],
                               features['acousticness'], features['instrumentalness'],
                               features['time_signature'], features['danceability'],
                               features['key'], features['duration_ms'],
                               features['loudness'], features['valence'],
                               features['mode'], features['uri']]
    else:
        track_AudioFeatures = [None]*14


    #trackData = ( [ track_Name, track_ID, track_explicit] + track_AudioFeatures )
    trackData = ( [track_ID] + track_AudioFeatures )

    listOfTrack_AudioFeatures.append(trackData)

    #break
    #print(features)

    IDs_AlreadyUsed.append(track_ID)

    time.sleep(0.15)
    #print(i, i/63902)
    
    #i += 1
```

      4%|▍         | 1313/30915 [1:37:16<36:33:00,  4.45s/it]


    ---------------------------------------------------------------------------

    gaierror                                  Traceback (most recent call last)

    ~/anaconda3/lib/python3.5/site-packages/urllib3/connection.py in _new_conn(self)
        140             conn = connection.create_connection(
    --> 141                 (self.host, self.port), self.timeout, **extra_kw)
        142 


    ~/anaconda3/lib/python3.5/site-packages/urllib3/util/connection.py in create_connection(address, timeout, source_address, socket_options)
         59 
    ---> 60     for res in socket.getaddrinfo(host, port, family, socket.SOCK_STREAM):
         61         af, socktype, proto, canonname, sa = res


    ~/anaconda3/lib/python3.5/socket.py in getaddrinfo(host, port, family, type, proto, flags)
        732     addrlist = []
    --> 733     for res in _socket.getaddrinfo(host, port, family, type, proto, flags):
        734         af, socktype, proto, canonname, sa = res


    gaierror: [Errno 8] nodename nor servname provided, or not known

    
    During handling of the above exception, another exception occurred:


    NewConnectionError                        Traceback (most recent call last)

    ~/anaconda3/lib/python3.5/site-packages/urllib3/connectionpool.py in urlopen(self, method, url, body, headers, retries, redirect, assert_same_host, timeout, pool_timeout, release_conn, chunked, body_pos, **response_kw)
        600                                                   body=body, headers=headers,
    --> 601                                                   chunked=chunked)
        602 


    ~/anaconda3/lib/python3.5/site-packages/urllib3/connectionpool.py in _make_request(self, conn, method, url, timeout, chunked, **httplib_request_kw)
        345         try:
    --> 346             self._validate_conn(conn)
        347         except (SocketTimeout, BaseSSLError) as e:


    ~/anaconda3/lib/python3.5/site-packages/urllib3/connectionpool.py in _validate_conn(self, conn)
        849         if not getattr(conn, 'sock', None):  # AppEngine might not have  `.sock`
    --> 850             conn.connect()
        851 


    ~/anaconda3/lib/python3.5/site-packages/urllib3/connection.py in connect(self)
        283         # Add certificate verification
    --> 284         conn = self._new_conn()
        285 


    ~/anaconda3/lib/python3.5/site-packages/urllib3/connection.py in _new_conn(self)
        149             raise NewConnectionError(
    --> 150                 self, "Failed to establish a new connection: %s" % e)
        151 


    NewConnectionError: <urllib3.connection.VerifiedHTTPSConnection object at 0x1149b69e8>: Failed to establish a new connection: [Errno 8] nodename nor servname provided, or not known

    
    During handling of the above exception, another exception occurred:


    MaxRetryError                             Traceback (most recent call last)

    ~/anaconda3/lib/python3.5/site-packages/requests/adapters.py in send(self, request, stream, timeout, verify, cert, proxies)
        439                     retries=self.max_retries,
    --> 440                     timeout=timeout
        441                 )


    ~/anaconda3/lib/python3.5/site-packages/urllib3/connectionpool.py in urlopen(self, method, url, body, headers, retries, redirect, assert_same_host, timeout, pool_timeout, release_conn, chunked, body_pos, **response_kw)
        638             retries = retries.increment(method, url, error=e, _pool=self,
    --> 639                                         _stacktrace=sys.exc_info()[2])
        640             retries.sleep()


    ~/anaconda3/lib/python3.5/site-packages/urllib3/util/retry.py in increment(self, method, url, response, error, _pool, _stacktrace)
        387         if new_retry.is_exhausted():
    --> 388             raise MaxRetryError(_pool, url, error or ResponseError(cause))
        389 


    MaxRetryError: HTTPSConnectionPool(host='accounts.spotify.com', port=443): Max retries exceeded with url: /api/token (Caused by NewConnectionError('<urllib3.connection.VerifiedHTTPSConnection object at 0x1149b69e8>: Failed to establish a new connection: [Errno 8] nodename nor servname provided, or not known',))

    
    During handling of the above exception, another exception occurred:


    ConnectionError                           Traceback (most recent call last)

    <ipython-input-66-4a727bbe24b1> in <module>()
         20 
         21 
    ---> 22     features = sp.audio_features(track_ID)[0] # Get audio features for this specific track
         23 
         24     if features != None:


    ~/anaconda3/lib/python3.5/site-packages/spotipy/client.py in audio_features(self, tracks)
        824         if isinstance(tracks, str):
        825             trackid = self._get_id('track', tracks)
    --> 826             results = self._get('audio-features/?ids=' + trackid)
        827         else:
        828             tlist = [self._get_id('track', t) for t in tracks]


    ~/anaconda3/lib/python3.5/site-packages/spotipy/client.py in _get(self, url, args, payload, **kwargs)
        144         while retries > 0:
        145             try:
    --> 146                 return self._internal_call('GET', url, payload, kwargs)
        147             except SpotifyException as e:
        148                 retries -= 1


    ~/anaconda3/lib/python3.5/site-packages/spotipy/client.py in _internal_call(self, method, url, payload, params)
         98         if not url.startswith('http'):
         99             url = self.prefix + url
    --> 100         headers = self._auth_headers()
        101         headers['Content-Type'] = 'application/json'
        102 


    ~/anaconda3/lib/python3.5/site-packages/spotipy/client.py in _auth_headers(self)
         88             return {'Authorization': 'Bearer {0}'.format(self._auth)}
         89         elif self.client_credentials_manager:
    ---> 90             token = self.client_credentials_manager.get_access_token()
         91             return {'Authorization': 'Bearer {0}'.format(token)}
         92         else:


    ~/anaconda3/lib/python3.5/site-packages/spotipy/oauth2.py in get_access_token(self)
         55             return self.token_info['access_token']
         56 
    ---> 57         token_info = self._request_access_token()
         58         token_info = self._add_custom_values_to_token_info(token_info)
         59         self.token_info = token_info


    ~/anaconda3/lib/python3.5/site-packages/spotipy/oauth2.py in _request_access_token(self)
         72 
         73         response = requests.post(self.OAUTH_TOKEN_URL, data=payload,
    ---> 74             headers=headers, verify=True, proxies=self.proxies)
         75         if response.status_code is not 200:
         76             raise SpotifyOauthError(response.reason)


    ~/anaconda3/lib/python3.5/site-packages/requests/api.py in post(url, data, json, **kwargs)
        110     """
        111 
    --> 112     return request('post', url, data=data, json=json, **kwargs)
        113 
        114 


    ~/anaconda3/lib/python3.5/site-packages/requests/api.py in request(method, url, **kwargs)
         56     # cases, and look like a memory leak in others.
         57     with sessions.Session() as session:
    ---> 58         return session.request(method=method, url=url, **kwargs)
         59 
         60 


    ~/anaconda3/lib/python3.5/site-packages/requests/sessions.py in request(self, method, url, params, data, headers, cookies, files, auth, timeout, allow_redirects, proxies, hooks, stream, verify, cert, json)
        506         }
        507         send_kwargs.update(settings)
    --> 508         resp = self.send(prep, **send_kwargs)
        509 
        510         return resp


    ~/anaconda3/lib/python3.5/site-packages/requests/sessions.py in send(self, request, **kwargs)
        616 
        617         # Send the request
    --> 618         r = adapter.send(request, **kwargs)
        619 
        620         # Total elapsed time of the request (approximately)


    ~/anaconda3/lib/python3.5/site-packages/requests/adapters.py in send(self, request, stream, timeout, verify, cert, proxies)
        506                 raise SSLError(e, request=request)
        507 
    --> 508             raise ConnectionError(e, request=request)
        509 
        510         except ClosedPoolError as e:


    ConnectionError: HTTPSConnectionPool(host='accounts.spotify.com', port=443): Max retries exceeded with url: /api/token (Caused by NewConnectionError('<urllib3.connection.VerifiedHTTPSConnection object at 0x1149b69e8>: Failed to establish a new connection: [Errno 8] nodename nor servname provided, or not known',))



```python
#listOfTrack_AudioFeatures = []

#IDs_AlreadyUsed = []

l1 = All_TrackIDs_InAnalysis

l2 = IDs_AlreadyUsed

TrackIDs_StillNotParsed = [x for x in l1 if x not in l2]
for track_ID in tqdm(TrackIDs_StillNotParsed):
    #sp.audio_analysis("2xmrfQpmS2iJExTlklLoAL")
    
    #track = sp.track(track_ID)
    #print(track_SP)
    
    #track_Name = track["name"]     

    #track_explicit = track['explicit']
    
    
    
    features = sp.audio_features(track_ID)[0] # Get audio features for this specific track

    if features != None:
    
        track_AudioFeatures = [features['energy'], features['liveness'],
                               features['tempo'], features['speechiness'],
                               features['acousticness'], features['instrumentalness'],
                               features['time_signature'], features['danceability'],
                               features['key'], features['duration_ms'],
                               features['loudness'], features['valence'],
                               features['mode'], features['uri']]
    else:
        track_AudioFeatures = [None]*14


    #trackData = ( [ track_Name, track_ID, track_explicit] + track_AudioFeatures )
    trackData = ( [track_ID] + track_AudioFeatures )

    listOfTrack_AudioFeatures.append(trackData)

    #break
    #print(features)

    IDs_AlreadyUsed.append(track_ID)

    time.sleep(0.15)
    #print(i, i/63902)
    
    #i += 1
```

    
      0%|          | 0/29602 [00:00<?, ?it/s][A
      0%|          | 1/29602 [00:00<4:17:21,  1.92it/s][A
      0%|          | 2/29602 [00:00<3:32:56,  2.32it/s][A
      0%|          | 3/29602 [00:01<3:31:20,  2.33it/s][A
      0%|          | 4/29602 [00:01<3:19:37,  2.47it/s][A
      0%|          | 5/29602 [00:01<3:13:29,  2.55it/s][A
      0%|          | 6/29602 [00:02<3:10:45,  2.59it/s][A
      0%|          | 7/29602 [00:02<3:07:33,  2.63it/s][A
      0%|          | 8/29602 [00:02<3:04:14,  2.68it/s][A
      0%|          | 9/29602 [00:03<3:01:58,  2.71it/s][A
      0%|          | 10/29602 [00:03<3:00:23,  2.73it/s][A
      0%|          | 11/29602 [00:03<2:59:14,  2.75it/s][A
    Exception in thread Thread-13:
    Traceback (most recent call last):
      File "/Users/max/anaconda3/lib/python3.5/threading.py", line 914, in _bootstrap_inner
        self.run()
      File "/Users/max/anaconda3/lib/python3.5/site-packages/tqdm/_tqdm.py", line 144, in run
        for instance in self.tqdm_cls._instances:
      File "/Users/max/anaconda3/lib/python3.5/_weakrefset.py", line 60, in __iter__
        for itemref in self.data:
    RuntimeError: Set changed size during iteration
    
      8%|▊         | 2398/29602 [13:55<2:38:03,  2.87it/s]

    retrying ...1secs


     68%|██████▊   | 20069/29602 [1:57:56<56:01,  2.84it/s]  



    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    ~/anaconda3/lib/python3.5/site-packages/urllib3/connectionpool.py in _make_request(self, conn, method, url, timeout, chunked, **httplib_request_kw)
        379             try:  # Python 2.7, use buffering of HTTP responses
    --> 380                 httplib_response = conn.getresponse(buffering=True)
        381             except TypeError:  # Python 2.6 and older, Python 3


    TypeError: getresponse() got an unexpected keyword argument 'buffering'

    
    During handling of the above exception, another exception occurred:


    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-67-4a727bbe24b1> in <module>()
         20 
         21 
    ---> 22     features = sp.audio_features(track_ID)[0] # Get audio features for this specific track
         23 
         24     if features != None:


    ~/anaconda3/lib/python3.5/site-packages/spotipy/client.py in audio_features(self, tracks)
        824         if isinstance(tracks, str):
        825             trackid = self._get_id('track', tracks)
    --> 826             results = self._get('audio-features/?ids=' + trackid)
        827         else:
        828             tlist = [self._get_id('track', t) for t in tracks]


    ~/anaconda3/lib/python3.5/site-packages/spotipy/client.py in _get(self, url, args, payload, **kwargs)
        144         while retries > 0:
        145             try:
    --> 146                 return self._internal_call('GET', url, payload, kwargs)
        147             except SpotifyException as e:
        148                 retries -= 1


    ~/anaconda3/lib/python3.5/site-packages/spotipy/client.py in _internal_call(self, method, url, payload, params)
        106         if self.trace_out:
        107             print(url)
    --> 108         r = self._session.request(method, url, headers=headers, proxies=self.proxies, **args)
        109 
        110         if self.trace:  # pragma: no cover


    ~/anaconda3/lib/python3.5/site-packages/requests/sessions.py in request(self, method, url, params, data, headers, cookies, files, auth, timeout, allow_redirects, proxies, hooks, stream, verify, cert, json)
        506         }
        507         send_kwargs.update(settings)
    --> 508         resp = self.send(prep, **send_kwargs)
        509 
        510         return resp


    ~/anaconda3/lib/python3.5/site-packages/requests/sessions.py in send(self, request, **kwargs)
        616 
        617         # Send the request
    --> 618         r = adapter.send(request, **kwargs)
        619 
        620         # Total elapsed time of the request (approximately)


    ~/anaconda3/lib/python3.5/site-packages/requests/adapters.py in send(self, request, stream, timeout, verify, cert, proxies)
        438                     decode_content=False,
        439                     retries=self.max_retries,
    --> 440                     timeout=timeout
        441                 )
        442 


    ~/anaconda3/lib/python3.5/site-packages/urllib3/connectionpool.py in urlopen(self, method, url, body, headers, retries, redirect, assert_same_host, timeout, pool_timeout, release_conn, chunked, body_pos, **response_kw)
        599                                                   timeout=timeout_obj,
        600                                                   body=body, headers=headers,
    --> 601                                                   chunked=chunked)
        602 
        603             # If we're going to release the connection in ``finally:``, then


    ~/anaconda3/lib/python3.5/site-packages/urllib3/connectionpool.py in _make_request(self, conn, method, url, timeout, chunked, **httplib_request_kw)
        381             except TypeError:  # Python 2.6 and older, Python 3
        382                 try:
    --> 383                     httplib_response = conn.getresponse()
        384                 except Exception as e:
        385                     # Remove the TypeError from the exception chain in Python 3;


    ~/anaconda3/lib/python3.5/http/client.py in getresponse(self)
       1196         try:
       1197             try:
    -> 1198                 response.begin()
       1199             except ConnectionError:
       1200                 self.close()


    ~/anaconda3/lib/python3.5/http/client.py in begin(self)
        295         # read until we get a non-100 response
        296         while True:
    --> 297             version, status, reason = self._read_status()
        298             if status != CONTINUE:
        299                 break


    ~/anaconda3/lib/python3.5/http/client.py in _read_status(self)
        256 
        257     def _read_status(self):
    --> 258         line = str(self.fp.readline(_MAXLINE + 1), "iso-8859-1")
        259         if len(line) > _MAXLINE:
        260             raise LineTooLong("status line")


    ~/anaconda3/lib/python3.5/socket.py in readinto(self, b)
        574         while True:
        575             try:
    --> 576                 return self._sock.recv_into(b)
        577             except timeout:
        578                 self._timeout_occurred = True


    ~/anaconda3/lib/python3.5/site-packages/urllib3/contrib/pyopenssl.py in recv_into(self, *args, **kwargs)
        278     def recv_into(self, *args, **kwargs):
        279         try:
    --> 280             return self.connection.recv_into(*args, **kwargs)
        281         except OpenSSL.SSL.SysCallError as e:
        282             if self.suppress_ragged_eofs and e.args == (-1, 'Unexpected EOF'):


    ~/anaconda3/lib/python3.5/site-packages/OpenSSL/SSL.py in recv_into(self, buffer, nbytes, flags)
       1622             result = _lib.SSL_peek(self._ssl, buf, nbytes)
       1623         else:
    -> 1624             result = _lib.SSL_read(self._ssl, buf, nbytes)
       1625         self._raise_ssl_error(self._ssl, result)
       1626 


    KeyboardInterrupt: 



```python
#listOfTrack_AudioFeatures = []

#IDs_AlreadyUsed = []

l1 = All_TrackIDs_InAnalysis

l2 = IDs_AlreadyUsed

TrackIDs_StillNotParsed = [x for x in l1 if x not in l2]
for track_ID in tqdm(TrackIDs_StillNotParsed):
    #sp.audio_analysis("2xmrfQpmS2iJExTlklLoAL")
    
    #track = sp.track(track_ID)
    #print(track_SP)
    
    #track_Name = track["name"]     

    #track_explicit = track['explicit']
    
    
    
    features = sp.audio_features(track_ID)[0] # Get audio features for this specific track

    if features != None:
    
        track_AudioFeatures = [features['energy'], features['liveness'],
                               features['tempo'], features['speechiness'],
                               features['acousticness'], features['instrumentalness'],
                               features['time_signature'], features['danceability'],
                               features['key'], features['duration_ms'],
                               features['loudness'], features['valence'],
                               features['mode'], features['uri']]
    else:
        track_AudioFeatures = [None]*14


    #trackData = ( [ track_Name, track_ID, track_explicit] + track_AudioFeatures )
    trackData = ( [track_ID] + track_AudioFeatures )

    listOfTrack_AudioFeatures.append(trackData)

    #break
    #print(features)

    IDs_AlreadyUsed.append(track_ID)

    time.sleep(0.05)
    #print(i, i/63902)
    
    #i += 1
```

    100%|██████████| 9533/9533 [1:05:59<00:00,  2.41it/s]



```python
audioFeaturesDF_AllTracks = pd.DataFrame(listOfTrack_AudioFeatures, 
                                columns=  [ "track_id",  #["track_name", "track_id", "explicit",
                                            'energy', 'liveness',
                                            'tempo', 'speechiness',
                                            'acousticness', 'instrumentalness',
                                            'time_signature', 'danceability',
                                            'key', 'duration_ms', 'loudness',
                                            'valence', 'mode', 'track_uri'])
```


```python
audioFeaturesDF_AllTracks.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>track_id</th>
      <th>energy</th>
      <th>liveness</th>
      <th>tempo</th>
      <th>speechiness</th>
      <th>acousticness</th>
      <th>instrumentalness</th>
      <th>time_signature</th>
      <th>danceability</th>
      <th>key</th>
      <th>duration_ms</th>
      <th>loudness</th>
      <th>valence</th>
      <th>mode</th>
      <th>track_uri</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4WhFiV51QCIM5fa7ws4WZA</td>
      <td>0.8280</td>
      <td>0.2880</td>
      <td>115.648</td>
      <td>0.0521</td>
      <td>0.0245</td>
      <td>0.0108</td>
      <td>4.0</td>
      <td>0.566</td>
      <td>0.0</td>
      <td>320227.0</td>
      <td>-6.800</td>
      <td>0.737</td>
      <td>1.0</td>
      <td>spotify:track:4WhFiV51QCIM5fa7ws4WZA</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3SWZ9fHtWMxwkFok5qhhpO</td>
      <td>0.8300</td>
      <td>0.0882</td>
      <td>140.732</td>
      <td>0.0954</td>
      <td>0.5390</td>
      <td>0.0000</td>
      <td>4.0</td>
      <td>0.689</td>
      <td>8.0</td>
      <td>171560.0</td>
      <td>-8.774</td>
      <td>0.815</td>
      <td>1.0</td>
      <td>spotify:track:3SWZ9fHtWMxwkFok5qhhpO</td>
    </tr>
    <tr>
      <th>2</th>
      <td>16i6f7yJWs1j67fxBjfc7z</td>
      <td>0.4790</td>
      <td>0.0808</td>
      <td>118.117</td>
      <td>0.0279</td>
      <td>0.7470</td>
      <td>0.3100</td>
      <td>4.0</td>
      <td>0.544</td>
      <td>3.0</td>
      <td>302880.0</td>
      <td>-10.878</td>
      <td>0.434</td>
      <td>1.0</td>
      <td>spotify:track:16i6f7yJWs1j67fxBjfc7z</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1rAPOKdbcenPjBnDjqZu9Z</td>
      <td>0.2670</td>
      <td>0.1060</td>
      <td>96.975</td>
      <td>0.0284</td>
      <td>0.7200</td>
      <td>0.9130</td>
      <td>4.0</td>
      <td>0.385</td>
      <td>5.0</td>
      <td>218867.0</td>
      <td>-15.564</td>
      <td>0.155</td>
      <td>1.0</td>
      <td>spotify:track:1rAPOKdbcenPjBnDjqZu9Z</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6v8oJ72mgI06BodWqXmEw7</td>
      <td>0.0867</td>
      <td>0.3450</td>
      <td>93.951</td>
      <td>0.9200</td>
      <td>0.9200</td>
      <td>0.0000</td>
      <td>3.0</td>
      <td>0.658</td>
      <td>0.0</td>
      <td>1895547.0</td>
      <td>-24.962</td>
      <td>0.239</td>
      <td>1.0</td>
      <td>spotify:track:6v8oJ72mgI06BodWqXmEw7</td>
    </tr>
  </tbody>
</table>
</div>




```python
audioFeaturesDF_AllTracks.shape
```




    (63902, 15)




```python
audioFeaturesDF_AllTracks.to_csv("./Track_AudioFeatures_AllTracks.csv")
```


```python
audioFeaturesDF_AllTracks_1 = pd.read_csv("./Track_AudioFeatures_AllTracks.csv").drop(["Unnamed: 0"], axis = 1)
```


```python
audioFeaturesDF_AllTracks_1.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>track_id</th>
      <th>energy</th>
      <th>liveness</th>
      <th>tempo</th>
      <th>speechiness</th>
      <th>acousticness</th>
      <th>instrumentalness</th>
      <th>time_signature</th>
      <th>danceability</th>
      <th>key</th>
      <th>duration_ms</th>
      <th>loudness</th>
      <th>valence</th>
      <th>mode</th>
      <th>track_uri</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4WhFiV51QCIM5fa7ws4WZA</td>
      <td>0.8280</td>
      <td>0.2880</td>
      <td>115.648</td>
      <td>0.0521</td>
      <td>0.0245</td>
      <td>0.0108</td>
      <td>4.0</td>
      <td>0.566</td>
      <td>0.0</td>
      <td>320227.0</td>
      <td>-6.800</td>
      <td>0.737</td>
      <td>1.0</td>
      <td>spotify:track:4WhFiV51QCIM5fa7ws4WZA</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3SWZ9fHtWMxwkFok5qhhpO</td>
      <td>0.8300</td>
      <td>0.0882</td>
      <td>140.732</td>
      <td>0.0954</td>
      <td>0.5390</td>
      <td>0.0000</td>
      <td>4.0</td>
      <td>0.689</td>
      <td>8.0</td>
      <td>171560.0</td>
      <td>-8.774</td>
      <td>0.815</td>
      <td>1.0</td>
      <td>spotify:track:3SWZ9fHtWMxwkFok5qhhpO</td>
    </tr>
    <tr>
      <th>2</th>
      <td>16i6f7yJWs1j67fxBjfc7z</td>
      <td>0.4790</td>
      <td>0.0808</td>
      <td>118.117</td>
      <td>0.0279</td>
      <td>0.7470</td>
      <td>0.3100</td>
      <td>4.0</td>
      <td>0.544</td>
      <td>3.0</td>
      <td>302880.0</td>
      <td>-10.878</td>
      <td>0.434</td>
      <td>1.0</td>
      <td>spotify:track:16i6f7yJWs1j67fxBjfc7z</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1rAPOKdbcenPjBnDjqZu9Z</td>
      <td>0.2670</td>
      <td>0.1060</td>
      <td>96.975</td>
      <td>0.0284</td>
      <td>0.7200</td>
      <td>0.9130</td>
      <td>4.0</td>
      <td>0.385</td>
      <td>5.0</td>
      <td>218867.0</td>
      <td>-15.564</td>
      <td>0.155</td>
      <td>1.0</td>
      <td>spotify:track:1rAPOKdbcenPjBnDjqZu9Z</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6v8oJ72mgI06BodWqXmEw7</td>
      <td>0.0867</td>
      <td>0.3450</td>
      <td>93.951</td>
      <td>0.9200</td>
      <td>0.9200</td>
      <td>0.0000</td>
      <td>3.0</td>
      <td>0.658</td>
      <td>0.0</td>
      <td>1895547.0</td>
      <td>-24.962</td>
      <td>0.239</td>
      <td>1.0</td>
      <td>spotify:track:6v8oJ72mgI06BodWqXmEw7</td>
    </tr>
  </tbody>
</table>
</div>




```python
audioFeaturesDF_AllTracks_1.shape
```




    (63902, 15)




```python
#audioFeaturesDF_Inter = pd.DataFrame(listOfTrack_AudioFeatures, 
#                                columns=  [ "track_id",  #["track_name", "track_id", "explicit",
#                                            'energy', 'liveness',
#                                            'tempo', 'speechiness',
#                                            'acousticness', 'instrumentalness',
#                                            'time_signature', 'danceability',
#                                            'key', 'duration_ms', 'loudness',
#                                            'valence', 'mode', 'track_uri'])


#audioFeaturesDF_Inter.shape
#audioFeaturesDF_Inter.to_csv("./Track_AudioFeatures_Intermediate.csv")
#audioFeaturesDF_Inter.head()
```
