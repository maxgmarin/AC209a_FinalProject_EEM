---
layout: page
title: AC209a Final Project: Predictive modeling of playlist success
tagline: Predictive modeling of playlist success
description: Predictive modeling of playlist success
---

# AC209a Final Report: Predicting playlist success on the Spotify platform

#### Group Members: Max, Erica, and Elmer



Hello, welcome to our website for our data science final project.


<br>
<br>
<br>

## Table of contents
1. [Problem Statement and Motivation](#Problem_Statement)

2. [Introduction and Description of Data](#introduction)

    1. [Introduction](#intro1)

    2. [Exploratory Data Analysis](#EDA)


3. [Literature Review/Related Work](#paragraph2)

4. [Modeling Approach and Project Trajectory](#paragraph2)



5. [Results, Conclusions, and Future Work](#paragraph2)
    1. [Results](#subparagraph1)
    2. [Conclusions](#subparagraph1)
    3. [Future Work](#subparagraph1)


6. [References](#paragraph2)


## 1. Problem Statement and Motivation <a name="Problem_Statement"></a>
-  Goal: Our primary goal is to construct a predictive model for playlist success (as measured by # of followers).

-  Talk about: Spotify wants to keep users interested in their platform. A way they plan to achieve this is by continuously providing “fresh” content to the user. This keeps the user happy because they are discovering new music they like. 

## 2. Introduction and Description of Data <a name="paragraph1"></a>
Some text


Here is the Jupyter notebook where all audio features were obtained through the Spotify API: <br>
[ParsingAll_AudioFeatures_120417](notebook_Markdown/ParsingAll_AudioFeatures_120417.html)


### 2. Exploratory Data Analysis <a name="EDA1"></a>

![TestPlot](images/test.png)</br>
This is the distribution of the number of followers for each playlist. We can observe that the distribution is left-skewed, and therefore, requires additional transformations before using it as a response variable. 

</br>
The Today’s Top Hits is an outlier. It has more than twice the number of followers of RapCavier which is second in number of followers. (Note: there were some playlists that we couldn’t retrieved from spotify api so some of the top playlists may not be present here).

### Sub paragraph <a name="subparagraph1"></a>
This is a sub paragraph, formatted in heading 3 style


## 3. Literature Review/Related Work <a name="literature"></a>
https://towardsdatascience.com/is-my-spotify-music-boring-an-analysis-involving-music-data-and-machine-learning-47550ae931de 

http://spotipy.readthedocs.io/en/latest/#installation 

https://developer.spotify.com/web-api/tutorial/ 

http://www.sciencedirect.com/science/article/pii/S0020025507004045

https://github.com/perrygeo/simanneal/blob/master/simanneal/anneal.py

## 4. Modeling Approach and Project Trajectory <a name="approach"></a>

## 5. Results, Conclusions, and Future Work <a name="results future"></a>
### Results <a name="results"></a>
#### Building a playlist <a name="building_playlist"></a>
##### Introduction <a name="introduc"></a>
To suggest a new playlist to a user, we implemented an algorithm called Simulated Annealing. This is a global optimization approach inspired from a concept in statistical physics. The algorithm essentially describes the motion of molecules when it is heated up to large temperature and when the system is cooled. The idea here is that the molecules will reach a global minimum energy point after cooling, and hence, we can use it a global optimization method.  
##### Method <a name="methoda"></a>
We can adopt this global minimization algorithm to minimize a predefined cost function (in physics it’ll be the energy of the system defined by the boltzmann distribution). The cost function essentially forces us to incorporate tracks that maximizes the chance of success. For sake of simplicity and computational time, we only used audio features for fitting our model and running this algorithm. We’ve taken a pre-existing code for ‘Simulated Annealing’ and adapted it for playlists generation. 
In this method, we first fit a logistic regression model with our training set, and from our test set, we randomly generate a playlist with about 30-35 tracks in it. We use this playlist and other initial conditions to initiate the algorithm. The algorithm will go through a number of iterations replacing a track in our playlist with another track from our set of remaining tracks. The algorithm will accept the new playlist only if it lowers our cost function. At the end, the algorithm will return a playlist that minimizes the overall cost function. 

##### Results <a name="Resultsa"></a>

ADD IMAGE</br>

ADD IMAGE</br>
We can see that our new playlist takes tracks from playlists with relatively low followers if we modify the cost function so that it penalizes more for playlists with large number of followers (blue bar graphs above). We can also see that there are some tracks from the same playlist that was used in the old playlist. This shows that the algorithm doesn’t replace the playlist completely, but I tries to retain certain characters. However, the mean popularity of the tracks in our playlist increased slightly. 
If we adjust the cost function so that it penalizes more when the playlist is not as successful, we can see that the tracks are obtained from playlists with large number of followers. Note: that the initial playlist used for the two cases were different from each other due to kernel restarting.

ADD IMAGE</br>

ADD IMAGE</br>


These plots show that regardless of the cost function the algorithm seeks to generate playlists that have similar mean track popularity.

### Conclusion <a name="conclusion"></a>

### Future Work <a name="futurework"></a>
#### Improving our predictive model for playlist success: <a name="predictivemodel"></a>
TEXT

#### Building a Playlist: <a name="buildplaylist"></a>
The approach we’ve taken was somewhat naive. We worked under the assumption that the spotify user has an eclectic taste for tracks which is why we seeded a randomly generated playlist to our algorithm. In reality, we know individuals want tracks related to the ones they liked. Therefore, for future work, we should limit our tracks we incorporate to our playlist to ones that are related to each other. The similar tracks can be obtained through spotify recommendations or we can take a mathematical approach in which calculate the “distance” between tracks using metrics such as Frobenius Distance. Furthermore, there are several parameters that we can tweak to generate a better playlist such as the “temperature” and “cooling length”. For example, increasing the cooling time will give our system enough time to find the global minimum. Finally, the model we used for generating the playlist just uses a subset of all our features. In the future, we can rerun this algorithm with an expanded feature and also try out different classification models other than the one used in the algorithm.

## Another paragraph <a name="paragraph2"></a>
The second paragraph text
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
- [Project Overview](pages/overview.html)
- [Data sources/processing](pages/independent_site.html)
- [Exploratory Data Analysis of Spotify Data](pages/user_site.html)
- [Exploratory Data Analysis of Spotify Data](pages/project_site.html)
- [Building a baseline model to predict playlist success](pages/nojekyll.html)
- [Improving out model](pages/local_test.html)
- [Prediction of song genre based on audio features](pages/resources.html)

















