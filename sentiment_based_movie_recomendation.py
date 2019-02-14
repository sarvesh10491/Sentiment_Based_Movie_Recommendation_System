import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

print("Loading System! Please wait.....")

scores = pd.read_csv(r'./ml-latest/genome-scores.csv') # Reading datasets
mvnames = pd.read_csv(r'./ml-latest/movies.csv')

# Data processing. We are rejecting the tags for which mean relevance is less than threshold descried by quantile(0.4)
g_new = scores.groupby(['tagId']).agg({"relevance": ["mean", "std"]})

qt = g_new.relevance.quantile(0.4)         # Threshold for tag rejection

tag_list =[]                               # Filtering the tags
for i in range(1,1128):
    if g_new['relevance']['mean'].iloc[i-1] > qt[0]:
        tag_list.append(i)

dataf_tags = pd.DataFrame(tag_list,columns=['tagId'])
tags_array = np.asarray(dataf_tags).ravel()


scor = pd.merge(scores, dataf_tags, on=['tagId'], how='inner') # Join with main dataset based on reduced tags criteria

scorar = np.asarray(scor)

res = np.reshape(scorar,[677,13176,3])

trans=np.transpose(res)

feature_mat = trans[2]  # Creating a numpy matrix with row as objects and columns as 677 features

movieids = np.transpose(trans[0])[0] # List of movieids in final cleaned dataset
movieids = list(movieids)

neigh = NearestNeighbors(n_neighbors=4,algorithm='ball_tree').fit(feature_mat) # Nearest neighbor algorithm to get similar movies based on distance in feature space

mvframes = pd.DataFrame(movieids,columns=['movieId'])
movienames = pd.merge(mvnames, mvframes, on=['movieId'], how='inner')

emotion_labels = ['joy', 'fear', 'anger', 'sadness', 'disgust', 'shame', 'guilt'] # List of possible input emotions

emotion_dict = {}

for i in range(len(emotion_labels)): #Mapping between emotion and number which will be used to index in lists 
    emotion_dict[emotion_labels[i]] = i

buckets = []      # List of emotion buckets
for i in emotion_labels:
    buckets.append([]) # Each bucket is a list of movies for that emotion

# Pre filled lists with our recommendations
happylist = [movienames.iloc[46],movienames.iloc[50],movienames.iloc[52]]
sadlist = [movienames.iloc[4],movienames.iloc[18],movienames.iloc[19]]
fearlist = [movienames.iloc[68],movienames.iloc[89],movienames.iloc[90]]
angerlist = [movienames.iloc[91],movienames.iloc[103],movienames.iloc[111]]
disgustlist = [movienames.iloc[141],movienames.iloc[145],movienames.iloc[151]]
shamelist = [movienames.iloc[154],movienames.iloc[156],movienames.iloc[165]]
guiltlist = [movienames.iloc[462],movienames.iloc[473],movienames.iloc[487]]

buckets[emotion_dict['joy']] = happylist
buckets[emotion_dict['sadness']] = sadlist
buckets[emotion_dict['fear']] = fearlist
buckets[emotion_dict['anger']] = angerlist
buckets[emotion_dict['disgust']] = disgustlist
buckets[emotion_dict['shame']] = shamelist
buckets[emotion_dict['guilt']] = guiltlist

bucket_index = np.zeros((7,), dtype=int)

user_input = 1

# Takes user's mood on terminal for demo purpose only. Actually emotion input comes from Naive Bayes' model 3rd party code that extracts
# emotion based on sentiment analysis of textual input data
while(user_input != -1): 
	emotion = input("\n\nHow are you feeling?\nChoose from this:\n joy\n sadness\n fear\n anger\n disgust\n shame\n guilt\n\nEnter here: ")
	recommendations = []
	for i in buckets[emotion_dict[emotion]]:
	    dist,indices = neigh.kneighbors(np.reshape(feature_mat[movieids.index(i['movieId'])],[1,677])) #Movies similar to that in bucket
	    for j in range(1,len(indices[0])):
        	recommendations.append(movienames[movienames.movieId==movieids[indices[0][j]]])
	print("Our recommendations for you: \n\n")    
	print(recommendations)


	user_input = int(input("\n\nSelect which movieId did you see?(Enter -1 to exit!): \t")) # Movie that user actually watches
	if user_input != -1:
		print(movienames.iloc[movieids.index(user_input)])
		change = True
		for movies in buckets[emotion_dict[emotion]]:
		    if movies.movieId == user_input:
		        change = False

		# Replaces bucket with movie actually watched by user for that emotion hence tuning model according to user's preferences.
		if change:
		    buckets[emotion_dict[emotion]][bucket_index[emotion_dict[emotion]]] = movienames.iloc[movieids.index(user_input)]
		    bucket_index[emotion_dict[emotion]] = (bucket_index[emotion_dict[emotion]] +1)%3