import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# Load dataset
music_data = pd.read_csv('F:\\Roshan\\machine learning\\tut1\\music.csv')

# Split into features and target
x = music_data.drop(columns=['genre'])
y = music_data['genre']

model = DecisionTreeClassifier()
model.fit(x, y)

tree.export_graphviz(model, out_file='music-recommender.dot', 
                     feature_names=['age','gender'],
                       class_names=sorted(y.unique()),
                       label='all',rounded=True, filled=True)  # Save the model to a file