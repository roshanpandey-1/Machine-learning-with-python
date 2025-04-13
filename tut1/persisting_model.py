import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

# # Load dataset
# music_data = pd.read_csv('F:\\Roshan\\machine learning\\tut1\\music.csv')

# # Split into features and target
# x = music_data.drop(columns=['genre'])
# y = music_data['genre']

# model = DecisionTreeClassifier()
# model.fit(x, y)

model = joblib.load('music-recommender.joblib')  # Save the model to a file
predictions = model.predict([[21, 1]])  # Predict the genre for a new data point
print(predictions)  # Print the predicted genre

