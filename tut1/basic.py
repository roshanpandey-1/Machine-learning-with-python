import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
music_data = pd.read_csv('F:\\Roshan\\machine learning\\tut1\\music.csv')

# Split into features and target
x = music_data.drop(columns=['genre'])
y = music_data['genre']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Train model
model = DecisionTreeClassifier()
model.fit(x_train, y_train)
predictions = model.predict(x_test)

score =accuracy_score(y_test, predictions)  # Check accuracy of the model
print(score)

