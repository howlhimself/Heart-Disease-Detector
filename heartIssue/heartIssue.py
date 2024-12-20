import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
import kagglehub


path = kagglehub.dataset_download("johnsmith88/heart-disease-dataset")
print("Path to dataset files:", path)


csv_path = f"{path}/heart.csv" 
data = pd.read_csv(csv_path)

print(data.head())


x = data.drop(columns=['target'])  
y = data['target']


scaler = StandardScaler()
x = scaler.fit_transform(x)


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = Sequential([
    Dense(units=128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(units=64, activation='relu'),
    Dense(units=32, activation='relu'),
    Dense(units=1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


model.fit(X_train, y_train, epochs=20, batch_size=32)


loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")


model.save('heartIssueFinder.model')

feature_names = [
    "age",
    "sex",
    "chest pain type (4 values)",
    "resting blood pressure",
    "serum cholestoral in mg/dl",
    "fasting blood sugar > 120 mg/dl",
    "resting electrocardiographic results (values 0,1,2)",
    "maximum heart rate achieved",
    "exercise induced angina",
    "oldpeak (ST depression induced by exercise relative to rest)",
    "slope of the peak exercise ST segment",
    "number of major vessels (0-3) colored by fluoroscopy",
    "thal (0 = normal; 1 = fixed defect; 2 = reversible defect)"
]


user_input = []
for feature in feature_names:
    value = float(input(f"Enter value for {feature}: "))
    user_input.append(value)

input_array = np.array(user_input).reshape(1, -1)

prediction = model.predict(input_array)

if prediction[0][0] > 0.5: 
    print("The model predicts a heart issue (positive).")
else:
    print("The model predicts no heart issue (negative).")