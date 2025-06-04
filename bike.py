from ucimlrepo import fetch_ucirepo 
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Input, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# fetch dataset 
seoul_bike_sharing_demand = fetch_ucirepo(id=560) 
  
# data (as pandas dataframes) 
X = seoul_bike_sharing_demand.data.features 
y = seoul_bike_sharing_demand.data.targets 

# variable information 
print(seoul_bike_sharing_demand.variables) 

df= pd.concat([X, y], axis=1)
df = df.drop(columns=['Date'])
X = df.drop(columns=['Functioning Day'])
y = df['Functioning Day']

numerical_features = X.select_dtypes(include=['int64']).columns.tolist()
categorical_features =  X.select_dtypes(include=['object']).columns.tolist()

#Preprocess the data
numerical_transformers = Pipeline(steps=[
    ('scaler', StandardScaler())
])

catergorical_transformers = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformers, numerical_features),
    ('cat', catergorical_transformers, categorical_features)
])

labelEnc = LabelEncoder()
y = labelEnc.fit_transform(y)

#Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

print(X_train.shape)

#Build the DNN
model = Sequential([
    Input(shape=(X_train.shape[1],)),  # dynamic input shape
    Dense(50, activation='relu'),
    Dense(50, activation='relu'),
    Dense(50, activation='relu'),
    Dropout(0.3),
    Dense(50, activation='relu'),
    Dense(50, activation='relu'),
    Dense(1, activation='sigmoid')  # Sigmoid activation function because label is binary
])

model.summary()
model.compile(optimizer='adam',
              loss='binary_crossentropy',  
              metrics=['accuracy'])
early_stop = EarlyStopping(patience=3, restore_best_weights=True)
history = model.fit(X_train, 
                    y_train, 
                    epochs=10, 
                    validation_split=0.2,
                    callbacks=[early_stop])
test_loss, test_acc = model.evaluate(X_test, y_test)

# Accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

y_pred = model.predict(X_test).argmax(axis=1)

cm = confusion_matrix(y_test, y_pred)
class_names = np.arange(len(labelEnc.classes_))

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

print(classification_report(y_test, y_pred, target_names=class_names))