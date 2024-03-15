# These are the standard models used in the publication "Generalizability Assessment of AI Models Across Hospitals: A Comparative Study in Low-Middle Income and High Income Countries" (https://www.medrxiv.org/content/10.1101/2023.11.05.23298109v1).

# Data from OUH studied here are available from the Infections in Oxfordshire Research Database (https://oxfordbrc.nihr.ac.uk/research-themes/modernising-medical-microbiology-and-big-infection-diagnostics/infections-in-oxfordshire-research-database-iord/), subject to an application meeting the ethical and governance requirements of the Database. Data from UHB, PUH and BH are available on reasonable request to the respective trusts, subject to HRA requirements.

# Data from HTD and NHTD are available from the CCAA Vietnam Data Access Committee, subject to an application meeting the ethical and governance requirements.

# Load dataset

#This script automatically loads the Adult (Census) dataset (replace with your own data)

import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Define column names
column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

# Load dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
df = pd.read_csv(url, header=None, names=column_names)

# Display the first few rows of the dataset
print(df.head())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Encode categorical variables
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Define features (X) and target variable (y)
X = df.drop('income', axis=1)  # Features
y = df['income']  # Target variable

# Normalize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Print the shapes of the training and testing sets
print("Training set shape:", X_train.shape, y_train.shape)
print("Testing set shape:", X_test.shape, y_test.shape)

# Logistic Regression
print("Logistic regression")

model = LogisticRegression()

model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict_proba(X_test)[:, 1]

# Calculate accuracy
auc = roc_auc_score(y_test, y_pred)
print("Log Reg AUC:", auc)

# XGBoost
print("XGBoost")

model = XGBClassifier()

model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict_proba(X_test)[:, 1]

# Calculate accuracy
auc = roc_auc_score(y_test, y_pred)
print("XGB AUC:", auc)

# Neural Network
print("Neural Network")

X = torch.tensor(np.array(X_train), dtype=torch.float32)

y_train = np.array(y_train).reshape(-1, 1)

# define the model
num_hidden_nodes = 10
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(X_train.shape[1], num_hidden_nodes)
        self.act1 = nn.ReLU()
        self.output = nn.Linear(num_hidden_nodes, 1)
        self.act_output = nn.Sigmoid()
 
    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act_output(self.output(x))
        return x

model = Classifier()

n_epochs = 10
batch_size = 16

# train the model
loss_fn   = nn.BCELoss()  # binary cross entropy

optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(n_epochs):
    #print(ncr_weight)
    for i in range(0, len(X_train), batch_size):
        Xbatch = X[i:i+batch_size]
        ybatch = y_train[i:i+batch_size]
        Xbatch=torch.tensor(Xbatch)
        ybatch=torch.tensor(ybatch)
        y_pred = model(Xbatch)
        y_pred = y_pred[:,0]
        loss = loss_fn(y_pred.to(torch.float), ybatch[:,0].to(torch.float))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Finished epoch {epoch}, latest loss {loss}')

torch.save({'epoch': epoch,
            'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,}, 'model_checkpoint.pt')

X_test = torch.tensor(X_test, dtype=torch.float32)
#y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)
y_pred = model(X_test).detach().numpy()[:,0]

# Calculate accuracy
auc = roc_auc_score(y_test, y_pred)
print("NN AUC:", auc)


