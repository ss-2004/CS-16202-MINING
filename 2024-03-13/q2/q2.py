import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('weatherNumeric.csv')

X = df.iloc[:, :-1]
Y = df.iloc[:, -1]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

for i in range(1, 6):
    train_file = f'train{i}.csv'
    test_file = f'test{i}.csv'
    
    X_train_split, X_test_split, Y_train_split, Y_test_split = train_test_split(X, Y, test_size=0.2, random_state=i*10)
    
    train_data = pd.concat([X_train_split, Y_train_split], axis=1)
    test_data = pd.concat([X_test_split, Y_test_split], axis=1)
    
    train_data.to_csv(train_file, index=False)
    test_data.to_csv(test_file, index=False)

print("X_train:")
print(X_train.head())
print("\nX_test:")
print(X_test.head())
print("\nY_train:")
print(Y_train.head())
print("\nY_test:")
print(Y_test.head())
