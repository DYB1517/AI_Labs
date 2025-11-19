import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

df = pd.read_csv('../spaceship_titanic.csv')
df.head()
df.info()

print(df)
cols = df.columns
mtx = df.isnull()
print(mtx.sum())

vect_mod = ['HomePlanet', 'Cabin', 'Destination']
vect_med = ['Age',  'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
vect_mean = ['VIP','CryoSleep']

for col in vect_mod:
    if col:
        df[col] = df[col].fillna(df[col].mode()[0])
for col in vect_med:
    if col:
        if col:
            df[col] = df[col].fillna(df[col].median())
for col in vect_mean:
    df[col] = df[col].fillna(df[col].mean())
    df[col] = df[col].astype(bool)

df['VIP'] = df['VIP'].astype(int)

scaler = MinMaxScaler()

vect_sc = ['RoomService','FoodCourt','ShoppingMall', 'Spa']

for col in vect_sc:
    df[col] = scaler.fit_transform(df[[col]])
df = pd.get_dummies(df, columns=['Destination', 'HomePlanet', 'Cabin'], drop_first=True)

df = df.drop('PassengerId', axis='columns')
df = df.drop('Name', axis='columns')

ed_mtx = df.isnull()
print(ed_mtx.sum())
print(df)

df.to_csv('../spaceship_titanic_changed.csv', index = False)