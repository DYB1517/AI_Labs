import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

df = pd.read_csv('spaceship_titanic.csv')
df.head()
df.info()

print(df)
cols = df.columns
mtx = df.isnull()
print(mtx.sum())

vect_mod = ['HomePlanet', 'Cabin', 'Destination', 'Name']
vect_med = ['Age',  'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
vect_mean = ['VIP','CryoSleep']

for col in vect_mod:
    df[col] = df[col].fillna(df[col].mode()[0])
for col in vect_med:
    df[col] = df[col].fillna(df[col].median())
for col in vect_mean:
    df[col] = df[col].fillna(df[col].mean())

ed_mtx = df.isnull()
print(ed_mtx.sum())
scaler = MinMaxScaler()

for col in vect_med:
    df[col] = scaler.fit_transform(df[[col]])
df = pd.get_dummies(df, columns=['Destination', 'HomePlanet'], drop_first=True)

df.to_csv('spaceship_titanic_changed.csv', index = False)