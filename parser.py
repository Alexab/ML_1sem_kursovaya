import pandas as pd

### загрузим файлы данных, из которых будем собирать финальный набор, с которым, в свою очередь, уже будем работать

df_walking_1 = pd.read_csv('./dataset/Walking_1.csv')
df_walking_2 = pd.read_csv('./dataset/Walking_2.csv')
df_walking_3 = pd.read_csv('./dataset/Walking_3.csv')

df_upstairs_1 = pd.read_csv('./dataset/Upstairs_1.csv')

df_downstairs_1 = pd.read_csv('./dataset/Downstairs_1.csv')

df_sitting_1 = pd.read_csv('./dataset/Sitting_1.csv')

df_nothing_1 = pd.read_csv('./dataset/Nothing_1.csv')

### объединим разные файлы, относящиеся к одному виду - в один файл

df_walking = pd.concat([df_walking_1, df_walking_2, df_walking_3],
                       ignore_index=True)
df_upstairs = df_upstairs_1
df_downstairs = df_downstairs_1
df_sitting = df_sitting_1
df_nothing = df_nothing_1

### перед общей конкатенацией данных необходимо проименовать виды действий

walking = pd.DataFrame(columns=["ACTIVITY"])
for i in range (len(df_walking.index)):
    walking = walking.append({'ACTIVITY': 'WALKING'}, ignore_index=True)
df_walking = pd.concat([walking, df_walking], axis=1)

upstairs = pd.DataFrame(columns=["ACTIVITY"])
for i in range (len(df_upstairs.index)):
    upstairs = upstairs.append({'ACTIVITY': 'UPSTAIRS'}, ignore_index=True)
df_upstairs = pd.concat([upstairs, df_upstairs], axis=1)

downstairs = pd.DataFrame(columns=["ACTIVITY"])
for i in range (len(df_downstairs.index)):
    downstairs = downstairs.append({'ACTIVITY': 'DOWNSTAIRS'}, ignore_index=True)
df_downstairs = pd.concat([downstairs, df_downstairs], axis=1)

sitting = pd.DataFrame(columns=["ACTIVITY"])
for i in range (len(df_sitting.index)):
    sitting = sitting.append({'ACTIVITY': 'SITTING'}, ignore_index=True)
df_sitting = pd.concat([sitting, df_sitting], axis=1)

nothing = pd.DataFrame(columns=["ACTIVITY"])
for i in range (len(df_nothing.index)):
    nothing = nothing.append({'ACTIVITY': 'NOTHING'}, ignore_index=True)
df_nothing = pd.concat([nothing, df_nothing], axis=1)

### объединим файлы в общий набор данных и сохраним

data = pd.concat([df_walking,
                  df_upstairs,
                  df_downstairs,
                  df_sitting,
                  df_nothing], ignore_index=True)

data.to_csv("./dataset.csv", index=False)