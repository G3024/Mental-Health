import pandas as pd
data = pd.read_csv("mental health.csv")


# get rid of columns that we dont need
data.drop('Timestamp', axis='columns', inplace=True)


# take age columns
age = data.Age
age = pd.DataFrame({'Age':age})
data.drop('Age', inplace=True, axis='columns')



# fit_transform current study
curr_stdy = []
for i in data['Your current year of Study']:
    if i == 'Year 1':
        curr_stdy.append(1)
    elif i == 'Year 2':
        curr_stdy.append(2)
    elif i == 'Year 3':
        curr_stdy.append(3)
    elif i == 'Year 4':
        curr_stdy.append(4)
    else:
        curr_stdy.append(2)
curr_stdy = pd.DataFrame({'curr study':curr_stdy})


# label_encoder
from sklearn.preprocessing import LabelEncoder
for i in data:
    data[i] = LabelEncoder().fit_transform(data[i])
data.drop('Your current year of Study', axis='columns', inplace=True)
fix_conc = pd.concat([data, curr_stdy, age], axis='columns')

target = fix_conc['Did you seek any specialist for a treatment?']
fix_conc.drop('Did you seek any specialist for a treatment?', axis='columns', inplace=True)

# training data

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(fix_conc, target, train_size=0.2)


from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=10000)
model.fit(xtrain, ytrain)
model.score(xtest, ytest)# 0.9390243902439024


xtest, '\n''\n', model.predict(xtest)