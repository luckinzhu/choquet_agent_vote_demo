import pandas as pd

# data = pd.read_csv('test.csv')
# column1 = 'content'
# column2 = 'label'
# data[column1],data[column2] = data[column2],data[column1]
# data.to_csv('test1.csv', header=0, quoting=1, index=False)

data = pd.read_csv('train.csv')
data['content'] = data['content'].str.replace('\n',';')

data.to_csv('train3.csv', header=False, quoting=1, index=False)