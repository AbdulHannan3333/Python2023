import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier

st.write('''
# Random Forest Classifier
## Made by Abdul hannan
This app predicts the type of iris based on sepal length, sepal width, petal length and petal width
''')

st.sidebar.header("Change Iris Parameters")

# aik function define krna h 

def change_iris_parameters():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    # Dictionay define
    data = {'sepal_length' : sepal_length,
            'sepal_width' : sepal_width,
            'petal_length' : petal_length,
            'petal_width': petal_width}
    # ab is data ko dataframe may convert kr dena with 0 index k sth
    parameters = pd.DataFrame(data, index=[0])
    return parameters

df = change_iris_parameters()

st.subheader("Iris Parameters")
st.write(df)

st.header("Iris dataset donwload from seaborn library")
iris = sns.load_dataset('iris')
st.write(iris.head(10))

#ploting
st.subheader("Plotly k plots") 
fig = px.scatter(iris, x='sepal_length',
y='petal_length', color = 'species' )
st.plotly_chart(fig)

fig = px.bar(iris, x='species',
y='petal_length', color = 'species' )
st.plotly_chart(fig)

st.subheader("GDP Plot, Animated")
df1 = px.data.gapminder()
fig2 = px.scatter(df1, x='gdpPercap', y='lifeExp', animation_frame='year', animation_group='country', size='pop',
                  color='continent', hover_name='country', log_x=True, size_max=55, range_x=[100,100000], range_y=[25, 90])
st.plotly_chart(fig2)

fig3 = px.bar(df1, x='continent', y='pop', color='continent', animation_frame='year', animation_group='country',
              range_y=[0, 400000000])
st.plotly_chart(fig3)

fig4 = px.scatter_3d(iris, x='sepal_length', y='sepal_width', z='petal_width', color='species')
st.plotly_chart(fig4)

#spilt into training and testing 
X = iris[['sepal_length', 'sepal_width', 'petal_length','petal_width']]
y = iris['species']

# train the model
model = RandomForestClassifier()
model.fit(X, y)

# prediction
prediction = model.predict(df)

prediction_proba = model.predict(df)
st.subheader("Class labels and their crossponding index number")
st.write(iris['species'].unique())

st.subheader("Prediction")
p = st.write(prediction[0])
st.write(p)

st.subheader("Prediction Probality")
st.write(prediction_proba)