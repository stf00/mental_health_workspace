
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.offline as ply
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import datetime
import random
import plotly.express as px

import markdown
from plotly.subplots import make_subplots

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

st.set_page_config(layout="wide")

st.write ("""<style> body {
    background-image: url("https://onlyvectorbackgrounds.com/wp-content/uploads/2019/06/Light-Flat-Shapes-Simple-Background-Silver.jpg");
  background-repeat: repeat;
} </style>""", unsafe_allow_html=True)

st.write(""" <style>body {
  margin: 40px;
}
.box {
  background-color: #444;
  color: #fff;
  border-radius: 5px;
  padding: 20px;
  font-size: 150%;
}
.box:nth-child(even) {
  background-color: #ccc;
  color: #000;
}
.wrapper {
  width: 600px;
  display: grid;
  grid-gap: 10px;
  grid-template-columns: repeat(6, 100px);
  grid-template-rows: 100px 100px 100px;
  grid-auto-flow: column;
}</style>""",unsafe_allow_html=True)

from PIL import Image
img_file_buffer = st.file_uploader("health.jpeg")
if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
    img_array = np.array(image) # if you want to pass it to OpenCV
    st.image(image, use_column_width=True)


st.title("Exploratory Data Analysis of Mental Health at The Workplace")

st.text(" \n")
st.text(" \n")
st.text(" \n")


st.markdown("""
All of us have the right to decent and productive work in conditions of freedom, equity, security and human dignity. For persons with mental health problems, achieving this right is particularly challenging. The burden of mental health disorders on health and productivity has long been underestimated.
There is a critical need to identify the instruments and variables that allow us to see not how people die but rather how they live, we now know that the problems of mental illness loom large around the world. Because of the extent and pervasiveness of mental health problems, the World Health Organization (WHO) recognizes mental health as a top priority.
Although it is difficult to quantify the impact of work alone on personal identity, self-esteem and social recognition, most mental health professionals agree that the workplace environment can have a significant impact on an individual’s mental well-being. The impact of mental health problems in the workplace has serious consequences not only for the individual but also for the productivity of the enterprise. Employee performance, rates of illness, absenteeism, accidents and staff turnover are all affected by employees’ mental health status.px
"""
)

st.markdown(
    """
We will be analyzing a dataset from a 2014 survey that measures attitudes towards mental health and frequency of mental health disorders in the tech workplace, identifying variables that effect the employees’ mental health that can also eventually affect their decision to get the right treatment needed.
You can download it [**from here**](https://www.kaggle.com/osmi/mental-health-in-tech-survey).
""")

st.text(" \n")


##############################################################################
#DATA PREPOCESSING

#reading in CSV's from a file path
data = pd.read_csv('mentalhealth.csv')

#dealing with missing data
#Let’s get rid of the variables "Timestamp",“comments”, “state” just to make our lives easier.
data = data.drop(['comments'], axis= 1)
data = data.drop(['state'], axis= 1)
data = data.drop(['Timestamp'], axis= 1)

data.isnull().sum().max() #just checking that there's no missing data missing...

#complete missing age with mean
data['Age'].fillna(data['Age'].median(), inplace = True)

# Fill with media() values < 18 and > 120
s = pd.Series(data['Age'])
s[s<18] = data['Age'].median()
data['Age'] = s
s = pd.Series(data['Age'])
s[s>120] = data['Age'].median()
data['Age'] = s

#Ranges of Age
data['age_range'] = pd.cut(data['Age'], [0,20,30,65,100], labels=["0-20", "21-30", "31-65", "66-100"], include_lowest=True)


# Assign default values for each data type
defaultInt = 0
defaultString = 'NaN'
defaultFloat = 0.0

# Create lists by data tpe
intFeatures = ['Age']
stringFeatures = ['Gender', 'Country', 'self_employed', 'family_history', 'treatment', 'work_interfere',
                 'no_employees', 'remote_work', 'tech_company', 'anonymity', 'leave', 'mental_health_consequence',
                 'phys_health_consequence', 'coworkers', 'supervisor', 'mental_health_interview', 'phys_health_interview',
                 'mental_vs_physical', 'obs_consequence', 'benefits', 'care_options', 'wellness_program',
                 'seek_help']
floatFeatures = []

# Clean the NaN's
for feature in data:
    if feature in intFeatures:
        data[feature] = data[feature].fillna(defaultInt)
    elif feature in stringFeatures:
        data[feature] = data[feature].fillna(defaultString)
    elif feature in floatFeatures:
        data[feature] = data[feature].fillna(defaultFloat)
    else:
        print('Error: Feature %s not recognized.' % feature)
#clean 'Gender'
#Slower case all columm's elements
gender = data['Gender'].str.lower()

#Select unique elements
gender = data['Gender'].unique()

#Made gender groups
male_str = ["male", "m", "male-ish", "maile", "mal", "male (cis)", "make", "male ", "man","msle", "mail", "malr","cis man", "Cis Male", "cis male"]
trans_str = ["trans-female", "something kinda male?", "queer/she/they", "non-binary","nah", "all", "enby", "fluid", "genderqueer", "androgyne", "agender", "male leaning androgynous", "guy (-ish) ^_^", "trans woman", "neuter", "female (trans)", "queer", "ostensibly male, unsure what that really means"]
female_str = ["cis female", "f", "female", "woman",  "femake", "female ","cis-female/femme", "female (cis)", "femail"]

for (row, col) in data.iterrows():

    if str.lower(col.Gender) in male_str:
        data['Gender'].replace(to_replace=col.Gender, value='male', inplace=True)

    if str.lower(col.Gender) in female_str:
        data['Gender'].replace(to_replace=col.Gender, value='female', inplace=True)

    if str.lower(col.Gender) in trans_str:
        data['Gender'].replace(to_replace=col.Gender, value='trans', inplace=True)

#Get rid of bullshit
stk_list = ['A little about you', 'p']
data = data[~data['Gender'].isin(stk_list)]

#There are only 0.20% of self work_interfere so let's change NaN to "Don't know
#Replace "NaN" string from defaultString

data['work_interfere'] = data['work_interfere'].replace([defaultString], 'Don\'t know' )

#missing data
total = data.isnull().sum().sort_values(ascending=False)
percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])




#DATA################################################################
data=pd.DataFrame(data)
#####################################################################

c1,c2=st.beta_columns(2)
#c1, c2, c3, = st.beta_columns((2, 3, 1))
def AgeGender():
    fig = px.histogram(data, x="Age", color="Gender",title="Age Distribution by Gender", marginal="box",
    hover_data=data.columns)
    c1.plotly_chart(fig)

def GenderTreatment():
    fig = px.bar(data, x="treatment", y="Gender", color='Gender',title="Treatment Status According to Gender Distribution")
    c2.plotly_chart(fig)

####
# List all of the shades/tints codes; in this example I am using the tints codes
my_color_scale = [[0.0, '#72DBC5'], [0.2, '#00CC96'], [0.4,'#EA9F97'], [0.6,'#EA9F96'], [0.8,'#A3ACF7'],[1,'#636EFA']]
def MapTreatmentYes():
    df_treatment_yes=data.loc[(data["treatment"]=="Yes")]
    new=df_treatment_yes.groupby(['Country']).size().to_frame('count').reset_index()
    fig=px.choropleth(new,locations="Country",locationmode="country names",color="count",color_continuous_scale=my_color_scale)
    fig.update_layout(title_text="Undergoing Treatment")
    c1.plotly_chart(fig)

def MapTreatmentNo():
    df_treatment_no=data.loc[(data["treatment"]=="No")]
    new=df_treatment_no.groupby(['Country']).size().to_frame('count').reset_index()
    fig=px.choropleth(new,locations="Country",locationmode="country names",color="count",color_continuous_scale=my_color_scale)
    fig.update_layout(title_text="Not Undergoing Treatment",)
    c2.plotly_chart(fig)


#####
def FamilyHistory():
    fig = go.Figure(data=[go.Pie(labels=data["family_history"].value_counts().index, values=data["family_history"].value_counts().values, hole=.3)])
    fig.update_layout(title=go.layout.Title(text='Family History of Mental Illness',x = 0, font=dict(size=18,color='black')))
    c1.plotly_chart(fig)

def Familytreat():
    fig5 = px.bar(data, x="family_history",color='treatment',barmode='group')
    fig5.update_layout(title=go.layout.Title(text='Treatment Status Based of Family History',x = 0, font=dict(size=18,color='black')))
    c2.plotly_chart(fig5)

def Workinterfere():
    fig1 = go.Figure(data=[go.Pie(labels=data["work_interfere"].value_counts().index, values=data["work_interfere"].value_counts().values, hole=.3)])
    fig1.update_layout(title=go.layout.Title(text='Mental Health Interfance With Work Performance',x = 0, font=dict(size=18,color='black')))
    c1.plotly_chart(fig1)

def Employer():
    fig2 = go.Figure(data=[go.Pie(labels=data["benefits"].value_counts().index, values=data["benefits"].value_counts().values, hole=.3)])
    fig2.update_layout(title=go.layout.Title(text='Awareness of Existing Mental Health Benifits Provided By The Employer',x = 0, font=dict(size=18,color='black')))
    c1.plotly_chart(fig2)

def Anonymity():
    fig3 = go.Figure(data=[go.Pie(labels=data["anonymity"].value_counts().index, values=data["anonymity"].value_counts().values, hole=.3)])
    fig3.update_layout(title=go.layout.Title(text='Impotance of Protecting Anonymity if Mental Health Resources Are Used',x = 0, font=dict(size=18,color='black')))
    c2.plotly_chart(fig3)










###LABELS###################################################################

#Encoding data

data_ = data.iloc[:,1:]
encoded_data = data_.apply(LabelEncoder().fit_transform)
fin_encoded_data = pd.concat([data['Age'],encoded_data], axis = 1) # pd.concat([df1,df2],axis =1)
fin_encoded_data.head()
country = data.groupby(data['Country'])

def Country_mean_age():
    ca=country['Age'].median().sort_values()
    st.write(ca)

def Worktech():
    treat = data_.groupby(data['treatment'])
    wtt=treat['tech_company', 'work_interfere'].describe()
    st.write(wtt)

def techcompany():
    treat = data.groupby(data['treatment'])
    techcomp=treat['tech_company', 'work_interfere'].describe()
    st.write(techcomp)

#Corrolation##########################################################
#Encoding data
from sklearn import preprocessing

data1=data.copy()
labelDict = {}
for feature in data1:
    le = preprocessing.LabelEncoder()
    le.fit(data1[feature])
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    data1[feature] = le.transform(data1[feature])
    # Get labels
    labelKey = 'label_' + feature
    labelValue = [*le_name_mapping]
    labelDict[labelKey] =labelValue

for key, value in labelDict.items():
    print(key, value)



#correlation matrix

corrmat = data1.corr()

#treatment correlation matrix
def corr_treat():
    k = 10 #number of variables for heatmap
    cols = corrmat.nlargest(k,'treatment')['treatment'].index
    cm = np.corrcoef(data1[cols].values.T)
    sns.set(font_scale=1.25)
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, square=True,cmap="Blues", linewidth=.2,annot=True,annot_kws={'size': 6});
    c1.pyplot(f)

def varcorr1():
    fig6 = px.parallel_categories(data1[['treatment', 'obs_consequence', 'family_history', 'work_interfere']])
    c1.plotly_chart(fig6)

def varcorr2():
    fig7 = px.parallel_categories(data1[['treatment', 'care_options', 'anonymity', 'Gender', 'benefits', 'mental_health_consequence']])
    c2.plotly_chart(fig7)

##Predictions#########################################################

df=data1.copy()

def variablescorr():
    treat_corr = df.corr()["treatment"]
    c2.write(treat_corr)

st.set_option('deprecation.showPyplotGlobalUse', False)
ML_data=pd.DataFrame(df)

x=df[['Gender', 'family_history',
       'work_interfere', 'no_employees',
       'benefits', 'care_options', 'wellness_program',
       'anonymity', 'mental_health_interview','obs_consequence', 'age_range']]
y=df['treatment']
sc=StandardScaler()
x_scaled=sc.fit_transform(x)
x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,test_size=0.3,random_state=90)

model_log=LogisticRegression()
model_log.fit(x_train,y_train)
y_pred_log=model_log.predict(x_test)

def Logistic_Reg_model():
    model_log=LogisticRegression()
    model_log.fit(x_train,y_train)
    y_pred_log=model_log.predict(x_test)
    from sklearn.metrics import confusion_matrix
    confusion_matrix_log =confusion_matrix(y_test,y_pred_log)

    sns.heatmap(confusion_matrix_log/np.sum(confusion_matrix_log), annot=True,
            fmt='.2%', cmap="Blues")
    ax= plt.subplot()
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
    ax.set_title('Confusion Matrix123');
    ax.xaxis.set_ticklabels(['False', 'True']); ax.yaxis.set_ticklabels(['True', 'False'])
    c1.pyplot()



#model 2

x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,test_size=0.3,random_state=90)
model_Rand=RandomForestClassifier()
model_Rand.fit(x_train,y_train)
y_pred_log=model_log.predict(x_test)


def RandomForest():
    x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,test_size=0.3,random_state=90)
    model_Rand=RandomForestClassifier()
    model_Rand.fit(x_train,y_train)
    y_pred_log=model_log.predict(x_test)
    from sklearn.metrics import confusion_matrix
    confusion_matrix_Rand =confusion_matrix(y_test,y_pred_log)
    sns.heatmap(confusion_matrix_Rand/np.sum(confusion_matrix_Rand), annot=True,
        fmt='.2%', cmap='Blues')
    ax= plt.subplot()
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
    ax.set_title('Confusion Matrix2');
    ax.xaxis.set_ticklabels(['False', 'True']); ax.yaxis.set_ticklabels(['True', 'False'])
    c1.pyplot()

#Model 3
def plotKNN():
    k_range=range(1,26)
    scores={}
    scores_list=[]
    for k in k_range:
        knn=KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_train,y_train)
        y_pred=knn.predict(x_test)
        scores[k]=metrics.accuracy_score(y_test,y_pred)
        scores_list.append(metrics.accuracy_score(y_test,y_pred))
    plt.plot(k_range,scores_list)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Testing Accuracy')
    c2.pyplot()


######################
#Model 4
from sklearn import svm
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
model_SVM= svm.SVC(kernel='linear') # Linear Kernel
model_SVM.fit(x_train, y_train)

def SVM():
    y_pred = model_SVM.predict(x_test)
    confusion_matrix_tree =confusion_matrix(y_test,y_pred)
    sns.heatmap(confusion_matrix_tree/np.sum(confusion_matrix_tree), annot=True,
            fmt='.2%', cmap='Blues')
    ax= plt.subplot()
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
    ax.set_title('Confusion MatrixSVM');
    ax.xaxis.set_ticklabels(['False', 'True']); ax.yaxis.set_ticklabels(['True', 'False'])
    c1.pyplot()


model_KNN=KNeighborsClassifier(n_neighbors=13)
model_KNN.fit(x_train,y_train)
def KNN_model():
    x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,test_size=0.3,random_state=90)
    y_pred_KNN=model_KNN.predict(x_test)
    confusion_matrix_KNN =confusion_matrix(y_test,y_pred_KNN)
    sns.heatmap(confusion_matrix_KNN/np.sum(confusion_matrix_KNN), annot=True,
            fmt='.2%', cmap='Blues')
    ax= plt.subplot()
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
    ax.set_title('Confusion Matrix');
    ax.xaxis.set_ticklabels(['False', 'True']); ax.yaxis.set_ticklabels(['True', 'False'])
    c1.pyplot()



#SIDEBAR##############################################################



# Security
#passlib,hashlib,bcrypt,scrypt
st.sidebar.text(" \n \n ")
st.sidebar.title("Login")

def main():
		password = st.sidebar.text_input("Password",type='password')

main()

if st.button("Let's Go!"):
    st.write('Explore the sidebar on your left!')

#
st.sidebar.text(" \n \n ")
st.sidebar.title("Exploratory Data")
st.sidebar.text(" \n \n ")

st.sidebar.subheader("View Data")
if st.sidebar.checkbox('Raw Data'):
    if st.checkbox('First 10 Columns of raw data'):
        st.text(data.head(5))
    if st.checkbox('Variables available for analysis'):
        st.markdown("""<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="Exploring-mental-health-in-the-tech-industry-in-2014"><center>Exploring mental health in the tech industry in 2014</center><a class="anchor-link" href="https://www.kaggle.com/ptfrwrd/mental-health-eda-important-features#Exploring-mental-health-in-the-tech-industry-in-2014" target="_self">¶</a></h1><p>Some information about explored data:</p>
<ul>
<li>"<em>This dataset is from a 2014 survey that measures attitudes towards mental health and frequency of mental health disorders in the tech workplace</em>". </li>
<li><strong>Features:</strong></li>
</ul>
<table>
<thead>
<tr><th>Feature name</th><th>Description</th></tr>
</thead>
<tbody>
<tr><td>Timestamp</td><td> - </td></tr>
<tr><td>Age</td><td> - </td></tr>
<tr><td>Gender</td><td> - </td></tr>
<tr><td>Country</td><td> - </td></tr>
<tr><td>State</td><td> (only for US) </td></tr>
<tr><td>Self employed</td><td> Are you self-employed? </td></tr>
<tr><td>Family history</td><td> Family history of mental illness </td></tr>
<tr><td>Treatment</td><td>Is treatment for a mental health condition was?</td></tr>
<tr><td>Work interfere</td><td> Is mental health condition affects work? </td></tr>
<tr><td>No employees</td><td> The number of employees in your company or organization </td></tr>
<tr><td>Remote work</td><td> Having remote work (outside of an office) at least 50% of the time </td></tr>
<tr><td>Tech company</td><td> The employer is primarily a tech company/organization </td></tr>
<tr><td>Benefits</td><td> Providing mental health benefits by the employer </td></tr>
<tr><td>Care options:</td><td> Providing options for mental health care by the employer </td></tr>
<tr><td>Wellness program</td><td> Discussion about mental health as part of an employee wellness program by the employes </td></tr>
<tr><td>Seek help</td><td> Providing resources by the employer to learn more about mental health issues and how to seek help </td></tr>
<tr><td>Anonymity</td><td> Protecting anonymity if you choose to take advantage of mental health or substance abuse treatment resources</td></tr>
<tr><td>Leave</td><td> How easy is it for you to take medical leave for a mental health condition? </td></tr>
<tr><td>Mental-health consequence: </td><td>  Having negative consequences caused by discussing a mental health issue with your employer</td></tr>
<tr><td>Phys-health consequence</td><td> Having negative consequences caused by discussing a physical health issue with your employer </td></tr>
<tr><td>Coworkers</td><td> Would you be willing to discuss a mental health issue with your coworkers?</td></tr>
<tr><td>Supervisor</td><td> Would you be willing to discuss a mental health issue with your direct supervisor(s)? </td></tr>
<tr><td>Mental health interview:</td><td> Would you bring up a mental health issue with a potential employer in an interview? </td></tr>
<tr><td>Phys health interview</td><td> Would you bring up a physical health issue with a potential employer in an interview? </td></tr>
<tr><td>Mental vs Physical</td><td> Do you feel that your employer takes mental health as seriously as physical health? </td></tr>
<tr><td>Obs consequence</td><td> Have you heard of or observed negative consequences for coworkers with mental health conditions in your workplace? </td></tr>
<tr><td>Comments</td><td> Any additional notes or comments </td></tr>
</tbody>
</table>
</div>
        """
        ,unsafe_allow_html=True)

#######SIDEBAR BAR BAR ###


st.sidebar.subheader("Choose Question to Explore")
location_filtered=[]
option = st.sidebar.selectbox(' ', (
'None',
'Country distribution of people that are taking treatment',
'Gender distribution of people undergoing treatment',
'Treatment in relation to family history',
'Attributes  that affect treatment decision',
'Mental health interferance with work performance',
'Existance of mental health benifits at the workspace'

))


if option=="Country distribution of people that are taking treatment":
    if st.checkbox('Distribution of people that are undergoing treatment across countries'):
        MapTreatmentYes()
    if st.checkbox('Distribution of people that are not undergoing treatment across countries'):
        MapTreatmentNo()
    if st.checkbox('Mean age per country'):
        Country_mean_age()

if option=="Gender distribution of people undergoing treatment":
    if st.checkbox('Age distribution by gender'):
        AgeGender()
    if st.checkbox('Distribution of people undergoing treatment by gender'):
        GenderTreatment()

if option=="Treatment in relation to family history":
    if st.checkbox('Status of family history of mental illness'):
        FamilyHistory()
    if st.checkbox('Treatment status based of family history of mental illness'):
        Familytreat()

if option=="Attributes affect treatment decision":
    if st.checkbox('Relation between skeeping treatment and 3 other relevent attributes'):
        varcorr1()
    if st.checkbox('Relation between skeeping treatment and 5 other relevent attributes'):
        varcorr2()
    if st.checkbox('Correlation matrix'):
        corr_treat()
    if st.checkbox('Treatment correlation table with attributes'):
        variablescorr()

if option=="Mental health interferance with work performance":
    Workinterfere()
    Worktech()

if option=="Existance of mental health benifits at the workspace":
    Employer()
    Anonymity()




#PREDICTION SIDEBAR ###
st.sidebar.subheader("Predicting decisions to get the right mental health treatment needed ")
option = st.sidebar.selectbox(' ', ("None","Logistic Regression","K-Nearest Neighbors Algorithm","Random Forest Classifier","Support Vector Machine"))


if option=='Logistic Regression':
    st.sidebar.markdown('You selected: _Logistic Regression_')
    st.markdown("""<h2> <b> Logistic Regression """,unsafe_allow_html=True)
    st.write('Accuracy of logistic regression classifier on test set:{:.2f}'.format(model_log.score(x_test,y_test)))
    Logistic_Reg_model()

if option=='K-Nearest Neighbors Algorithm':
    st.sidebar.markdown('You selected: _K-Nearest Neighbors Algorithm_')
    st.markdown("""<h1>K-Nearest Neighbors Algorithm """,unsafe_allow_html=True)
    st.write('Accuracy of KNN on test set:{:.2f}'.format(model_KNN.score(x_test,y_test)))
    KNN_model()
    plotKNN()


if option=='Random Forest Classifier':
    st.sidebar.markdown('You selected: _Random Forest Classifier_')
    st.markdown("""<h2> <b> Random Forest Classifier """,unsafe_allow_html=True)
    st.write('Accuracy of Random Forest classifier on test set:{:.2f}'.format(model_Rand.score(x_test,y_test)))
    RandomForest()

if option=='Support Vector Machine':
    st.sidebar.markdown('You selected: _Support Vector Machine_')
    st.markdown("""<h2> <b> Support Vector Machine """,unsafe_allow_html=True)
    st.write('Accuracy of SVM on test set:{:.2f}'.format(model_SVM.score(x_test,y_test)))
    SVM()

##Password##############################


################
