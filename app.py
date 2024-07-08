
import pandas as pd
import streamlit as st
import plotly.express as px
from PIL import Image
import os
from sklearn.impute import SimpleImputer
import numpy as np
import imblearn
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report, consensus_score,  roc_curve, auc

icon = Image.open('icon.png')
logo = Image.open('logo.png')
banner = Image.open('banner.jpg') 

st.set_page_config(layout = 'wide' ,
                   page_title='Python' ,
                   page_icon = icon)
st.title('Application of Machine Learning')
st.text('Simple Machine Learning Web Application with Streamlit')

# Sidebar Container
st.sidebar.image(image = logo)
menu = st.sidebar.selectbox('', ['Homepage' , 'EDA' , 'Modeling'])

if menu == 'Homepage':

    # Homepage Container
    st.header('Homepage')
    st.image(banner,use_column_width = 'always')

    dataset = st.selectbox('Select dataset' , ['Loan prediction' , 'Water Potability'])
    st.markdown('Selected:  **{0}** Dataset'.format(dataset))

    if dataset == 'Loan prediction':
        st.warning("""
                   **The problem**:
        Dream Housing Finance company deals in all home loans. They have presence across all urban, semi urban and rural areas. Customer first apply for home loan after that company validates the customer eligibility for loan. 
        The company wants to automate the loan eligibility process(real time) based on customer detail provided while filling online application form. These details are Gender, Martial Status, Education, Number of Dependents, Income, Loan Amount, Credit History and others. 
        To automate this process, they have given a problem to identify the customer segments , those are eligible for loan amount so that they can specifically target these customers. It's a classification problem , given information  about the application we have to predict 
        wether the they will be to pay the loan or not. We will start by exploratory data analysis, then preprocessing, and finally we will be testing different models such as decision trees. The data consists of the following rows:
                   """)
        
        st.info("""
                **Loan_ID** : Unique loan ID.

                **Gender** : Male/ Female.

                **Married** : Applicant married (Y/N).

                **Dependents** : Number of dependents.

                **Education** : Applicant Education (Graduate/ Under Graduate).

                **Self_Employed** : Self employed (Y/N).

                **ApplicantIncome** : Applicant Income.

                **Loan_Amount** : Loan amount in thousands of dollars.

                **Loan_Amount_Term** : Term of loan in months.

                **Credit_History** : Credit history meets guideliness(Y/N).

                **Property_Area** : Urban/ Semi Urban/ Rural.
                
                **Loan_Status** : Loan approved (Y/N) this is the target variable.
                """)
    
    else:
        st.warning('''
                   **The problem**:
                    Access to safe drinking water is essential to health, a basic human right, and a component of effective policy for health protection. 
                   Ensuring that everyone has access to safe drinking water is critical for preventing disease and improving quality of life. 
                   The goal is to predict the potability of water based on various physicochemical attributes.
                   
                   ''')
        st.info('''
                **ph**: The pH level of the water, which measures the acidity or alkalinity. It ranges from 0 to 14, with 7 being neutral. Lower values indicate acidity, and higher values indicate alkalinity.
                
                **Hardness**: This measures the concentration of calcium and magnesium ions in the water. It is usually expressed in mg/L (milligrams per liter). Hard water has high levels of these ions, while soft water has lower levels.

                **Solids**: The total dissolved solids (TDS) in the water, measured in ppm (parts per million). This includes all minerals, salts, and organic matter dissolved in the water.

                **Chloramines**: The concentration of chloramines (chlorine compounds) in the water, measured in ppm. Chloramines are used for disinfection and maintaining water quality.

                **Sulfate**: The concentration of sulfate ions in the water, measured in mg/L. High sulfate levels can cause a laxative effect and affect the taste of the water.

                **Conductivity**: This measures the water's ability to conduct an electric current, which is related to the concentration of ions in the water. It is measured in μS/cm (microsiemens per centimeter).

                **Organic_carbon**: The amount of organic carbon in the water, measured in ppm. This is an indicator of the presence of organic pollutants.

                **Trihalomethanes**: The concentration of trihalomethanes (THMs) in the water, measured in ppm. THMs are chemical compounds that can form when chlorine reacts with organic matter in water.

                **Turbidity**: The measure of water clarity, indicating the presence of suspended particles. It is measured in NTU (nephelometric turbidity units). Higher turbidity indicates murkier water.

                **Potability**: A binary indicator of whether the water is safe to drink (potable) or not. It typically has two values: 1 for potable (safe to drink) and 0 for non-potable (not safe to drink).
               
                ''')
        
elif menu == 'EDA':
    # Outlier treatment
    def outlier_treatment (datacolumn):
        sorted(datacolumn)
        Q1,Q3 = np.percentile(datacolumn, [25,75])
        IQR = Q3-Q1
        lower_range = Q1 - (1.5 * IQR)
        upper_range = Q3 + (1.5 * IQR)
        return lower_range , upper_range
    
    # Datasetin tesviri
    def describeStat(df):
        st.dataframe (df)
        st.subheader ('Statistical Values')
        df.describe().T
        
        # Targetin balans veziyyeti
        st.subheader('Balance of Data')
        st.bar_chart(df.iloc[:,-1].value_counts())

        # null value 
        null_df = df.isnull().sum().to_frame().reset_index()
        null_df.columns = ['Columns', 'Counts']

        # beta kolumlar yaratmaq
        c_eda1, c_eda2, c_eda3 = st.columns([2.5, 1.5, 2.5])
        c_eda1.subheader('Null Variables')
        c_eda1.dataframe(null_df)

        c_eda2.subheader('Imputation')
        cat_method = c_eda2.radio('Categorical', ['Mode', 'Backfill', 'Ffil'])
        num_method = c_eda2.radio('Numercical', ['Mode', 'Median'])

        # Feature Engineering
        c_eda2.subheader('Feature Engineering')
        balance_problem = c_eda2.checkbox('Under Sampling')
        outlier_problem = c_eda2.checkbox('Clean Outlier')

        if c_eda2.button('Data preprocessing'):
            
            #Data cleaning
            cat_array = df.iloc[:,:-1].select_dtypes(include = 'object').columns
            num_array = df.iloc[:,:-1].select_dtypes(exclude = 'object').columns

            if cat_array.size > 0:

                if cat_method == 'Mode':
                    imp_cat = SimpleImputer(missing_values= np.nan, strategy = 'most_frequent')
                    df[cat_array] = imp_cat.fit_transform(df[cat_array])

                elif cat_method == 'Backfill':
                    df[cat_array].fillna(method = 'backfill', inplace = True)

                else:
                    df[cat_array].fillna(method = 'ffill', inplace = True)
            
            if num_array.size > 0:
                
                if num_method == 'Mode':
                    imp_num = SimpleImputer(missing_values= np.nan, strategy = 'most_frequent')
                
                else:
                    imp_num = SimpleImputer(missing_values= np.nan, strategy = 'median')

                df[num_array] = imp_num.fit_transform(df[num_array])

            df.dropna(axis = 0, inplace =True) # backfill or ffill secilerse lazim ola biler.

            if balance_problem:

                rus = RandomUnderSampler()
                x = df.iloc[:,:-1]
                y = df.iloc[:,-1]
                x, y = rus.fit_resample(x, y)
                df = pd.concat([x, y],axis = 1)

            if outlier_problem:
                for col in num_array:
                    lowerbound, upperbound = outlier_treatment(df[col])
                    df[col] = np.clip(df[col], a_min= lowerbound, a_max= upperbound)

            
            null_df = df.isnull().sum().to_frame().reset_index()
            null_df.columns = ['Columns', 'Counts']

            c_eda3.subheader('Null Variables')
            c_eda3.dataframe(null_df)
            st.subheader('Balance of Data')
            st.bar_chart(df.iloc[:,-1].value_counts())
            
            # Filter only numerical columns for correlation matrix
            df_numeric = df.select_dtypes(include=[np.number])
            heatmap = px.imshow(df_numeric.corr())
            st.plotly_chart(heatmap)
            st.dataframe(df)

            if os.path.exists('formodel.csv'):
                os.remove('formodel.csv')
            df.to_csv('formodel.csv', index = False)

    # Homepage Container
    st.header ('Exploratory Data Analysis')
    dataset = st.selectbox('Select dataset', ['Loan Prediction','Water Potability'])

    if dataset == 'Loan Prediction':
        df = pd.read_csv('loan_prediction.csv')
        describeStat(df)

    else:
        df = pd.read_csv('water_potability.csv')
        describeStat(df)


else:
    #Modeling Container
    st.header('Modeling')
    if not os.path.exists('formodel.csv'):
        st.header ('Please Run Preprocessing')

    else:
        df = pd.read_csv('formodel.csv')
        st.dataframe(df)

        c_model1, c_model2 = st.columns(2)

        c_model1.subheader('Scaling')
        scaling_method = c_model1.radio('', ['Standart', 'Robust', 'MinMax'])
        c_model2.subheader('Encoders')
        encoder_method = c_model2.radio('', ['Label', 'One-Hot'])

        st.header('Train and Test Splitting')

        c_model1_1, c_model2_1 = st.columns(2)
        random_state = c_model1_1.text_input('Random State')
        test_size = c_model2_1.text_input('Percentage')

        model = st.selectbox('Select Model', ['Xgboost', 'Catboost'])
        st.markdown('Selected:    **{0}**  Model'.format(model))

        if st.button('Run Model'):

            cat_array = df.iloc[:,:-1].select_dtypes(include = 'object').columns
            num_array = df.iloc[:,:-1].select_dtypes(exclude = 'object').columns
            y = df.iloc[:,[-1]]

            if num_array.size > 0:

                if scaling_method == 'Standart':
                    sc = StandardScaler()
                elif scaling_method == 'Robust':
                    sc = RobustScaler()
                else:
                    sc = MinMaxScaler()

                df[num_array] = sc.fit_transform(df[num_array])

            if cat_array.size > 0:

                if encoder_method == 'Label':
                    lb = LabelEncoder()
                    for col in cat_array:
                        df[col] = lb.fit_transform(df[col])

                else:
                    df.drop(df.iloc[:,[-1]], axis = 1 , inplace = True)
                    dms_df = df[cat_array]
                    dms_df = pd.get_dummies(dms_df, drop_first = True)
                    df_ = df.drop(cat_array, axis = 1)
                    df = pd.concat([df_, dms_df, y], axis = 1)
            
            st.dataframe(df)

    # Modeling 
            x = df.iloc[:,:-1]
            y = df.iloc[:,[-1]]

            X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = float(test_size), random_state= int(random_state))

            st.markdown('Size of X_train = {0}'.format(X_train.shape))
            st.markdown('Size of X_test = {0}'.format(X_test.shape))
            st.markdown('Size of y_train = {0}'.format(y_train.shape))
            st.markdown('Size of y_test = {0}'.format(y_test.shape))

            st.header('The model was successfully completed')
            st.subheader('Evaluation')
           

            if model == 'Xgboost':
                model = xgb.XGBClassifier().fit(X_train, y_train)

            else:
                model = CatBoostClassifier().fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_score = model.predict_proba(X_test)[:,1]

            st.markdown('Confusion Matrix')
            st.write(confusion_matrix(y_test, y_pred))

            report = classification_report(y_test, y_pred, output_dict = True)
            df_report = pd.DataFrame(report).transpose()

            st.dataframe(df_report)

            accuracy = str(round(accuracy_score(y_test, y_pred),2))+'%'
            st.markdown('Accuracy Score = '+ accuracy)

        # Roc curve
            fpr, tpr, thresholds = roc_curve(y_test, y_score)
            fig = px.area(
                        x = fpr,
                        y = tpr,
                        title = 'Roc Curve',
                        labels = dict(x ='False Positive Rate',
                                      y = 'True Positive Rate'),
                        width = 700,
                        height = 700
                            )
            fig.add_shape(
                        type = 'line',
                        line = dict(dash = 'dash'),
                        x0 = 0,
                        x1 = 1,
                        y0 = 0,
                        y1 = 1
                            )      
            st.plotly_chart(fig)     
            auc_score = f'AUC Score = {auc(fpr, tpr): .4f}'
            st.markdown(auc_score)
            st.title('Thanks For Using')
            
    


