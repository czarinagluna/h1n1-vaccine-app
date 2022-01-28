from os import statvfs
import streamlit as st
import pickle
import numpy as np
from PIL import Image

import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

def load_model():
    with open('vaccination_clf.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

classifier = data['model']
le_concern = data['le_concern']
le_knowledge = data['le_knowledge']
le_b1 = data['le_b1']
le_b2 = data['le_b2'] 
le_b3 = data['le_b3'] 
le_b4 = data['le_b4'] 
le_b5 = data['le_b5'] 
le_b6 = data['le_b6'] 
le_b7 = data['le_b7']
le_dr_h1n1 = data['le_dr_h1n1'] 
le_dr_seas = data['le_dr_seas'] 
le_chronic = data['le_chronic'] 
le_child = data['le_child'] 
le_hcw = data['le_hcw'] 
le_ins = data['le_ins']
le_age = data['le_age'] 
le_edu = data['le_edu'] 
le_race = data['le_race'] 
le_sex = data['le_sex'] 
le_income = data['le_income'] 
le_marital = data['le_marital'] 
le_rent = data['le_rent'] 
le_employ = data['le_employ']

def show_predict_page():
    img = Image.open('nyclogo.png')

    st.image(img)
    
    st.title('H1N1 Vaccination Survey')
    
    st.write("""### Please answer the survey questions.""")

    ale_concern = ('Not at all concerned', 'Not very concerned', 'Somewhat concerned', 'Very concerned')
    ale_knowledge = ('No knowledge', 'A little knowledge', 'A lot of knowledge')
    ale_b1 = ('No', 'Yes')
    ale_b2 = ('No', 'Yes') 
    ale_b3 = ('No', 'Yes') 
    ale_b4 = ('No', 'Yes') 
    ale_b5 = ('No', 'Yes') 
    ale_b6 = ('No', 'Yes') 
    ale_b7 = ('No', 'Yes')
    ale_dr_h1n1 = ('No', 'Yes') 
    ale_dr_seas = ('No', 'Yes') 
    ale_chronic = ('No', 'Yes') 
    ale_child = ('No', 'Yes') 
    ale_hcw = ('No', 'Yes') 
    ale_ins = ('No', 'Yes', 'Not sure')
    ale_age = ('65+ Years', '55 - 64 Years', '45 - 54 Years', '18 - 34 Years', '35 - 44 Years') 
    ale_edu = ('College Graduate', 'Some College', '12 Years', '< 12 Years ') 
    ale_race = ('White', 'Black', 'Hispanic', 'Other or Multiple') 
    ale_sex = ('Female', 'Male') 
    ale_income = ('<= $75,000, Above Poverty', '> $75,000', 'Below Poverty') 
    ale_marital = ('Married', 'Not Married') 
    ale_rent = ('Own', 'Rent') 
    ale_employ = ('Employed', 'Not in Labor Force', 'Unemployed')

    concern = st.selectbox('Level of concern about the H1N1 flu.', ale_concern)
    knowledge = st.selectbox('Level of knowledge about H1N1 flu.', ale_knowledge)
    b1 = st.selectbox('Has taken antiviral medications.', ale_b1)
    b2 = st.selectbox('Has avoided close contact with others with flu-like symptoms.', ale_b2)
    b3 = st.selectbox('Has bought a face mask.', ale_b3)
    b4 = st.selectbox('Has frequently washed hands or used hand sanitizer.', ale_b4) 
    b5 = st.selectbox('Has reduced time at large gatherings.', ale_b5) 
    b6 = st.selectbox('Has reduced contact with people outside of own household.', ale_b6)
    b7 = st.selectbox('Has avoided touching eyes, nose, or mouth.', ale_b7)
    dr_h1n1 = st.selectbox('H1N1 flu vaccine was recommended by doctor.', ale_dr_h1n1) 
    dr_seas = st.selectbox('Seasonal flu vaccine was recommended by doctor.', ale_dr_seas) 
    chronic = st.selectbox('Has any of the following chronic medical conditions: asthma or an other lung condition, diabetes, a heart condition, a kidney condition, sickle cell anemia or other anemia, a neurological or neuromuscular condition, a liver condition, or a weakened immune system caused by a chronic illness or by medicines taken for a chronic illness.', ale_chronic ) 
    child = st.selectbox('Has regular close contact with a child under the age of six months.', ale_child) 
    hcw = st.selectbox('Is a healthcare worker.', ale_hcw) 
    ins = st.selectbox('Has health insurance.', ale_ins)

    h1n1_eff = st.slider("Opinion about H1N1 vaccine effectiveness.", 1, 5, 3)
    h1n1_risk = st.slider("Opinion about risk of getting sick with H1N1 flu without vaccine.", 1, 5, 3)
    h1n1_sick = st.slider("Worry of getting sick from taking H1N1 vaccine.", 1, 5, 3)
    seas_eff = st.slider("Opinion about seasonal flu vaccine effectiveness.", 1, 5, 3)
    seas_risk = st.slider("Opinion about risk of getting sick with seasonal flu without vaccine.", 1, 5, 3)
    seas_sick = st.slider("Worry of getting sick from taking seasonal flu vaccine.", 1, 5, 3)

    age = st.selectbox('Age group:', ale_age)
    edu = st.selectbox('Education level:', ale_edu)
    race = st.selectbox('Race:', ale_race)
    sex = st.selectbox('Sex:', ale_sex)
    income = st.selectbox('Income:', ale_income)
    marital = st.selectbox('Marital Status:', ale_marital)
    rent = st.selectbox('Housing Situation:', ale_rent)
    employ = st.selectbox('Employment Status:', ale_employ)

    household_adults = st.slider("Number of other adults in household (highest is 3 and above).", 0, 3, 0)
    household_children = st.slider("Number of children in household (highest is 3 and above).", 0, 3, 0)

    ok = st.button("Complete.")
    if ok:
        X = np.array([[concern, knowledge, b1, b2, b3, b4, b5, b6, b7, dr_h1n1, dr_seas, chronic, child, hcw, ins, 
        h1n1_eff, h1n1_risk, h1n1_sick, seas_eff, seas_risk, seas_sick, age, edu, race, sex, income, marital, rent, employ, household_adults, household_children]])
        X[:, 0] = le_concern.transform(X[:,0])
        X[:, 1] = le_knowledge.transform(X[:,1])
        X[:, 2] = le_b1.transform(X[:,2])
        X[:, 3] = le_b2 .transform(X[:,3]) 
        X[:, 4] = le_b3.transform(X[:,4]) 
        X[:, 5] = le_b4.transform(X[:,5])
        X[:, 6] = le_b5.transform(X[:,6])
        X[:, 7] = le_b6.transform(X[:,7])
        X[:, 8] = le_b7.transform(X[:,8])
        X[:, 9] = le_dr_h1n1.transform(X[:,9])
        X[:, 10] = le_dr_seas.transform(X[:,10])
        X[:, 11] = le_chronic.transform(X[:,11])
        X[:, 12] = le_child.transform(X[:,12]) 
        X[:, 13] = le_hcw.transform(X[:,13])
        X[:, 14] = le_ins.transform(X[:,14])
        X[:, 21] = le_age.transform(X[:,21])
        X[:, 22] = le_edu.transform(X[:,22])
        X[:, 23] = le_race.transform(X[:,23])
        X[:, 24] = le_sex.transform(X[:,24])
        X[:, 25] = le_income.transform(X[:,25])
        X[:, 26] = le_marital.transform(X[:,26])
        X[:, 27] = le_rent.transform(X[:,27]) 
        X[:, 28] = le_employ.transform(X[:,28])
        X = X.astype(float)

        status = classifier.predict(X)
        st.subheader(f'Vaccination status: {status}')