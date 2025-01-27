import numpy as np
import pandas as pd  # Import pandas to load and process the CSV file
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from sklearn.decomposition import PCA


def get_one_encoding(labels):
    # Set sparse_output=False to get dense array
    encoder = OneHotEncoder(sparse_output=False)
    one_hot_encoded = encoder.fit_transform(labels.reshape(-1, 1))

    return one_hot_encoded


def get_data(dataset, data_encoding, seed=0, test_size=0.3):
    """
    Get data of the required dataset

    Parameters:
        - dataset (str): Name of the dataset.
        - data_encoding (str) : Specifies the data encoding method to be used. This parameter affects how data (i.e., target labels)
          is processed within the model. Example: 'no-encoding', 'one-hot-encoding'.
        - seed (int): Seed for random number generation (used only for synthetic exp).
        - test_size (float): Proportion of the dataset to include in the test split.

    Returns:
        - data_train (tuple): Tuple containing features and labels for the training set.
        - data_test (tuple): Tuple containing features and labels for the testing set.
        - map_class_dict (dict): A dictionary that maps the predicted class values (used internally by the model)
          to their original dataset class values.
    """
    already_split = False  # Flag to indicate if the dataset is sepsis. In this case, we already have two datasets separated: one for training and another for testing.
    categorical_data = False
    map_class_dict = {}
    if dataset == "diabetes":
        # DESCRIPTION:
        # -------------------------
        # The objective of the dataset is to diagnostically predict whether or not a patient has diabetes,
        # based on certain diagnostic measurements included in the dataset. Several constraints were placed on the
        # selection of these instances from a larger database. In particular, all patients here are females at least
        # 21 years old of Pima Indian heritage.
        # -------------------------
        # NUMBER OF INSTANCES: 768
        # NUMBER OF ATTRIBUTES:  8 + class (all numeric-valued)
        #       1. Number of times pregnant
        #       2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
        #       3. Diastolic blood pressure (mm Hg)
        #       4. Triceps skin fold thickness (mm)
        #       5. 2-Hour serum insulin (mu U/ml)
        #       6. Body mass index (weight in kg/(height in m)^2)
        #       7. Diabetes pedigree function
        #       8. Age (years)
        #       9. Class variable (0 or 1)
        # -------------------------
        # KAGGLE: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
        # MORE INFO: https://www.openml.org/search?type=data&sort=runs&id=37&status=active
        # POSSIBLE COMPARISON (TO INVESTIGATE):
        # - paper: https://link.springer.com/article/10.1186/s12911-024-02582-4
        # - code: https://github.com/ChristelSirocchi/medical-informed-ML
        # DOUBTS: Are the missing values? How are features scaled in other approaches?

        # Assuming the diabetes dataset is stored in 'data/datasets/diabetes.csv'
        file_path = "data/datasets/diabetes.csv"

        df = pd.read_csv(file_path)
        x = df.iloc[:, :-1].values  # Features
        y = df.iloc[:, -1].values  # Target variable

        if data_encoding == "one-hot-encoding":
            y = get_one_encoding(y)
        elif data_encoding == "no-encoding":
            y[y == 0] = -1
            map_class_dict[-1] = 0
            y = y.reshape(-1, 1)

    elif dataset == "gestational":
        # DESCRIPTION:
        # -------------------------
        # Gestational diabetes is a type of high blood sugar that develops during pregnancy. It can occur at any stage
        # of pregnancy and cause problems for both the mother and the baby, during and after birth. The risks can be
        # reduced if they are early detected and managed, especially in areas where only periodic tests of pregnant
        # women are available. The dataset was obtained from the Kurdistan region laboratories, which collected
        # information from pregnant women with and without diabetes.
        # -------------------------
        # NUMBER OF INSTANCES: 1012
        # NUMBER OF ATTRIBUTES:  6 + class
        #       1. Age (16-45)
        #       2. Pregnancy No. (1-9)
        #       3. Weight (43-126)
        #       4. Height (135-196)
        #       5. BMI (15 - 54.3)
        #       6. Heredity (0-1)
        #       7. Class variable: nodiabetes=0 or diabetes=1
        # MISSING VALUES: No
        # -------------------------
        # KAGGLE: https://www.kaggle.com/datasets/rasooljader/gestational-diabetes/data
        # POSSIBLE COMPARISON (TO INVESTIGATE):
        # - paper: ?
        # - code: ?
        # DOUBTS: Are features scaled in other approaches using this dataset?

        # Assuming the gestacional dataset is stored in 'data/datasets/gestational_diabetes.csv'
        file_path = "data/datasets/gestational_diabetes.csv"

        df = pd.read_csv(file_path, sep=",")
        x = df.iloc[:, :-1].values  # Features
        y = df.iloc[:, -1].values  # Target variable

        if data_encoding == "one-hot-encoding":
            y = get_one_encoding(y)
        elif data_encoding == "no-encoding":
            y[y == 0] = -1
            map_class_dict[-1] = 0
            y = y.reshape(-1, 1)

    elif dataset == "haberman":
        # DESCRIPTION:
        # -------------------------
        # The dataset contains cases from a study that was conducted between 1958 and 1970 at the University of
        # Chicago's Billings Hospital on the survival of patients who had undergone surgery for breast cancer.
        # -------------------------
        # NUMBER OF INSTANCES: 306
        # NUMBER OF ATTRIBUTES:  3 + class
        #       1. Age: Age of patient at time of operation (integer)
        #       2. Operation Year: Patient's year of operation (integer)
        #       3. Number of positive axillary nodes detected  (integer)
        #       4. Class variable: Survival Status (1 = the patient survived 5 years or longer, 2 = the patient died within 5 year)
        # MISSING VALUES: No
        # -------------------------
        # UCI: https://archive.ics.uci.edu/dataset/43/haberman+s+survival
        # POSSIBLE COMPARISON (TO INVESTIGATE):
        # - paper: ?
        # - code: ?
        #
        # DOUBTS: Are features scaled in other approaches using this dataset?

        # Assuming the haberman dataset is stored in 'data/datasets/haberman.data'
        file_path = "data/datasets/haberman.data"

        df = pd.read_csv(file_path, sep=",")
        x = df.iloc[:, :-1].values  # Features
        y = df.iloc[:, -1].values  # Target variable

        if data_encoding == "one-hot-encoding":
            y = get_one_encoding(y)
            map_class_dict[0] = 1
            map_class_dict[1] = 2
        elif data_encoding == "no-encoding":
            y[y == 2] = -1
            map_class_dict[-1] = 2
            y = y.reshape(-1, 1)

    elif dataset == "heart":
        # DESCRIPTION:
        # -------------------------
        # This database contains 76 attributes, but all published experiments refer to using a subset of 14 of them.
        # In particular, the Cleveland database is the only one that has been used by ML researchers to date.
        # The "goal" field refers to the presence of heart disease in the patient.  It is integer valued from 0
        # (no presence) to 4. Experiments with the Cleveland database have concentrated on simply attempting to
        # distinguish presence (values 1,2,3,4) from absence (value 0). The names and social security numbers of the
        # patients were recently removed from the database, replaced with dummy values.
        #
        # One file has been "processed", that one containing the Cleveland database.  All four unprocessed files also exist in this directory.
        # -------------------------
        # NUMBER OF INSTANCES: should be 303 but in Paulo's version is 270
        # NUMBER OF ATTRIBUTES:  5 + class
        #       1.  Age: age in years (integer)
        #       2.  Sex: (1 = male; 0 = female) (categorical)
        #       3.  Cp: (categorical)
        #       4.  Trestbps: resting blood pressure (on admission to the hospital) (integer)
        #       5.  Chol: serum cholestoral (integer)
        #       6.  Fbs: fasting blood sugar > 120 mg/dl (1 = true; 0 = false) (categorical)
        #       7.  Restecg: resting electrocardiographic results
        #               -- Value 0: normal
        #               -- Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
        #               -- Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria(categorical)
        #       8.  Thalach: maximum heart rate achieved (integer)
        #       9.  Exang: exercise induced angina (1 = yes; 0 = no)
        #       10. Oldpeak: ST depression induced by exercise relative to rest (integer)
        #       11. Slope: the slope of the peak exercise ST segment
        #               -- Value 1: upsloping
        #               -- Value 2: flat
        #               -- Value 3: downsloping
        #       12. Ca: number of major vessels (0-3) colored by flourosopy
        #       13. Thal: 3 = normal; 6 = fixed defect; 7 = reversable defect
        #       14. Num (predicted attribute): diagnosis of heart disease (angiographic disease status)
        #         -- Value 0: < 50% diameter narrowing (0 absence)
        #         -- Value 1: > 50% diameter narrowing (1,2,3,4 collapse to 1 presence)
        #         (in any major vessel: attributes 59 through 68 are vessels)
        # MISSING VALUES: No (at least for the 14 used attributes)
        # -------------------------
        # UCI: https://archive.ics.uci.edu/dataset/45/heart+disease
        # POSSIBLE COMPARISON (TO INVESTIGATE):
        # - link (https://archive.ics.uci.edu/dataset/45/heart+disease) (probably with the version having 303 samples)
        # DOUBTS: Are features scaled in other approaches using this dataset?

        # TODO: If we use version with 303, we have to do class collapsing to 0, 1 (binary problem)
        # Assuming the heart dataset is stored in 'data/datasets/processed_cleveland.data'
        file_path = "data/dataset/processed_cleveland.data"
        # TODO: implement new code here (see Issue#10)

    elif dataset == "liver":
        # DESCRIPTION:
        # -------------------------
        # The first 5 variables are all blood tests which are thought to be sensitive to liver disorders that might
        # arise from excessive alcohol consumption. Each line in the dataset constitutes the record of a single male
        # individual.
        # -------------------------
        # NUMBER OF INSTANCES: 345
        # NUMBER OF ATTRIBUTES:  5 + drinks (target?) and selector
        #       1. Mcv: mean corpuscular volume (continuos)
        #       2. Alkphos: alkaline phosphotase (continuos)
        #       3. Sgpt: alanine aminotransferase  (continuos)
        #       4. Sgot: aspartate aminotransferas (continuos)
        #       5. Gammagt: gamma-glutamyl transpeptidase (continuos)
        #       6. Drinks: number of half-pint equivalents of alcoholic beverages drunk per day (continuos-target?)
        #       7. Selector: field created by the BUPA researchers to split the data into train/test sets (categorical)
        # MISSING VALUES: No
        # -------------------------
        # UCI: https://archive.ics.uci.edu/dataset/60/liver+disorders
        # POSSIBLE COMPARISON (TO INVESTIGATE):
        # - paper: ?
        # - code: ?
        #
        # DOUBTS: Are features scaled in other approaches using this dataset?

        # Assuming the diabetes dataset is stored in 'data/datasets/liver_data.txt'
        file_path = "data/datasets/liver_data.txt"

        # TODO: See Issue#5

    elif dataset == "mammography":
        # DESCRIPTION:
        # -------------------------
        # This data set can be used to predict the severity (benign or malignant) of a mammographic mass lesion
        # from BI-RADS attributes and the patient's age. It contains a BI-RADS assessment, the patient's age and three
        # BI-RADS attributes together with the ground truth (the severity field) for 516 benign and 445 malignant masses
        # that have been identified on full field digital mammograms collected at the Institute of
        # Radiology of the University Erlangen-Nuremberg between 2003 and 2006
        # -------------------------
        # NUMBER OF INSTANCES: should be 961 but in Paulo's version is 830
        # NUMBER OF ATTRIBUTES:  5 + class
        #       1. BI-RADS assessment: 1 to 5 (ordinal, non-predictive!)
        #       2. Age: patient's age in years (integer)
        #       3. Shape: mass shape: round=1 oval=2 lobular=3 irregular=4 (nominal)
        #       4. Margin: mass margin: circumscribed=1 microlobulated=2 obscured=3 ill-defined=4 spiculated=5 (nominal)
        #       5. Density: mass density high=1 iso=2 low=3 fat-containing=4 (ordinal)
        #       6. Class variable: Severity: benign=0 or malignant=1 (binominal, goal field!)
        # MISSING VALUES:
        #     - BI-RADS assessment:    2
        #     - Age:                   5
        #     - Shape:                31
        #     - Margin:               48
        #     - Density:              76
        #     - Severity:              0
        # -------------------------
        # UCI: https://archive.ics.uci.edu/dataset/161/mammographic+mass
        # POSSIBLE COMPARISON (TO INVESTIGATE):
        # - paper: https://link.springer.com/article/10.1007/s10552-024-01942-9
        # - code: ?
        # DOUBTS: How are missing values handled and features scaled in other approaches?

        # Assuming the diabetes dataset is stored in 'data/datasets/mammographic_masses.data'
        # TODO: manage missing values
        file_path = "data/datasets/mammographic_masses.data"

        # TODO: implement new code here (see Issue#11)

    elif dataset == "maternal-hr":
        # DESCRIPTION:
        # -------------------------
        # Data has been collected from different hospitals, community clinics, maternal health cares from the
        # rural areas of Bangladesh through the IoT based risk monitoring system. Age, Systolic Blood Pressure as
        # SystolicBP, Diastolic BP as DiastolicBP, Blood Sugar as BS, Body Temperature as BodyTemp, HeartRate and
        # RiskLevel. All these are the responsible and significant risk factors for maternal mortality, that is one of
        # the main concern of SDG of UN.
        # -------------------------
        # NUMBER OF INSTANCES: 1014
        # NUMBER OF ATTRIBUTES: 6 + 3 classes
        #       1. Age: any ages in years when a women during pregnant (integer)
        #       2. SystolicBP: upper value of Blood Pressure in mmHg, another significant attribute during pregnancy (integer)
        #       3. DiastolicBP: lower value of Blood Pressure in mmHg, another significant attribute during pregnancy (integer)
        #       4. BS: Blood glucose levels is in terms of a molar concentration (integer)
        #       5. BodyTemp: mass density high=1 iso=2 low=3 fat-containing=4 (integer)
        #       6. HeartRate: A normal resting heart rate (integer)
        #       7. RiskLevel: Predicted Risk Intensity Level during pregnancy considering the previous attribute
        #                     (target, categorical: low, medium, and high)
        # MISSING VALUES: No
        #
        # -------------------------
        # AVAILABLE AT: https://archive.ics.uci.edu/dataset/863/maternal+health+risk
        # (also KAGGLE: https://www.kaggle.com/datasets/csafrit2/maternal-health-risk-data)
        # POSSIBLE COMPARISON (TO INVESTIGATE):
        # - paper: https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2023.1213436/full (check others?)
        # - code: ?
        # - scholar: https://scholar.google.com/scholar?cites=1019849447949783013&as_sdt=2005&sciodt=0,5&hl=en
        #
        # DOUBTS: Are features scaled in other approaches using this dataset?
        # Answer: In Togunwa et al. (2023) the StandardScaler was applied to scale the data (and we do the same as final step for all datasets).
        
        categorical_data = False

        # Assuming the maternal health risk dataset is stored in 'data/datasets/maternal_health_risk_data_set.csv'
        file_path = "data/datasets/maternal_health_risk_data_set.csv"
        df = pd.read_csv(file_path, sep=",")

        # Code the target (categorical) variables into numerical variables to facilitate computation,
        # low-risk, mid-risk and high-risk classes are coded as 0, 1, and 2 respectively.
        df['RiskLevel'] = df['RiskLevel'].map({'low risk': 0, 'mid risk': 1, 'high risk': 2})

        # The heart rate variable had a minimum value of 7, which is not biologically plausible,
        # consequently, two data instances with this outlier value are re-imputed with the mode value of 70.
        df['HeartRate'] = df['HeartRate'].replace(7, 70)
        
        x = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        if data_encoding == "one-hot-encoding":
            y = get_one_encoding(y)
        elif data_encoding == "no-encoding":
            # Map the target classes to -1, 0, 1
            y[y == 0] = -1
            y[y == 1] = 0
            y[y == 2] = 1
            map_class_dict[-1] = 0
            map_class_dict[0] = 1
            map_class_dict[1] = 2
            y = y.reshape(-1, 1)

    elif dataset == "obesity":
        # DESCRIPTION:
        # -------------------------
        # This dataset include data for the estimation of obesity levels in individuals from the countries of Mexico,
        # Peru and Colombia, based on their eating habits and physical condition. The data contains 17 attributes and
        # 2111 records, the records are labeled with the class variable NObesity (Obesity Level), that allows
        # classification of the data using the values of Insufficient Weight, Normal Weight, Overweight Level I,
        # Overweight Level II, Obesity Type I, Obesity Type II and Obesity Type III. 77% of the data was generated
        # synthetically using the Weka tool and the SMOTE filter, 23% of the data was collected directly from users
        # through a web platform.
        # -------------------------
        # NUMBER OF INSTANCES: 2111
        # NUMBER OF ATTRIBUTES: 16 + 7 classes
        #       1. Gender (categorical)
        #       2. Age (continuous)
        #       3. Height (continuous)
        #       4. Weight (continuous)
        #       5. Family_history_with_overweight: Has a family member suffered or suffers from overweight? (binary)
        #       6. FAVC: Do you eat high caloric food frequently? (binary)
        #       7. FCVC: Do you usually eat vegetables in your meals? (integer)
        #       8. NCP: How many main meals do you have daily? (continuos)
        #       9. CAEC: Do you eat any food between meals? (categorical)
        #       10. SMOKE: Do you smoke? (binary)
        #       11. CH2O: How much water do you drink daily? (continuos)
        #       12. SCC: Do you monitor the calories you eat daily? (binary)
        #       13. FAF: How often do you have physical activity? (continuos)
        #       14. TUE: How much time do you use technological devices such as cell phone, videogames, television, computer and others? (integer)
        #       15. CALC:  How often do you drink alcohol? (categorical)
        #       16. MTRANS: Which transportation do you usually use? (categorical)
        #       17. TUE: Obesity level (target, categorical: low, high, medium)
        #
        # MISSING VALUES: No
        #
        # -------------------------
        # AVAILABLE AT: https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition
        # POSSIBLE COMPARISON (TO INVESTIGATE):
        # - paper:
        # - code: ?
        # - scholar: https://scholar.google.com/scholar?cites=9313437639045820138&as_sdt=2005&sciodt=0,5&hl=en
        #
        # DOUBTS: Are features scaled in other approaches using this dataset?
        #
        # Assuming the obesity dataset is stored in 'data/datasets/obesity.csv'
        # TODO: implement new code here (see Issue#14)
        categorical_data = False #Solo perchè lo faccio io, ma in realtà dovrebbe essere True
        already_split = True #We already have two datasets separated: one for training and another for testing.
        
        file_path = "data/datasets/obesity.csv"
        df = pd.read_csv(file_path, sep=",")

        binary_cols = ['Gender', 'family_history_with_overweight', 'SMOKE', 'SCC','FAVC']
        multi_cat_cols = ['CALC', 'MTRANS']
        ordinal_cols = ['CAEC']
        all_cat_cols = binary_cols + multi_cat_cols + ordinal_cols
        label_encoder = LabelEncoder()
        for col in all_cat_cols:
            df[col] = label_encoder.fit_transform(df[col])
        x_cat = df[all_cat_cols]
        
        numerical_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
        x = df[numerical_cols].astype(np.float32)  
        
        y = df.iloc[:, -1].values
        if data_encoding == "one-hot-encoding":
            y = get_one_encoding(y)
        elif data_encoding == "no-encoding":
            for i, label in enumerate(np.unique(y)):
                y[y == label] = i
                map_class_dict[i] = label
            y = y.reshape(-1, 1)
        
        
        #PCA to reduce the number of features
        pca = PCA(n_components=8)
        
        x_train_reals, x_test_reals, x_cat_train, x_cat_test, y_train, y_test = train_test_split(x, x_cat, y, test_size=test_size, random_state=seed)
        
        x_train = np.hstack((x_train_reals, x_cat_train))
        x_test = np.hstack((x_test_reals, x_cat_test))
        
        x_train = pca.fit_transform(x_train)
        x_test = pca.transform(x_test)
        
        
        
    elif dataset == "preeclampsia":
        # DESCRIPTION:
        # -------------------------
        # -------------------------
        # NUMBER OF INSTANCES: 1640
        # NUMBER OF ATTRIBUTES: paulo's version uses 6 + class (but original one has more).
        # MISSING VALUES: No
        #
        # -------------------------
        # AVAILABLE AT: https://www.icpsr.umich.edu/web/HMCA/studies/21640 (restricted access)
        # POSSIBLE COMPARISON (TO INVESTIGATE):
        # - paper: https://www.sciencedirect.com/science/article/pii/S0020025524004699 (check others?)
        # - code: ?
        #
        # DOUBTS: Are features scaled in other approaches using this dataset?
        #
        # Assuming the preclampsia dataset is stored in 'data/datasets/preeclampsia.csv'
        # TODO: This is in stand-by
        file_path = "data/datasets/preeclampsia.csv"

        df = pd.read_csv(file_path, sep=",")
        x = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        if data_encoding == "one-hot-encoding":
            y = get_one_encoding(y)
        elif data_encoding == "no-encoding":
            y[y == 0] = -1
            map_class_dict[-1] = 0
            y = y.reshape(-1, 1)

    elif dataset == "sepsis":
        # DESCRIPTION:
        # -------------------------
        # The dataset consists of 110,204 admissions of 84,811 hospitalized subjects between 2011 and 2012 in Norway
        # who were diagnosed with infections, systemic inflammatory response syndrome, sepsis by causative microbes, or
        # septic shock. The prediction task is to determine whether a patient survived or is deceased at a time of about
        # 9 days after collecting their medical record at the hospital. This is an important prediction problem in
        # clinical medicine. Sepsis is a life-threatening condition triggered by an immune overreaction to infection,
        # leading to organ failure or even death. Sepsis is associated with immediate death risk, often killing patients
        # within one hour. This renders many laboratory tests and hospital analyses impractical for timely diagnosis and
        # treatment. Being able to predict the survival of patients within minutes with as few and easy-to-retrieve
        # medical features as possible is very important.
        # -------------------------
        # NUMBER OF INSTANCES: 110,204
        # NUMBER OF ATTRIBUTES:  3 + class
        #       1. Age: Age of the patient in years
        #       2. Sex of person (0: male, 1: female)
        #       3. Episode_number (Number of prior Sepsis episodes)
        #       4. Class variable: Status of the patient after 9,351 days of being admitted to the hospital.
        #          Values are encoded as follows: {1: Alive, 0: Dead}
        # MISSING VALUES: No
        #
        # Are there recommended data splits?
        #
        # No recommendation, standard train-test split could be used. Can use three-way holdout split
        # (i.e., training, validation/development, testing) when doing model selection.
        # NOTE: there is validation cohort from South Korea used by Chicco and Jurman (2020) as an external validation
        # cohort to confirm the generalizability of their proposed approach (We are going to compare with them).
        # -------------------------
        # UCI: https://archive.ics.uci.edu/dataset/827/sepsis+survival+minimal+clinical+records
        # POSSIBLE COMPARISON (TO INVESTIGATE):
        # - paper: https://pubmed.ncbi.nlm.nih.gov/33051513/ (check others?)
        # - code: ?
        #
        # DOUBTS: Are features scaled in other approaches using this dataset?

        # Assuming the sepsis dataset is stored in 'data/datasets/sepsis/'
        # TODO: implement new code here (see Issue#12)
        
        already_split = True #We already have two datasets separated: one for training and another for testing.
        
        filepath_primary_cohort = "data/datasets/sepsis/s41598-020-73558-3_sepsis_survival_primary_cohort.csv"
        filepath_validation_cohort = "data/datasets/sepsis/s41598-020-73558-3_sepsis_survival_validation_cohort.csv"
        
        df_primary = pd.read_csv(filepath_primary_cohort, sep=",")
        df_validation = pd.read_csv(filepath_validation_cohort, sep=",")
        
        
        #Train data
        x_train = df_primary.iloc[:, :-1].values
        y_train = df_primary.iloc[:, -1].values
        
        
        oversample = RandomOverSampler(sampling_strategy='minority') 
        x_train, y_train = oversample.fit_resample(x_train, y_train)
        
        #Test data
        x_test = df_validation.iloc[:, :-1].values
        y_test = df_validation.iloc[:, -1].values
        
        
        if data_encoding == "one-hot-encoding":
            y_train = get_one_encoding(y_train)
            y_test = get_one_encoding(y_test)
            
        elif data_encoding == "no-encoding":
            y_train[y_train == 0] = -1
            map_class_dict[-1] = 0
            y_train = y_train.reshape(-1, 1)
            
            y_test[y_test == 0] = -1
            y_test = y_test.reshape(-1, 1)
            

    else:
        raise ValueError("Invalid dataset name.")

    # Data normalization
    scaler = StandardScaler()

    if not already_split: #If it is not sepsis, we have to split the data. Otherwise, we already have two datasets separated: one for training and another for testing.
        if categorical_data:
            x_train, x_test, x_cat_train, x_cat_test, y_train, y_test = train_test_split(
            x, x_cat, y, test_size=test_size, random_state=seed)
        else:
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=test_size, random_state=seed)
    
    x_train_normalized = scaler.fit_transform(x_train)
    x_test_normalized = scaler.transform(x_test)

    if(categorical_data):
        x_train_normalized = np.hstack((x_train_normalized, x_cat_train))
        x_test_normalized = np.hstack((x_test_normalized, x_cat_test))
    
    data_train = (x_train_normalized, y_train)
    data_test = (x_test_normalized, y_test)
    return data_train, data_test, map_class_dict
