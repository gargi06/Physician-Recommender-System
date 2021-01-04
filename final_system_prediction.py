# To resolve some future warning, upgrade version of sklearn by un-commenting below line:
# pip install --user --upgrade scikit-learn==0.22.2

'''
This file is the main backend file for the entire project.
This file takes an input string of symptoms from user and returns the predicted system
'''

'''
Please note the various libraries used in this file and install them in your system before 
running this file.
'''

'''
CountVectorizer and TfidfVectorizer are used to covert input string to vectors,
so that it can be processed by the classification model.

Since we also need to convert the input string into vectors using the same tfidf object used 
while building the model, we have saved the tfidf object for later use using pickle.
'''

'''
The classification model used in this file is Random Forest Classifier from sklearn.ensemble
This model has been carefully picked after comparing results with various other classification models.
By proper hyper-parameter tuning, the accuracy of prediction can be further improved. 
'''

'''
The dataset used in this file needed a lot of manual corrections done which has increased
the length of this file manifold.
'''

'''
3 files are saved using pickle.
First --> tfidf object fitted on training data, to be used again to convert user input to vector
Second --> dataset after completing all manual entries, to be used again to find the final diease
Third --> random forest classification model, after training it.
'''

'''
The model is trained on 4 parameters : symptoms + causes + risk_factors + overview.
It outputs the final system, the probability of final system and the predicted diease belonging predicted system.
'''

try:
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    import numpy as np
    import pandas as pd
    import pickle
    from sklearn.ensemble import RandomForestClassifier
    f = open('final_system_model.sav')
    classifier = pickle.load(open('final_system_model.sav', 'rb'))

except:
    import re

    # importing
    # Please change the file path to appropriate location.
    df = pd.read_csv('data/df_diseases.csv')

    # dropping col's
    df.drop([df.columns[0], df.columns[2]], axis=1, inplace=True)

    # Filling NaN values with empty string
    df.fillna('', inplace=True)

    # some pre-processing
    for i in range(len(df)):

        df.loc[i, 'symptoms'] = re.sub(
            '[^A-Za-z0-9., ]+', '', df.loc[i, 'symptoms'])
        df.loc[i, 'causes'] = re.sub(
            '[^A-Za-z0-9., ]+', '', df.loc[i, 'causes'])
        df.loc[i, 'risk_factor'] = re.sub(
            '[^A-Za-z0-9., ]+', '', df.loc[i, 'risk_factor'])

        df.loc[i, 'overview'] = re.sub(
            '[^A-Za-z0-9., ]+', '', df.loc[i, 'overview'])
        df.loc[i, 'treatment'] = re.sub(
            '[^A-Za-z0-9., ]+', '', df.loc[i, 'treatment'])
        df.loc[i, 'medication'] = re.sub(
            '[^A-Za-z0-9., ]+', '', df.loc[i, 'medication'])

        df.loc[i, 'home_remedies'] = re.sub(
            '[^A-Za-z0-9., ]+', '', df.loc[i, 'home_remedies'])

    # some more preprocessing
    df['name'] = df['name'].str.lower()
    df['symptoms'] = df['symptoms'].str.lower()
    df['causes'] = df['causes'].str.lower()

    df['risk_factor'] = df['risk_factor'].str.lower()
    df['overview'] = df['overview'].str.lower()
    df['treatment'] = df['treatment'].str.lower()

    df['medication'] = df['medication'].str.lower()
    df['home_remedies'] = df['home_remedies'].str.lower()

    # Organ Dictionary
    organ_dict = {"muscular system": ["human skeleton", "joints", "ligaments", "muscular system", "tendons"],
                  "digestive system": ["mouth", "teeth", "tongue", "salivary glands", "parotid glands", "submandibular glands", "sublingual glands", "pharynx", "esophagus", "stomach", "small intestine", "duodenum", "jejunum", "ileum", "large intestine", "liver", "gallbladder", "mesentery", "pancreas", "anal canal", "anus", "blood cells"],
                  "respiratory system": ["nasal cavity", "pharynx", "larynx", "trachea", "bronchi", "lungs", "diaphragm"],
                  "urinary system": ["kidneys", "ureter", "bladder", "urethra"],
                  "reproductive organs": ["testes", "epididymis", "vas deferens", "seminal vesicles", "prostate", "bulbourethral glands", "ovaries", "fallopian tubes", "uterus", "vagina", "vulva", "clitoris", "placenta", "penis", "scrotum"],
                  "endocrine system": ["pituitary gland", "pineal gland", "thyroid gland", "parathyroid glands", "adrenal glands", "pancreas"],
                  "circulatory system": ["heart", "patent foramen ovale", "arteries", "veins", "capillaries", "lymphatic vessel", "lymph node", "bone marrow", "thymus", "spleen", "gut-associated lymphoid tissue", "tonsils", "interstitium"],
                  "nervous system": ["brain", "cerebrum", "cerebral hemispheres", "diencephalon", "the brainstem", "midbrain", "pons", "medulla oblongata", "cerebellum", "the spinal cord", "the ventricular system", "choroid plexus", "nerves", "cranial nerves", "spinal nerves", "ganglia", "enteric nervous system", "eye", "cornea", "iris", "ciliary body", "lens", "retina", "ear", "outer ear", "earlobe", "eardrum", "middle ear", "ossicles", "inner ear", "cochlea", "vestibule of the ear", "semicircular canals", "olfactory epithelium", "tongue", "taste buds"],
                  "integumentary system": ["mammary glands", "skin", "subcutaneous tissue"]}

    organ_keys = organ_dict.keys()
    # Creating a column named organs
    df['organs'] = [[] for _ in range(len(df))]

    # Filling the organ column
    for i in range(len(df)):
        s = df['symptoms'][i]+df['causes'][i] + \
            df['risk_factor'][i]+df['overview'][i]
        s_tokens = s.split(' ')

        for value in organ_keys:
            val = organ_dict.get(value)

            for j in val:
                if(j in s_tokens):
                    df.loc[i, 'organs'].append(j)

    # Creating systems column
    df['systems'] = [[] for _ in range(len(df))]
    key_list = list(organ_dict.keys())
    val_list = list(organ_dict.values())

    # Filling the systems column by matching organs column with the dict values
    for i in range(len(df)):
        for val in df['organs'][i]:
            for k in range(len(val_list)):
                if(val in val_list[k]):
                    df['systems'][i].append(key_list[k])

    # manually entered organs and systems
    #####################################################################################
    df.at[525, 'organs'] = ['large intestine']
    df.at[525, 'systems'] = ['digestive system']

    df.at[537, 'organs'] = ['nose', 'eyes', 'throat', 'skin']
    df.at[537, 'systems'] = ['respiratory system']

    df.at[539, 'organs'] = ['blood']
    df.at[539, 'systems'] = ['circulatory system']

    df.at[541, 'organs'] = ['brain']
    df.at[541, 'systems'] = ['nervous system']  # psycological disorder

    df.at[550, 'organs'] = ['blood cells']
    df.at[550, 'systems'] = ['circulatory system']  # type of blood cancer

    df.at[551, 'organs'] = ['brain']
    df.at[551, 'systems'] = ['nervous system']  # psycological disorder

    # infection in salivary glands due to bacteria
    df.at[552, 'organs'] = ['salivary gland']
    df.at[552, 'systems'] = ['digestive system']

    df.at[554, 'organs'] = ['muscles']
    df.at[554, 'systems'] = ['muscular system']

    df.at[555, 'organs'] = ['muscles']
    df.at[555, 'systems'] = ['muscular system']

    df.at[558, 'organs'] = ['muscles']
    df.at[558, 'systems'] = ['muscular system']  # bone marrow cancer

    df.at[563, 'organs'] = ['muscles']
    # extreme pain in muscles due to various reasons
    df.at[563, 'systems'] = ['muscular system']

    df.at[564, 'organs'] = ['legs', 'arms']
    # cancer that develops lump in legs or arms
    df.at[564, 'systems'] = ['muscular system']

    df.at[566, 'organs'] = ['nose', 'eyes', 'mouth']
    df.at[566, 'systems'] = ['respiratory system']  # cancer developing in nose

    # tumor nerves in peripheral nervous system
    df.at[576, 'organs'] = ['nerves']
    df.at[576, 'systems'] = ['nervous system']

    # the disease is classified as a disease of the respiratory sytem, but actually for nose fracture the
    # patient must be recommended to an orthopedic
    df.at[593, 'organs'] = ['nose']
    df.at[593, 'systems'] = ['respiratory system']  # nose fracture

    df.at[594, 'organs'] = ['fat']
    df.at[594, 'systems'] = ['integumentary system']  # obesity

    df.at[603, 'organs'] = ['brain']
    df.at[603, 'systems'] = ['nervous system']  # a type of brain tumor

    df.at[609, 'organs'] = ['joints']
    df.at[609, 'systems'] = ['muscular system']

    df.at[614, 'organs'] = ['joints', 'bones']
    df.at[614, 'systems'] = ['muscular system']

    df.at[617, 'organs'] = ['bone']
    df.at[617, 'systems'] = ['muscular system']  # bone cancer

    df.at[623, 'organs'] = ['thyroid gland']
    # cancer that occurs in the cells of thyroid gland
    df.at[623, 'systems'] = ['endocrine system']

    df.at[637, 'organs'] = ['nerves']
    df.at[637, 'systems'] = ['nervous system']  # tumor in certain nerves

    df.at[640, 'organs'] = ['salivary glands']
    df.at[640, 'systems'] = ['digestive system']  # tumor in salivary glands

    df.at[641, 'organs'] = ['pumonary veins']
    # defect in pulmonary veins of the heart
    df.at[641, 'systems'] = ['circulatory system']

    df.at[643, 'organs'] = ['brain']
    df.at[643, 'systems'] = ['nervous system']  # mental disorder

    df.at[647, 'organs'] = ['chest']
    df.at[647, 'systems'] = ['circulatory system']

    df.at[649, 'organs'] = ['hair', 'scalp']
    # tiny insects infect the scalp
    df.at[649, 'systems'] = ['integumentary system']

    df.at[670, 'organs'] = ['foot']
    df.at[670, 'systems'] = ['muscular system']

    df.at[679, 'organs'] = ['brain', 'pineal gland']
    # cancer develops in the pineal gland of brain
    df.at[679, 'systems'] = ['nervous system']

    df.at[689, 'organs'] = ['tissue']
    df.at[689, 'systems'] = ['muscular system']

    df.at[694, 'organs'] = ['lungs']
    df.at[694, 'systems'] = ['respiratory system']

    df.at[700, 'organs'] = ['muscles']
    df.at[700, 'systems'] = ['muscular system']

    df.at[701, 'organs'] = ['rectum', 'large intestine']
    df.at[701, 'systems'] = ['digestive system']

    df.at[702, 'organs'] = ['sinus', 'nose']
    df.at[702, 'systems'] = ['respiratory system']

    df.at[704, 'organs'] = ['popliteal artery', 'knee joint']
    df.at[704, 'systems'] = ['circulatory system']

    df.at[706, 'organs'] = ['knee', 'joint']
    df.at[706, 'systems'] = ['muscular system']

    df.at[714, 'organs'] = ['brain']
    df.at[714, 'systems'] = ['nervous system']  # psycological disorder

    df.at[715, 'organs'] = ['urine']
    df.at[715, 'systems'] = ['reproductive system']

    df.at[716, 'organs'] = ['thyroid gland']
    df.at[716, 'systems'] = ['endocrine system']

    df.at[722, 'organs'] = ['None']
    df.at[722, 'systems'] = ['Reproductive system']

    df.at[728, 'organs'] = ['brain']
    # psycological disorder, addiction
    df.at[728, 'systems'] = ['nervous system']

    df.at[735, 'organs'] = ['adrenal gland']
    df.at[735, 'systems'] = ['endocrine system']

    df.at[748, 'organs'] = ['muscles']
    df.at[748, 'systems'] = ['muscular system']

    df.at[750, 'organs'] = ['large intestine']
    # swelling of large intestine due to bacteria
    df.at[750, 'systems'] = ['digestive system']

    df.at[757, 'organs'] = ['pulmonary valve']
    df.at[757, 'systems'] = ['cirulatory system']

    df.at[758, 'organs'] = ['heart']
    df.at[758, 'systems'] = ['ciruclatory system']

    df.at[769, 'organs'] = ['body']
    df.at[769, 'systems'] = ['nervous system']

    df.at[775, 'organs'] = ['brain']
    df.at[775, 'systems'] = ['nervous system']

    df.at[776, 'organs'] = ['rectum']
    df.at[776, 'systems'] = ['reproductive system']

    # end of apurv code

    # Gargi

    # df['name'][270] = 'fibrocystic breasts'
    df.at[270, 'organs'] = ['breasts']
    df.at[270, 'systems'] = ['reproductive system']

    # df['name'][274] = 'fibrous dysplasia'
    df.at[274, 'organs'] = ['bone']
    df.at[274, 'systems'] = ['muscular system']

    # df['name'][281] = 'foot fracture (see: broken foot)'
    df.at[281, 'organs'] = ['foot', 'bones']
    df.at[281, 'systems'] = ['muscular system']

    # df['name'][282] = 'fracture, arm (see: broken arm)'
    df.at[282, 'organs'] = ['bone']
    df.at[282, 'systems'] = ['muscular system']

    # df['name'][298] = 'gastrointestinal stromal tumor (gist)'
    df.at[298, 'organs'] = ['stomach', 'intestine']
    df.at[298, 'systems'] = ['digestive system']

    # df['name'][300] = 'gender identity disorder (see: gender dysphoria)'
    df.at[300, 'organs'] = ['reproductive system']
    df.at[300, 'systems'] = ['reproductive system']

    # df['name'][306] = 'giardia infection (giardiasis)'
    df.at[306, 'organs'] = ['stomach', 'intestine']
    df.at[306, 'systems'] = ['digestive system']

    # df['name'][309] = 'glioblastoma'
    df.at[309, 'organs'] = ['brain', 'spine']
    df.at[309, 'systems'] = ['nervous system']

    # df['name'][318] = 'growing pains'
    df.at[318, 'organs'] = ['legs', 'thighs', 'knees', 'calves']
    df.at[318, 'systems'] = ['muscular system']

    # df['name'][325] = 'hamstring injury'
    df.at[325, 'organs'] = ['thigh']
    df.at[325, 'systems'] = ['muscular system']

    # df['name'][326] = 'hand fracture (see: broken hand)'
    df.at[326, 'organs'] = ['hand']
    df.at[326, 'systems'] = ['muscular system']

    # df['name'][330] = 'head and neck cancers'
    df.at[330, 'organs'] = ['mouth', 'sinuses', 'nose', 'throat']
    df.at[330, 'systems'] = ['respiratory system']

    # df['name'][346] = 'hemifacial spasm
    df.at[346, 'organs'] = ['facial muscles']
    df.at[346, 'systems'] = ['muscular system']

    # df['name'][353] = 'hepatocellular carcinoma'
    df.at[353, 'organs'] = ['liver']
    df.at[353, 'systems'] = ['digestive system']

    # df['name'][361] = 'hip dysplasia'
    df.at[361, 'organs'] = ['hip']
    df.at[361, 'systems'] = ['muscular system']

    # df['name'][363] = 'hip impingement'
    df.at[363, 'organs'] = ['hip']
    df.at[363, 'systems'] = ['muscular system']

    # df['name'][364] = 'hip labral tear'
    df.at[364, 'organs'] = ['labrum', 'hip']
    df.at[364, 'systems'] = ['muscular system']

    # df['name'][374] = 'hydronephrosis'
    df.at[374, 'organs'] = ['abdomen', 'kidney']
    df.at[374, 'systems'] = ['digestive system', 'urinary system']

    # df['name'][386] = 'hypoglycemia, diabetic (see: diabetic hypoglycemia)'
    df.at[386, 'organs'] = ['low sugar']
    df.at[386, 'systems'] = ['digestive system']

    # df['name'][395] = 'idiopathic hypersomnia'
    df.at[395, 'organs'] = ['neurological sleep disorder']
    df.at[395, 'systems'] = ['nervous system']

    # df['name'][402] = 'ncomplete fracture (see: greenstick fractures)'
    df.at[402, 'organs'] = ['bones', 'forearm', 'legs']
    df.at[402, 'systems'] = ['muscular system']

    # df['name'][408] = 'inflammatory bowel disease (ibd)'
    df.at[408, 'organs'] = ['digestive tract']
    df.at[408, 'systems'] = ['digestive system']

    # df['name'][413] = 'ingrown toenails'
    df.at[413, 'organs'] = ['toenails', 'nails', 'finger']
    df.at[413, 'systems'] = ['integumentary system']

    # df['name'][416] = 'intestinal obstruction'
    df.at[416, 'organs'] = ['abdomen', 'intestine']
    df.at[416, 'systems'] = ['digestive system']

    # df['name'][418] = 'intraductal carcinoma (see: ductal carcinoma in situ (dcis))'
    df.at[418, 'organs'] = ['breasts']
    df.at[418, 'systems'] = ['reproductive system']

    # df['name'][419] = 'intussusception'
    df.at[419, 'organs'] = ['intestine']
    df.at[419, 'systems'] = ['digestive system']

    # df['name'][432] = 'Kaposi sarcoma'
    df.at[432, 'organs'] = ['skin', 'lymph nodes',
                            'mucous membranes lining the mouth', 'nose', 'throat']
    df.at[432, 'systems'] = ['integumentary system']

    # df['name'][436] = 'kidney cysts'
    df.at[436, 'organs'] = ['kidney']
    df.at[436, 'systems'] = ['urinary system']

    # df['name'][438] = 'klatskin tumor (see: hilar cholangiocarcinoma)'
    df.at[438, 'organs'] = ['hepatic duct', 'liver']
    df.at[438, 'systems'] = ['digestive system']

    # df['name'][439] = 'klinefelter syndrome'
    df.at[439, 'organs'] = ['genetic', 'reproductive']
    df.at[439, 'systems'] = ['reproductive system']

    # df['name'][450] = 'leg fracture (see: broken leg)'
    df.at[450, 'organs'] = ['leg', 'bone']
    df.at[450, 'systems'] = ['muscular system']

    # df['name'][451] = 'legg-calve-perthes disease'
    df.at[451, 'organs'] = ['femur', 'hip', 'bone', 'thigh bone']
    df.at[451, 'systems'] = ['circulatory system']

    # df['name'][452] = 'legionnaires disease'
    df.at[452, 'organs'] = ['heart', 'lungs', 'muscles']
    df.at[452, 'systems'] = ['muscular system',
                             'respiratory system', 'circulatory system']

    # df['name'][453] = 'leiomyosarcoma'
    df.at[453, 'organs'] = ['stomach', 'bladder', 'intestine', 'uterus']
    df.at[453, 'systems'] = ['reproductive system', 'urinary system']

    # df['name'][454] = 'leukemia, acute lymphocytic (see: acute lymphocytic leukemia)'
    df.at[454, 'organs'] = ['blood', 'bone marrow']
    df.at[454, 'systems'] = ['circulatory system']

    # df['name'][455] = 'leukemia, acute myelogenous (see: acute myelogenous leukemia)'
    df.at[455, 'organs'] = ['blood', 'bone marrow']
    df.at[455, 'systems'] = ['circulatory system']

    # df['name'][456] = 'leukemia, chronic lymphocytic (see: chronic lymphocytic leukemia)'
    df.at[456, 'organs'] = ['blood', 'bone marrow']
    df.at[456, 'systems'] = ['circulatory system']

    # df['name'][457] = 'leukemia, chronic myelogenous (see: chronic myelogenous leukemia)'
    df.at[457, 'organs'] = ['blood', 'bone marrow']
    df.at[457, 'systems'] = ['circulatory system']

    # df['name'][459] = 'leukemia, hairy cell (see: hairy cell leukemia)'
    df.at[459, 'organs'] = ['blood', 'bone marrow']
    df.at[459, 'systems'] = ['circulatory system']

    # df['name'][469] = 'liposarcoma'
    df.at[469, 'organs'] = ['abdomen', 'thigh', 'knee', 'fat cells']
    df.at[469, 'systems'] = ['muscular system']

    # df['name'][470] = 'listeriosis (see: listeria infection)'
    df.at[470, 'organs'] = ['brain', 'spinal cord', 'bloodstream']
    df.at[470, 'systems'] = ['nervous system', 'circulatory system']

    # df['name'][476] = 'lobular carcinoma in situ (lcis)'
    df.at[476, 'organs'] = ['breasts']
    df.at[476, 'systems'] = ['reproductive system']

    # df['name'][481] = 'low sex drive in women'
    df.at[481, 'organs'] = ['reproductive system']
    df.at[481, 'systems'] = ['reproductive system']

    # df['name'][501] = 'mammary duct ectasia'
    df.at[501, 'organs'] = ['breasts']
    df.at[501, 'systems'] = ['reproductive system']

    # df['name'][502] = 'manic-depressive illness (see: bipolar disorder)'
    df.at[502, 'organs'] = ['nervous system']
    df.at[502, 'systems'] = ['nervous system']

    # df['name'][507] = 'mcad deficiency'
    df.at[507, 'organs'] = ['skeletal- and heart muscle', 'liver', 'brain']
    df.at[507, 'systems'] = ['circulatory system',
                             'digestive system', 'nervous system']

    # df['name'][509] = 'medulloblastoma'
    df.at[509, 'organs'] = ['cerebellum', 'brain']
    df.at[509, 'systems'] = ['nervous system']

    # df['name'][523] = 'metatarsalgia'
    df.at[523, 'organs'] = ['foot']
    df.at[523, 'systems'] = ['muscular system']

    # end of gargi code

    # pradeumna

    df.at[779, 'organs'] = ['anus']
    df.at[779, 'systems'] = ['digestive system']

    df.at[791, 'organs'] = ['human skeleton']
    df.at[791, 'systems'] = ['muscular system']

    df.at[808, 'organs'] = ['brain']
    df.at[808, 'systems'] = ['nervous system']

    df.at[810, 'organs'] = ['stomach']
    df.at[810, 'systems'] = ['digestive system']

    df.at[811, 'organs'] = ['tendons']
    df.at[811, 'systems'] = ['muscular system']

    df.at[815, 'organs'] = ['joints']
    df.at[815, 'systems'] = ['muscular system']

    df.at[823, 'organs'] = ['human skeleton']
    df.at[823, 'systems'] = ['nervous system']

    df.at[826, 'organs'] = ['brain']
    df.at[826, 'systems'] = ['nervous system']

    df.at[827, 'organs'] = ['brain']
    df.at[827, 'systems'] = ['nervous system']

    df.at[831, 'organs'] = ['brain']
    df.at[831, 'systems'] = ['nervous system']

    df.at[841, 'organs'] = ['skin']
    df.at[841, 'systems'] = ['integumentary system']

    df.at[848, 'organs'] = ['brain']
    df.at[848, 'systems'] = ['nervous system']

    df.at[849, 'organs'] = ['brain']
    df.at[849, 'systems'] = ['nervous system']

    df.at[853, 'organs'] = ['brain']
    df.at[853, 'systems'] = ['nervous system']

    df.at[859, 'organs'] = ['lungs']
    df.at[859, 'systems'] = ['respiratory system']

    df.at[867, 'organs'] = ['large intestine']
    df.at[867, 'systems'] = ['digestive system']

    df.at[876, 'organs'] = ['brain']
    df.at[876, 'systems'] = ['nervous system']

    df.at[889, 'organs'] = ['brain']
    df.at[889, 'systems'] = ['nervous system']

    df.at[905, 'organs'] = ['muscular']
    df.at[905, 'systems'] = ['muscular system']

    df.at[928, 'organs'] = ['vagina']
    df.at[928, 'systems'] = ['reproductive system']

    df.at[949, 'organs'] = ['muscular system']
    df.at[949, 'systems'] = ['muscular system']

    df.at[962, 'organs'] = ['brain']
    df.at[962, 'systems'] = ['nervous system']

    df.at[969, 'organs'] = ['skin']
    df.at[969, 'systems'] = ['integumentary system']

    df.at[977, 'organs'] = ['joints']
    df.at[977, 'systems'] = ['muscular system']

    df.at[982, 'organs'] = ['bone marrow']
    df.at[982, 'systems'] = ['circulatory system']

    df.at[995, 'organs'] = ['muscular system']
    df.at[995, 'systems'] = ['muscular system']

    df.at[996, 'organs'] = ['joints']
    df.at[996, 'systems'] = ['muscular system']

    df.at[1006, 'organs'] = ['kidneys']
    df.at[1006, 'systems'] = ['urinary system']

    df.at[1009, 'organs'] = ['skin']
    df.at[1009, 'systems'] = ['integumentary system']

    df.at[1014, 'organs'] = ['bone marrow']
    df.at[1014, 'systems'] = ['circulatory system']

    df.at[1015, 'organs'] = ['blood']
    df.at[1015, 'systems'] = ['circulatory system']

    df.at[1022, 'organs'] = ['thyroid gland']
    df.at[1022, 'systems'] = ['endocrine system']

    df.at[1024, 'organs'] = ['thyroid gland']
    df.at[1024, 'systems'] = ['endocrine system']

    df.at[1029, 'organs'] = ['human skeleton']
    df.at[1029, 'systems'] = ['muscular system']

    df.at[1042, 'organs'] = ['human skeleton']
    df.at[1042, 'systems'] = ['muscular system']

    df.at[1045, 'organs'] = ['heart']
    df.at[1045, 'systems'] = ['circulatory system']

    df.at[1057, 'organs'] = ['brain']
    df.at[1057, 'systems'] = ['nervous system']

    df.at[1060, 'organs'] = ['vagina']
    df.at[1060, 'systems'] = ['reproductive system']

    df.at[1061, 'organs'] = ['brain']
    df.at[1061, 'systems'] = ['nervous system']

    df.at[1080, 'organs'] = ['intestines']
    df.at[1080, 'systems'] = ['digestive system']

    df.at[1083, 'organs'] = ['anus']
    df.at[1083, 'systems'] = ['digestive system']

    df.at[1086, 'organs'] = ['muscular system']
    df.at[1086, 'systems'] = ['muscular system']

    df.at[1124, 'organs'] = ['trachea']
    df.at[1124, 'systems'] = ['respiratory system']

    df.at[1143, 'organs'] = ['blood']
    df.at[1143, 'systems'] = ['circulatory system']

    df.at[1147, 'organs'] = ['blood']
    df.at[1147, 'systems'] = ['circulatory system']

    df.at[1159, 'organs'] = ['blood']
    df.at[1159, 'systems'] = ['circulatory system']

    df.at[1167, 'organs'] = ['human skeleton']
    df.at[1167, 'systems'] = ['muscular system']

    df.at[1177, 'organs'] = ['muscular system']
    df.at[1177, 'systems'] = ['muscular system']

    # end of praduemna code

    # prathamesh
    df.at[5, 'organs'] = ['brain']
    df.at[5, 'systems'] = ['nervous system']

    df.at[7, 'organs'] = ['adrenal gland']
    df.at[7, 'systems'] = ['endocrine system']

    df.at[17, 'organs'] = ['jaw']
    df.at[17, 'systems'] = ['muscular system']

    df.at[32, 'organs'] = ['uterus', 'vagina']
    df.at[32, 'systems'] = ['reproductive system']

    df.at[41, 'organs'] = ['appendix']  # which system??
    df.at[41, 'systems'] = ['digestive system']

    df.at[50, 'organs'] = ['lungs']
    df.at[50, 'systems'] = ['respiratory system']

    df.at[52, 'organs'] = ['brain']
    df.at[52, 'systems'] = ['nervous system']

    df.at[56, 'organs'] = ['heart']
    df.at[56, 'systems'] = ['circulatory system']

    df.at[57, 'organs'] = ['brain']
    df.at[57, 'systems'] = ['nervous system']

    df.at[58, 'organs'] = ['brain']
    df.at[58, 'systems'] = ['nervous system']

    # add breast as an organ of the reproductive system in the val_list
    df.at[60, 'organs'] = ['breasts']
    df.at[60, 'systems'] = ['reproductive system']

    df.at[63, 'organs'] = ['heart']
    df.at[63, 'systems'] = ['circulatory system']

    df.at[64, 'organs'] = ['skin']
    df.at[64, 'systems'] = ['integumentary system']

    df.at[72, 'organs'] = ['adrenal gland']
    df.at[72, 'systems'] = ['endocrine system']

    df.at[74, 'organs'] = ['heart']
    df.at[74, 'systems'] = ['circulatory system']

    df.at[78, 'organs'] = ['stomach', 'small intestine', 'large intestine']
    df.at[78, 'systems'] = ['digestive system']

    # add breast as an organ of the reproductive system in the val_list
    df.at[86, 'organs'] = ['breats']
    df.at[86, 'systems'] = ['reproductive system']

    df.at[87, 'organs'] = ['ligament', 'joints']
    df.at[87, 'systems'] = ['muscular system']

    df.at[88, 'organs'] = ['ligament']
    df.at[88, 'systems'] = ['muscular system']

    df.at[89, 'organs'] = ['human skeleton', 'joint']
    df.at[89, 'systems'] = ['muscular system']

    df.at[91, 'organs'] = ['lungs']
    df.at[91, 'systems'] = ['respiratory system']

    df.at[124, 'organs'] = ['brain']
    df.at[124, 'systems'] = ['nervous system']

    df.at[130, 'organs'] = ['human skeleton', 'joints']
    df.at[130, 'systems'] = ['muscular system']

    df.at[131, 'organs'] = ['brain', 'pinal nerves']
    df.at[131, 'systems'] = ['nervous system']

    df.at[132, 'organs'] = ['brain']
    df.at[132, 'systems'] = ['nervous system']

    df.at[133, 'organs'] = ['brain']
    df.at[133, 'systems'] = ['nervous system']

    df.at[144, 'organs'] = ['large intestine']
    df.at[144, 'systems'] = ['digestive system']

    df.at[147, 'organs'] = ['lungs']
    df.at[147, 'systems'] = ['respiratory system']

    df.at[148, 'organs'] = ['muscular system']
    df.at[148, 'systems'] = ['muscular system']

    df.at[152, 'organs'] = ['heart']
    df.at[152, 'systems'] = ['circulatory system']

    df.at[153, 'organs'] = ['muscular system']
    df.at[153, 'systems'] = ['muscular system']

    df.at[165, 'organs'] = ['brain']
    df.at[165, 'systems'] = ['nervous system']

    df.at[172, 'organs'] = ['skin']
    df.at[172, 'systems'] = ['integumentary system']

    df.at[173, 'organs'] = ['UNKNOWN']  # vomiting vala problem
    df.at[173, 'systems'] = ['digestive system']

    df.at[185, 'organs'] = ['brain']
    df.at[185, 'systems'] = ['nervous system']

    df.at[190, 'organs'] = ['muscular system']
    df.at[190, 'systems'] = ['muscular system']

    df.at[191, 'organs'] = ['ligament']
    df.at[191, 'systems'] = ['muscular system']

    df.at[192, 'organs'] = ['skin']
    df.at[192, 'systems'] = ['integumentary system']

    df.at[200, 'organs'] = ['UNKNOWN']  # watery stools due to indigestion
    df.at[200, 'systems'] = ['digestive system']

    df.at[206, 'organs'] = ['stomach', 'large intestine']
    df.at[206, 'systems'] = ['digestive system']

    df.at[208, 'organs'] = ['heart']
    df.at[208, 'systems'] = ['circulatory system']

    df.at[212, 'organs'] = ['brain']  # sleep deprivation
    df.at[212, 'systems'] = ['nervous system']

    df.at[213, 'organs'] = ['stomach', 'small intestine', 'pancreas']
    df.at[213, 'systems'] = ['digestive system']

    df.at[225, 'organs'] = ['brain']
    df.at[225, 'systems'] = ['nervous system']

    df.at[236, 'organs'] = ['brain']
    df.at[236, 'systems'] = ['nervous system']

    df.at[247, 'organs'] = ['brain']
    df.at[247, 'systems'] = ['nervous system']

    df.at[251, 'organs'] = ['bones']
    df.at[251, 'systems'] = ['muscular system']

    df.at[255, 'organs'] = ['muscular system', 'human skeleton']
    df.at[255, 'systems'] = ['muscular system']

    df.at[256, 'organs'] = ['skin']
    df.at[256, 'systems'] = ['integumentary system']

    df.at[270, 'organs'] = ['breasts']
    df.at[270, 'systems'] = ['reproductive system']

    # end of prathamesh code

    df.at[171, 'organs'] = ['skin']
    df.at[171, 'systems'] = ['integumentary system']

    df.at[390, 'organs'] = ['parathyroid gland']
    df.at[390, 'systems'] = ['endocrine system']

    df.at[987, 'organs'] = ['stomach', 'intestine']
    df.at[987, 'systems'] = ['digestive system']

    # Creating final system column and filling final system by taking mode of systems in system column
    from statistics import mode, StatisticsError

    df['final_system'] = [[] for _ in range(len(df))]

    for i in range(len(df)):
        try:
            var = mode(df['systems'][i])
            df.loc[i, 'final_system'] = var
        except StatisticsError:
            df.loc[i, 'final_system'] = 'ambigious'

    # making spelling corrections
    df.at[757, 'final_system'] = 'circulatory system'
    df.at[758, 'final_system'] = 'circulatory system'
    df.at[722, 'final_system'] = 'reproductive system'

    # Manualling entering final systems to resolve ambiguity

    # praduemna
    df.at[825, 'final_system'] = 'digestive system'
    df.at[832, 'final_system'] = 'nervous system'
    df.at[834, 'final_system'] = 'digestive system'
    df.at[836, 'final_system'] = 'urinary system'
    df.at[840, 'final_system'] = 'respiratory system'
    df.at[843, 'final_system'] = 'integumentary system'
    # Though the systems in 'systems' column are different this disease mainly affects breathing
    df.at[854, 'final_system'] = 'respiratory system'
    df.at[855, 'final_system'] = 'circulatory system'
    df.at[856, 'final_system'] = 'circulatory syste'
    df.at[857, 'final_system'] = 'muscular system'
    df.at[858, 'final_system'] = 'nervous system'
    df.at[862, 'final_system'] = 'reproductive organs'
    df.at[865, 'final_system'] = 'nervous system'
    # Can't say exactly but closest is digestive system
    df.at[866, 'final_system'] = 'digestive system'
    df.at[873, 'final_system'] = 'circulatory system'
    df.at[875, 'final_system'] = 'respiratory system'
    df.at[879, 'final_system'] = 'digestive system'
    df.at[880, 'final_system'] = 'integumentary system'
    df.at[881, 'final_system'] = 'integumentary system'
    df.at[882, 'final_system'] = 'integumentary system'
    df.at[886, 'final_system'] = 'nervous system'
    df.at[887, 'final_system'] = 'nervous system'
    df.at[888, 'final_system'] = 'nervous system'
    df.at[895, 'final_system'] = 'cirulatory system'

    df.at[901, 'final_system'] = 'digestive system'
    df.at[902, 'final_system'] = 'muscular system'
    df.at[904, 'final_system'] = 'integumentary system'
    df.at[909, 'final_system'] = 'respiratory system'
    df.at[913, 'final_system'] = 'integumentary system'
    df.at[914, 'final_system'] = 'human skeleton'
    df.at[916, 'final_system'] = 'nervous system'
    df.at[919, 'final_system'] = 'circulatory system'
    df.at[920, 'final_system'] = 'muscular system'
    df.at[924, 'final_system'] = 'integumentary system'
    df.at[926, 'final_system'] = 'digestive system'
    df.at[930, 'final_system'] = 'reproductive organs'
    df.at[933, 'final_system'] = 'circulatory system'
    df.at[935, 'final_system'] = 'muscular system'
    df.at[936, 'final_system'] = 'integumentary system'
    df.at[937, 'final_system'] = 'muscular system'

    df.at[938, 'final_system'] = 'circulatory system'
    df.at[939, 'final_system'] = 'circulatory system'
    df.at[940, 'final_system'] = 'nervous system'
    df.at[946, 'final_system'] = 'urinary system'
    df.at[948, 'final_system'] = 'circulatory system'
    df.at[955, 'final_system'] = 'integumentary system'
    df.at[961, 'final_system'] = 'nervous system'
    df.at[964, 'final_system'] = 'integumentary system'
    df.at[965, 'final_system'] = 'integumentary system'
    df.at[966, 'final_system'] = 'muscular system'
    df.at[967, 'final_system'] = 'integumentary system'
    df.at[968, 'final_system'] = 'circulatory system'
    df.at[970, 'final_system'] = 'integumentary system'
    df.at[975, 'final_system'] = 'muscular system'
    df.at[978, 'final_system'] = 'reproductive organs'
    df.at[980, 'final_system'] = 'nervous system'
    df.at[981, 'final_system'] = 'circulatory system'
    # Does not affect one particular system
    df.at[983, 'final_system'] = 'digestive system'
    df.at[984, 'final_system'] = 'circulatory system'
    df.at[989, 'final_system'] = 'circulatory system'
    df.at[992, 'final_system'] = 'circulatory system'
    df.at[1003, 'final_system'] = 'nervous system'
    df.at[1004, 'final_system'] = 'circulatory system'
    df.at[1012, 'final_system'] = 'circulatory system'
    df.at[1013, 'final_system'] = 'circulatory system'
    df.at[1016, 'final_system'] = 'circulatory system'
    df.at[1020, 'final_system'] = 'circulatory system'
    df.at[1021, 'final_system'] = 'endocrine system'
    df.at[1023, 'final_system'] = 'endocrine system'
    df.at[1026, 'final_system'] = 'integumentary system'
    df.at[1035, 'final_system'] = 'digestive system'
    df.at[1037, 'final_system'] = 'nervous system'
    df.at[1047, 'final_system'] = 'reproductive organs'
    df.at[1048, 'final_system'] = 'digestive system'
    # Also affects many other systems
    df.at[1049, 'final_system'] = 'integumentary system'
    df.at[1056, 'final_system'] = 'nervous system'
    df.at[1062, 'final_system'] = 'circulatory system'
    df.at[1069, 'final_system'] = 'circulatory system'
    df.at[1070, 'final_system'] = 'digestive system'
    df.at[1075, 'final_system'] = 'reproductive organs'
    df.at[1077, 'final_system'] = 'endocrine system'
    df.at[1079, 'final_system'] = 'endocrine system'
    df.at[1088, 'final_system'] = 'endocrine system'
    df.at[1089, 'final_system'] = 'integumentary system'
    df.at[1097, 'final_system'] = 'urinary system'

    df.at[1116, 'final_system'] = 'respiratory system'
    df.at[1118, 'final_system'] = 'integumentary system'
    df.at[1122, 'final_system'] = 'integumentary system'
    df.at[1125, 'final_system'] = 'circulatory system'
    df.at[1126, 'final_system'] = 'nervous system'
    df.at[1127, 'final_system'] = 'nervous system'
    df.at[1130, 'final_system'] = 'circulatory system'
    df.at[1131, 'final_system'] = 'circulatory system'
    df.at[1132, 'final_system'] = 'nervous system'
    df.at[1139, 'final_system'] = 'circulatory system'
    df.at[1140, 'final_system'] = 'integumentary system'
    df.at[1141, 'final_system'] = 'nervous system'
    df.at[1144, 'final_system'] = 'circulatory system'
    df.at[1151, 'final_system'] = 'muscular system'
    df.at[1155, 'final_system'] = 'nervous system'
    df.at[1156, 'final_system'] = 'integumentary system'
    df.at[1158, 'final_system'] = 'digestive system'
    df.at[1165, 'final_system'] = 'circulatory system'
    df.at[1166, 'final_system'] = 'integumentary system'
    df.at[1173, 'final_system'] = 'nervous system'
    df.at[1181, 'final_system'] = 'nervous system'

    # corrections:

    df.at[895, 'final_system'] = 'circulatory system'

    df.at[862, 'final_system'] = 'reproductive system'
    df.at[930, 'final_system'] = 'reproductive system'
    df.at[978, 'final_system'] = 'reproductive system'
    df.at[1047, 'final_system'] = 'reproductive system'
    df.at[1075, 'final_system'] = 'reproductive system'

    df.at[856, 'final_system'] = 'circulatory system'

    # prathamesh
    df.at[8, 'final_system'] = 'nervous system'
    df.at[10, 'final_system'] = 'nervous system'
    df.at[12, 'final_system'] = 'nervous system'
    df.at[13, 'final_system'] = 'digestive system'
    df.at[15, 'final_system'] = 'digestive system'
    df.at[19, 'final_system'] = 'nervous system'
    df.at[21, 'final_system'] = 'digestive system'
    df.at[22, 'final_system'] = 'muscular system'
    df.at[23, 'final_system'] = 'digestive system'
    df.at[25, 'final_system'] = 'nervous system'
    df.at[26, 'final_system'] = 'respiratory system'
    df.at[28, 'final_system'] = 'circulatory system'
    df.at[29, 'final_system'] = 'nervous system'
    df.at[30, 'final_system'] = 'circulatory system'
    df.at[34, 'final_system'] = 'respiratory system'
    df.at[35, 'final_system'] = 'circulatory system'
    df.at[40, 'final_system'] = 'circulatory system'
    df.at[48, 'final_system'] = 'respiratory system'
    df.at[49, 'final_system'] = 'respiratory system'
    df.at[62, 'final_system'] = 'digestive system'
    df.at[71, 'final_system'] = 'nervous system'
    df.at[75, 'final_system'] = 'nervous system'
    df.at[79, 'final_system'] = 'nervous system'
    df.at[82, 'final_system'] = 'nervous system'
    df.at[92, 'final_system'] = 'nervous system'
    df.at[94, 'final_system'] = 'integumentary system'
    df.at[99, 'final_system'] = 'digestive system'
    df.at[100, 'final_system'] = 'urinary system'
    df.at[101, 'final_system'] = 'nervous system'
    df.at[106, 'final_system'] = 'circulatory system'
    df.at[111, 'final_system'] = 'circulatory system'
    df.at[126, 'final_system'] = 'respiratory system'
    df.at[129, 'final_system'] = 'digestive system'
    df.at[134, 'final_system'] = 'digestive system'
    df.at[136, 'final_system'] = 'digestive system'
    df.at[137, 'final_system'] = 'circulatory system'
    df.at[138, 'final_system'] = 'nervous system'
    df.at[141, 'final_system'] = 'nervous system'
    df.at[145, 'final_system'] = 'nervous system'
    df.at[156, 'final_system'] = 'digestive system'
    df.at[163, 'final_system'] = 'respiratory system'
    df.at[168, 'final_system'] = 'nervous system'
    df.at[169, 'final_system'] = 'integumentary system'
    df.at[178, 'final_system'] = 'digestive system'
    df.at[181, 'final_system'] = 'nervous system'
    df.at[183, 'final_system'] = 'nervous system'
    df.at[186, 'final_system'] = 'integumentary system'
    df.at[187, 'final_system'] = 'integumentary system'
    df.at[195, 'final_system'] = 'nervous system'
    df.at[196, 'final_system'] = 'circulatory system'
    df.at[203, 'final_system'] = 'respiratory system'
    df.at[204, 'final_system'] = 'muscular system'
    df.at[209, 'final_system'] = 'nervous system'
    df.at[210, 'final_system'] = 'integumentary system'
    df.at[215, 'final_system'] = 'nervous system'
    df.at[216, 'final_system'] = 'integumentary system'
    df.at[218, 'final_system'] = 'nervous system'
    df.at[224, 'final_system'] = 'nervous system'
    df.at[227, 'final_system'] = 'circulatory system'
    df.at[232, 'final_system'] = 'muscular system'
    df.at[239, 'final_system'] = 'circulatory system'
    df.at[242, 'final_system'] = 'reproductive system'
    df.at[248, 'final_system'] = 'integumentary system'
    df.at[250, 'final_system'] = 'nervous system'
    df.at[265, 'final_system'] = 'nervous system'
    df.at[268, 'final_system'] = 'digestive system'
    df.at[272, 'final_system'] = 'nervous system'
    df.at[276, 'final_system'] = 'digestive system'
    df.at[280, 'final_system'] = 'nervous system'
    df.at[284, 'final_system'] = 'muscular system'
    df.at[289, 'final_system'] = 'endocrine system'
    df.at[302, 'final_system'] = 'digestive system'
    df.at[303, 'final_system'] = 'nervous system'
    df.at[307, 'final_system'] = 'digestive system'
    df.at[311, 'final_system'] = 'urinary system'
    df.at[312, 'final_system'] = 'nervous system'
    df.at[314, 'final_system'] = 'muscular system'
    df.at[316, 'final_system'] = 'endocirne system'
    df.at[317, 'final_system'] = 'urinary system'
    df.at[319, 'final_system'] = 'endocrine system'
    df.at[322, 'final_system'] = 'integumentary system'
    df.at[328, 'final_system'] = 'nervous system'
    df.at[338, 'final_system'] = 'circulatory system'
    df.at[341, 'final_system'] = 'circulatory system'
    df.at[343, 'final_system'] = 'circulatory system'
    df.at[344, 'final_system'] = 'circulatory system'
    df.at[348, 'final_system'] = 'muscular system'
    df.at[349, 'final_system'] = 'digestive system'
    df.at[350, 'final_system'] = 'digestive system'
    df.at[351, 'final_system'] = 'digestive system'
    df.at[352, 'final_system'] = 'digestive system'
    df.at[354, 'final_system'] = 'digestive system'
    df.at[365, 'final_system'] = 'integumentary system'
    df.at[373, 'final_system'] = 'nervous system'
    df.at[374, 'final_system'] = 'urinary system'
    df.at[376, 'final_system'] = 'circulatory system'
    df.at[379, 'final_system'] = 'circulatory system'
    df.at[382, 'final_system'] = 'urinary system'
    df.at[383, 'final_system'] = 'urinary system'
    df.at[387, 'final_system'] = 'nervous system'
    df.at[388, 'final_system'] = 'digestive system'
    df.at[389, 'final_system'] = 'nervous system'
    df.at[391, 'final_system'] = 'circulatory system'
    df.at[397, 'final_system'] = 'digestive system'
    df.at[399, 'final_system'] = 'integumentary system'

    # gargi

    df.at[403, 'final_system'] = 'digestive system'
    df.at[404, 'final_system'] = 'digestive system'
    df.at[411, 'final_system'] = 'respiratory system'
    df.at[420, 'final_system'] = 'reproductive system'
    df.at[423, 'final_system'] = 'digestive system'
    df.at[425, 'final_system'] = 'digestive system'
    df.at[427, 'final_system'] = 'circulatory system'
    df.at[431, 'final_system'] = 'muscular system'
    df.at[437, 'final_system'] = 'digestive system'
    df.at[452, 'final_system'] = 'respiratory system'
    df.at[453, 'final_system'] = 'digestive system'
    df.at[458, 'final_system'] = 'circulatory system'
    df.at[465, 'final_system'] = 'integumentary system'
    df.at[466, 'final_system'] = 'reproductive system'
    # type of cancer ## should suggest a oncologist
    df.at[467, 'final_system'] = 'digestive system'
    df.at[468, 'final_system'] = 'integumentary system'
    df.at[470, 'final_system'] = 'nervous system'
    df.at[471, 'final_system'] = 'digestive system'
    df.at[472, 'final_system'] = 'digestive system'
    df.at[473, 'final_system'] = 'digestive system'
    df.at[474, 'final_system'] = 'digestive system'
    df.at[475, 'final_system'] = 'circulatory system'
    df.at[477, 'final_system'] = 'circulatory system'
    df.at[480, 'final_system'] = 'circulatory system'  # not sure
    df.at[484, 'final_system'] = 'respiratory system'
    df.at[488, 'final_system'] = 'muscular system'
    df.at[494, 'final_system'] = 'nervous system'
    df.at[496, 'final_system'] = 'reproductive system'
    df.at[507, 'final_system'] = 'digestive system'
    df.at[510, 'final_system'] = 'digestive system'
    df.at[517, 'final_system'] = 'nervous system'
    df.at[521, 'final_system'] = 'respiratory system'
    df.at[524, 'final_system'] = 'nervous system'
    df.at[530, 'final_system'] = 'digestive system'
    df.at[535, 'final_system'] = 'circulatory system'
    df.at[540, 'final_system'] = 'digestive system'
    df.at[542, 'final_system'] = 'digestive system'
    df.at[544, 'final_system'] = 'integumentary system'
    df.at[546, 'final_system'] = 'circulatory system'
    df.at[547, 'final_system'] = 'integumentary system'
    df.at[553, 'final_system'] = 'muscular system'
    df.at[559, 'final_system'] = 'circulatory system'
    df.at[562, 'final_system'] = 'muscular system'
    df.at[567, 'final_system'] = 'respiratory system'
    df.at[569, 'final_system'] = 'muscular system'
    df.at[573, 'final_system'] = 'integumentary system'
    df.at[582, 'final_system'] = 'integumentary system'
    df.at[583, 'final_system'] = 'respiratory system'
    df.at[585, 'final_system'] = 'nervous system'
    df.at[586, 'final_system'] = 'urinary system'
    df.at[587, 'final_system'] = 'digestive system'
    df.at[590, 'final_system'] = 'digestive system'
    # not sure genetic disorder affects multiple systems
    df.at[591, 'final_system'] = 'circulatory system'
    df.at[595, 'final_system'] = 'digestive system'
    df.at[596, 'final_system'] = 'respiratory system'
    df.at[597, 'final_system'] = 'nervous system'
    df.at[598, 'final_system'] = 'nervous system'
    df.at[606, 'final_system'] = 'respiratory system'
    df.at[613, 'final_system'] = 'muscular system'
    df.at[615, 'final_system'] = 'muscular system'
    df.at[628, 'final_system'] = 'reproductive system'
    df.at[632, 'final_system'] = 'digestive system'
    df.at[636, 'final_system'] = 'nervous system'
    df.at[645, 'final_system'] = 'circulatory system'
    df.at[650, 'final_system'] = 'urinary system'
    df.at[652, 'final_system'] = 'integumentary system'
    df.at[653, 'final_system'] = 'integumentary system'
    df.at[655, 'final_system'] = 'circulatory system'
    df.at[656, 'final_system'] = 'reproductive system'
    df.at[671, 'final_system'] = 'respiratory system'
    df.at[676, 'final_system'] = 'urinary system'
    df.at[678, 'final_system'] = 'nervous system'
    df.at[681, 'final_system'] = 'digestive system'
    df.at[682, 'final_system'] = 'endocrine system'
    df.at[683, 'final_system'] = 'nervous system'
    df.at[685, 'final_system'] = 'nervous system'  # genetical
    df.at[691, 'final_system'] = 'respiratory system'
    df.at[692, 'final_system'] = 'respiratory system'
    df.at[695, 'final_system'] = 'nervous system'
    df.at[697, 'final_system'] = 'urinary system'
    df.at[707, 'final_system'] = 'circulatory system'
    df.at[723, 'final_system'] = 'reproductive system'
    df.at[724, 'final_system'] = 'reproductive system'
    df.at[731, 'final_system'] = 'reproductive system'
    df.at[734, 'final_system'] = 'digestive system'
    df.at[736, 'final_system'] = 'endocrine system'  # immune system
    df.at[741, 'final_system'] = 'reproductive system'  # genetic disorder
    df.at[742, 'final_system'] = 'endocrine system'
    df.at[745, 'final_system'] = 'reproductive system'
    df.at[746, 'final_system'] = 'urinary system'
    df.at[752, 'final_system'] = 'integumentary system'
    df.at[760, 'final_system'] = 'respiratory system'
    df.at[762, 'final_system'] = 'circulatory system'
    df.at[766, 'final_system'] = 'respiratory system'
    df.at[767, 'final_system'] = 'muscular system'  # rabies
    df.at[772, 'final_system'] = 'integumentary system'
    df.at[781, 'final_system'] = 'reproductive system'
    df.at[800, 'final_system'] = 'muscular system'
    df.at[806, 'final_system'] = 'integumentary system'
    df.at[807, 'final_system'] = 'integumentary system'
    df.at[812, 'final_system'] = 'respiratory system'
    df.at[813, 'final_system'] = 'respiratory system'
    df.at[817, 'final_system'] = 'circulatory system'
    df.at[821, 'final_system'] = 'respiratory system'
    df.at[822, 'final_system'] = 'muscular system'  # cancer in joints

    # correction
    df.at[316, 'final_system'] = 'endocrine system'

    df.at[914, 'final_system'] = 'muscular system'

    # end of manual entries
    #####################################################################################
    test_df = df.copy()

    test_df['input'] = ''

    for i in range(len(test_df)):
        test_df.loc[i, 'input'] = test_df.loc[i, 'symptoms']+test_df.loc[i,
                                                                         'causes']+test_df.loc[i, 'risk_factor']+test_df.loc[i, 'overview']

    # Training model on entire dataset
    X = test_df['input'].values

    Y = test_df['final_system'].values

    # Converting text into tfidf vector

    from sklearn.feature_extraction.text import CountVectorizer

    # saving CountVectorizer object to use it later during prediction
    count_vect = CountVectorizer(decode_error="replace").fit(X)

    file_name_for_vector = 'feature_final_system.pkl'

    pickle.dump(count_vect.vocabulary_, open(file_name_for_vector, 'wb'))

    X_counts = count_vect.transform(X)

    # TF-IDF
    from sklearn.feature_extraction.text import TfidfTransformer
    tfidf_transformer = TfidfTransformer().fit(X_counts)

    X_tfidf = tfidf_transformer.transform(X_counts)

    classifier = RandomForestClassifier(n_estimators=1000,
                                        bootstrap=True,
                                        max_features='sqrt').fit(X_tfidf, Y)

    filename = 'final_system_model.sav'

    pickle.dump(classifier, open(filename, 'wb'))

    # Saving the dataframe for later use
    test_df.to_pickle('dataframe.pkl')


def make_prediction(input1):
    # input1 is a string of natural language

    # loading saved dataframe from input
    df = pd.read_pickle('dataframe.pkl')

    # because TfidfTransformer.fit() method takes input of np array object, converting the string input to that format
    input_arr = np.array([input1])
    ser = pd.Series(input_arr)
    new_input = ser.values

    transformer = TfidfTransformer()

    file_name_for_vector = 'feature_final_system.pkl'

    # Loading saved model of CountVectorizer

    loaded_vec = CountVectorizer(
        decode_error="replace", vocabulary=pickle.load(open(file_name_for_vector, "rb")))

    input_tfidf = transformer.fit_transform(loaded_vec.transform(new_input))

    predicted_system = classifier.predict(input_tfidf)[0]
    predicted_system_probability = classifier.predict_proba(input_tfidf).max()

    result_list = df[df['final_system'] == predicted_system].symptoms

    # fuzzywuzzy is a library for string similarity score
    from fuzzywuzzy import fuzz
    from fuzzywuzzy import process

    ''' 
        For every disease belonging to the predicted system, comparing the string of
        symptoms with the input string and keeping a tract of the index and scores in 
        a list named 'scores'
    '''
    scores = []
    for index in result_list.index:
        scores.append((fuzz.partial_ratio(input1, result_list[index]), index))

    # Sorting the list 'scores' in descending order
    # Since it is a list of tuples, we cannot sort it using the normal sort function
    scores.sort(key=lambda x: x[0], reverse=True)

    predicted_disease = df.loc[scores[0][1], 'name']

    return predicted_system, predicted_system_probability, predicted_disease


'''For testing:'''
# x = 'My stomach is paining. I m facing problem while swallowing the food from the throat. Also my nose is blocked'
# system, probability, disease = make_prediction(x)
# print('The predicted system is: {} , its probability is: {} and the probable disease is: {}'.format(
#     system, probability, disease))
