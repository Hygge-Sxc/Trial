import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
from lime import lime_tabular
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import streamlit.components.v1 as components
from lime.lime_tabular import LimeTabularExplainer

# 设置页面配置
st.set_page_config(page_title="Heart Disease Prediction App", layout="wide")

# 应用标题和描述
st.title("Heart Disease Prediction App")
st.markdown("""
This app predicts the probability of having heart disease based on user input features.
You can input the required information, and the model will provide a prediction along with explanations using SHAP and LIME.
""")

# 加载训练好的模型
model = joblib.load('trained_random_forest_model.pkl')

# 加载X_train数据
X_train = pd.read_csv('X_train_saved.csv')

# 用户输入特征
def user_input_features():
    st.sidebar.header('Input Features')
    
    # 数值型特征
    age = st.sidebar.number_input('Age', min_value=1, max_value=120, value=50)
    trestbps = st.sidebar.number_input('Resting Blood Pressure (trestbps)', min_value=50, max_value=250, value=120)
    chol = st.sidebar.number_input('Serum Cholestoral in mg/dl (chol)', min_value=100, max_value=600, value=200)
    thalach = st.sidebar.number_input('Maximum Heart Rate achieved (thalach)', min_value=60, max_value=250, value=150)
    oldpeak = st.sidebar.number_input('ST depression induced by exercise relative to rest (oldpeak)', min_value=0.0, max_value=10.0, value=1.0)
    
    # 分类特征
    sex = st.sidebar.selectbox('Sex', ('Male', 'Female'))
    cp = st.sidebar.selectbox('Chest Pain Type (cp)', ('Typical angina', 'Atypical angina', 'Non-anginal pain', 'Asymptomatic'))
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', ('Yes', 'No'))
    restecg = st.sidebar.selectbox('Resting Electrocardiographic Results (restecg)', 
                                   ('Normal', 'ST-T wave abnormality', 'Left ventricular hypertrophy'))
    exang = st.sidebar.selectbox('Exercise Induced Angina (exang)', ('Yes', 'No'))
    slope = st.sidebar.selectbox('Slope of the peak exercise ST segment (slope)', 
                                 ('Upsloping', 'Flat', 'Downsloping'))
    ca = st.sidebar.selectbox('Number of major vessels colored by flourosopy (ca)', 
                              ('0', '1', '2', '3'))
    thal = st.sidebar.selectbox('Thalassemia (thal)', 
                                ('Normal', 'Fixed defect', 'Reversible defect'))
    
    # 创建输入数据字典
    data = {
        'age': age,
        'trestbps': trestbps,
        'chol': chol,
        'thalach': thalach,
        'oldpeak': oldpeak
    }
    
    # 处理分类特征并进行独热编码
    # Sex
    data['sex_1'] = 1 if sex == 'Male' else 0
    
    # Chest Pain Type (cp)
    cp_dict = {'Typical angina': 1, 'Atypical angina': 2, 'Non-anginal pain': 3, 'Asymptomatic': 4}
    cp_val = cp_dict[cp]
    data['cp_2'] = 1 if cp_val == 2 else 0
    data['cp_3'] = 1 if cp_val == 3 else 0
    data['cp_4'] = 1 if cp_val == 4 else 0
    
    # Fasting Blood Sugar (fbs)
    data['fbs_1'] = 1 if fbs == 'Yes' else 0
    
    # Resting Electrocardiographic Results (restecg)
    restecg_dict = {'Normal': 0, 'ST-T wave abnormality': 1, 'Left ventricular hypertrophy': 2}
    restecg_val = restecg_dict[restecg]
    data['restecg_1'] = 1 if restecg_val == 1 else 0
    data['restecg_2'] = 1 if restecg_val == 2 else 0
    
    # Exercise Induced Angina (exang)
    data['exang_1'] = 1 if exang == 'Yes' else 0
    
    # Slope of ST segment
    slope_dict = {'Upsloping': 1, 'Flat': 2, 'Downsloping': 3}
    slope_val = slope_dict[slope]
    data['slope_2'] = 1 if slope_val == 2 else 0
    data['slope_3'] = 1 if slope_val == 3 else 0
    
    # Number of major vessels (ca)
    data['ca_1'] = 1 if ca == '1' else 0
    data['ca_2'] = 1 if ca == '2' else 0
    data['ca_3'] = 1 if ca == '3' else 0
    
    # Thalassemia (thal)
    thal_dict = {'Normal': 3, 'Fixed defect': 6, 'Reversible defect': 7}
    thal_val = thal_dict[thal]
    data['thal_6'] = 1 if thal_val == 6 else 0
    data['thal_7'] = 1 if thal_val == 7 else 0
    
    # 创建DataFrame
    input_df = pd.DataFrame(data, index=[0])
    
    # 获取模型的特征名称
    if hasattr(model, 'feature_names_in_'):
        model_features = model.feature_names_in_
    else:
        raise AttributeError("模型缺少 'feature_names_in_' 属性。请确保模型包含特征名称或进行相应修改。")
    
    # 添加缺失的特征并按模型特征顺序排列
    for feature in model_features:
        if feature not in input_df.columns:
            input_df[feature] = 0
    
    input_df = input_df[model_features]
    
    return input_df

# 收集用户输入
input_df = user_input_features()

# 显示用户输入
st.subheader('User Input Features')
st.write(input_df)

# 模型预测
prediction = model.predict(input_df)[0]
prediction_proba = model.predict_proba(input_df)[0][1]

# 显示预测结果
st.subheader('Prediction')
if prediction == 1:
    st.error(f'The model predicts that the patient **has heart disease** with a probability of {prediction_proba:.2f}.')
else:
    st.success(f'The model predicts that the patient **does not have heart disease** with a probability of {1 - prediction_proba:.2f}.')

# 显示预测概率
st.subheader('Prediction Probability')
st.write(f'**Probability of having heart disease:** {prediction_proba:.2f}')
st.write(f'**Probability of not having heart disease:** {1 - prediction_proba:.2f}')

# -------------------- SHAP Explanation --------------------
st.subheader('SHAP Explanation')

# 初始化SHAP解释器
explainer = shap.TreeExplainer(model)

# 计算SHAP值
shap_values = explainer.shap_values(input_df)

# SHAP force plot
st.markdown('**SHAP Force Plot**')
force_plot = shap.force_plot(
    explainer.expected_value[1],
    shap_values[1],
    input_df,
    matplotlib=True
)
st.pyplot(force_plot)

# SHAP waterfall plot
st.markdown('**SHAP Waterfall Plot**')
fig, ax = plt.subplots(figsize=(10, 6))
shap.plots._waterfall.waterfall_legacy(explainer.expected_value[1], shap_values[1][0], feature_names=input_df.columns)
st.pyplot(fig)

# -------------------- LIME Explanation --------------------
st.subheader('LIME Explanation')

# 初始化LIME解释器
lime_explainer = LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X_train.columns,
    class_names=['Not sick', 'Sick'],  # 调整类别名称以匹配您的分类任务
    mode='classification'
)

# 生成LIME解释
lime_explanation = lime_explainer.explain_instance(
    data_row=input_df.iloc[0],
    predict_fn=model.predict_proba,
    num_features=10
)

# 显示LIME解释（HTML格式）
st.markdown('**LIME Explanation (Interactive)**')
components.html(lime_explanation.as_html(), height=300, scrolling=True)
