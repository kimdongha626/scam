import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE

# 페이지 설정
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("💳 Credit Card Fraud Detection Dashboard")

# 데이터 업로드
uploaded_file = st.sidebar.file_uploader("CSV 데이터 업로드", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("📊 데이터 미리보기")
    st.write(data.head())

    # Fraud 비율 시각화 (Streamlit 내장 차트)
    st.subheader("Fraud 비율")
    fraud_counts = data["Class"].value_counts()
    st.bar_chart(fraud_counts)

    # 특징과 라벨 분리
    X = data.drop("Class", axis=1)
    y = data["Class"]

    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # SMOTE 적용
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # 모델 리스트
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(probability=True),
        "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(64,32), max_iter=500)
    }

    results = {}
    st.subheader("📈 모델 성능 비교")

    for name, model in models.items():
        model.fit(X_train_res, y_train_res)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1]

        f1 = f1_score(y_test, y_pred)
        results[name] = f1

        st.markdown(f"### {name}")
        st.text(classification_report(y_test, y_pred))

        # ROC Curve 데이터 출력 (Streamlit line_chart)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_df = pd.DataFrame({"FPR": fpr, "TPR": tpr})
        st.line_chart(roc_df.set_index("FPR"))

    # 모델별 F1 Score 비교 (Streamlit bar_chart)
    st.subheader("📌 모델별 F1 Score 비교")
    f1_df = pd.DataFrame.from_dict(results, orient="index", columns=["F1 Score"])
    st.bar_chart(f1_df)

    # 사용자 입력 예측
    st.subheader("🔍 새로운 거래 예측")
    input_data = {}
    for col in X.columns:
        val = st.number_input(f"{col} 값 입력", value=float(X[col].mean()))
        input_data[col] = val

    input_df = pd.DataFrame([input_data])

    # 사용자가 모델 선택
    model_choice = st.selectbox("모델 선택", list(models.keys()))
    chosen_model = models[model_choice]
    chosen_model.fit(X_train_res, y_train_res)
    prediction = chosen_model.predict(input_df)[0]
    prob = chosen_model.predict_proba(input_df)[0][1]

    st.markdown("### 결과")
    if prediction == 1:
        st.error(f"🚨 사기 거래 가능성 높음 (확률: {prob:.2f})")
    else:
        st.success(f"✅ 정상 거래 (사기 확률: {prob:.2f})")
else:
    st.info("📥 CSV 파일을 업로드하면 분석이 시작됩니다.")
