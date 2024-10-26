from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import pandas as pd
import re

data = pd.read_csv("magaza_yorumlari_duygu_analizi.csv", encoding="utf-16")
data = data.dropna()

data["Durum"] = data["Durum"].map({"Olumlu": 0, "Tarafsız": 1, "Olumsuz": 2})

with open("stop.txt", "r", encoding="utf-8") as file:
    stopWords = set(file.read().split())

def harfDegistir(cumle):
    cumle = re.sub("[^a-zA-ZğĞüÜşŞıİöÖçÇ]", " ", cumle)
    cumle = cumle.lower()
    cumle = cumle.split()
    cumle = [word for word in cumle if word not in stopWords]
    return " ".join(cumle)

sonGorus = [harfDegistir(review) for review in data["Görüş"]]

maxFea = 2500
cv = CountVectorizer(max_features=maxFea)
spaceMatrix = cv.fit_transform(sonGorus).toarray()

x = spaceMatrix
y = data["Durum"].values
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=33)


rf = RandomForestClassifier(n_estimators=100, random_state=45)
rf.fit(xTrain, yTrain)

st.title("Deneme")

user_input = st.text_area("", "")

if st.button("hesapla"):
    if user_input:
        processed_input = harfDegistir(user_input)
        input_vector = cv.transform([processed_input]).toarray()
        
        prediction = rf.predict(input_vector)
        labels = {0: "Olumlu", 1: "Tarafsız", 2: "Olumsuz"}
        st.write(f"Sonuç: {labels[prediction[0]]}")
    else:
        print()
