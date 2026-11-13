from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow import keras

ROOT = Path(__file__).resolve().parent
FEATURES = ['Edad', 'Genero', 'Poliuria', 'Polidipsia', 'Perdida_repentina_peso',
            'Debilidad', 'Polifagia', 'Candidiasis_genital', 'Vision_borrosa',
            'Picazon', 'Irritabilidad', 'Cicatrizacion_lenta', 'Paresia_parcial',
            'Rigidez_muscular', 'Alopecia', 'Obesidad']
LABELS = {'Poliuria':'Poliuria', 'Polidipsia':'Polidipsia',
          'Perdida_repentina_peso':'Pérdida repentina de peso', 'Debilidad':'Debilidad',
          'Polifagia':'Polifagia', 'Candidiasis_genital':'Candidiasis genital',
          'Vision_borrosa':'Visión borrosa', 'Picazon':'Picazón',
          'Irritabilidad':'Irritabilidad', 'Cicatrizacion_lenta':'Cicatrización lenta',
          'Paresia_parcial':'Paresia parcial', 'Rigidez_muscular':'Rigidez muscular',
          'Alopecia':'Alopecia', 'Obesidad':'Obesidad'}

st.set_page_config(page_title='Predictor de diabetes - RNA', page_icon='🩺')
st.title('Predictor de diabetes — Red neuronal artificial')
st.caption('Uso académico. Resultado orientativo; consulte a un profesional de salud.')

@st.cache_resource
def cargar():
    return keras.models.load_model(ROOT / 'modelo_rna.keras'), joblib.load(ROOT / 'scaler.pkl')

if not (ROOT / 'modelo_rna.keras').exists() or not (ROOT / 'scaler.pkl').exists():
    st.error('Faltan modelo_rna.keras y/o scaler.pkl. Ejecute el notebook de entrenamiento y coloque ambos archivos junto a app.py.')
    st.stop()

modelo, scaler = cargar()
edad = st.number_input('Edad', min_value=16, max_value=90, value=45, step=1)
genero = st.selectbox('Género', ['Masculino (1)', 'Femenino (0)'])
values = {'Edad': edad, 'Genero': 1 if genero == 'Masculino (1)' else 0}
cols = st.columns(2)
for i, key in enumerate(FEATURES[2:]):
    with cols[i % 2]:
        values[key] = int(st.checkbox(LABELS[key]))
if st.button('Predecir', type='primary'):
    x = pd.DataFrame([[values[k] for k in FEATURES]], columns=FEATURES)
    prob = float(modelo.predict(scaler.transform(x), verbose=0)[0, 0])
    if prob >= 0.5:
        st.error(f'Clasificación: positivo — puntuación del modelo: {prob:.1%}')
    else:
        st.success(f'Clasificación: negativo — puntuación de positivo: {prob:.1%}')
    st.progress(float(np.clip(prob, 0, 1)))
    st.caption('La puntuación no sustituye un diagnóstico y puede no estar calibrada.')
