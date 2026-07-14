import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
import os
from dotenv import load_dotenv

# Configurações do layout da página
st.set_page_config(
    page_title="IA Diagnóstico - Saúde da Mulher",
    page_icon="👩‍⚕️",
    layout="wide"
)

# Injetar CSS customizado para garantir os detalhes azul turquesa
st.markdown("""
    <style>
    .stButton>button {
        background-color: #00CED1 !important;
        color: #0F172A !important;
        font-weight: 600;
        border-radius: 4px;
        padding: 10px 24px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #48D1CC !important;
        box-shadow: 0 4px 6px rgba(0,206,209,0.2);
    }
    h1, h2, h3, h4, h5 {
        color: #48D1CC !important;
        font-weight: 600;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .laudo-box {
        background-color: #1E293B;
        border-left: 4px solid #00CED1;
        border-top: 1px solid #334155;
        border-right: 1px solid #334155;
        border-bottom: 1px solid #334155;
        padding: 24px;
        border-radius: 6px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        font-size: 1.05em;
        line-height: 1.6;
        color: #E2E8F0;
    }
    [data-testid="stAppViewContainer"] {
        background-color: #0F172A;
    }
    [data-testid="stHeader"] {
        background: transparent;
    }
    div[data-testid="stMetricValue"] {
        color: #F8FAFC !important;
    }
    div[data-testid="stMetric"] {
        background-color: #1E293B;
        border: 1px solid #334155;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    </style>
""", unsafe_allow_html=True)

# Importando a LLM customizada
sys.path.append(os.path.abspath('src'))
from llm_explainer import MedicalLLMExplainer

# Carregar variáveis de ambiente (Chave Groq)
load_dotenv()

# Decorator para não recarregar o modelo do disco toda vez que apertar um botão
@st.cache_resource
def load_assets():
    model = joblib.load('outputs/models/best_rf_model_ga.pkl')
    scaler = joblib.load('outputs/models/scaler_final.pkl')
    
    # Carregando uma amostra do dataset real para pegar o nome das features
    X_train_raw = pd.read_csv('data/processed/X_train_raw.csv')
    feature_names = X_train_raw.columns.tolist()
    
    return model, scaler, feature_names

model, scaler, feature_names = load_assets()

# --- HEADER DASHBOARD ---
col_logo, col_title = st.columns([1, 8])
with col_title:
    st.title("🩺 Medical Dashboard")
    st.markdown("**Módulo Oncologia:** Sistema Preditivo e Analise Generativa de Rastreio (Fase 2)")
st.divider()

# --- CORPO PRINCIPAL ---
col_form, spacer, col_results = st.columns([0.8, 0.1, 2])

with col_form:
    st.markdown("### 📥 Análise Clínica")
    st.caption("Insira manualmente as 5 métricas primárias isoladas ou carregue os meta-dados do equipamento.")

    
    # Preenche input com dados dinamicamente com base nas 5 principais variáveis
    # Foco mantido no Top 5 (SHAP features)
    top_features_visiveis = ['radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'concavity_worst']
    
    user_inputs = {}
    for col_name in top_features_visiveis:
        user_inputs[col_name] = st.number_input(
            f"{col_name.replace('_', ' ').title()}", 
            value=0.00, 
            format="%.4f"
        )
        
    st.info("Parâmetros covariantes serão ajustados pelas medianas da macro população.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button("Executar Algoritmo e Gerar Laudo", use_container_width=True)

with col_results:
    st.markdown("### 🔬 Resultados e Explicabilidade")
    
    if analyze_btn:
        with st.spinner("O Algoritmo Genético está processando os dados e a IA está formatando o laudo..."):
            
            # --- PREPARAÇÃO DOS DADOS ---
            # Matriz de preditores com mediana da base de treino (evita enviesamento)
            medians = pd.read_csv('data/processed/X_train_raw.csv').median()
            input_vector = pd.DataFrame([medians], columns=feature_names)
            for k, v in user_inputs.items():
                input_vector.at[0, k] = v
                
            # Dimensionamento do vetor utilizando scaler
            input_scaled = scaler.transform(input_vector)
            
            # --- MODELO RANDOM FOREST PREDITO ---
            prediction = model.predict(input_scaled)[0]
            proba = model.predict_proba(input_scaled)[0]
            confidence = proba[prediction] * 100
            
            # --- FEATURES IMPORTANCES PARA O PROMPT ---
            importancias = model.feature_importances_
            top_indices = np.argsort(importancias)[::-1][:3]
            top_features_names = input_vector.columns[top_indices].tolist()
            
            # Formata os nomes que mais pesaram
            features_texto = ", ".join([f"{f} = {input_vector.iloc[0][f]:.2f}" for f in top_features_names])
            
            # --- LAUDO LLM ---
            try:
                explainer = MedicalLLMExplainer()
                laudo = explainer.generate_explanation(
                    prediction_class=prediction,
                    confidence=confidence,
                    top_features=features_texto
                )
                
                # --- EXIBIÇÃO VISUAL (UI Aprimorada) ---
                # Bloco de Alerta e Defesa Técnica
                if prediction == 1:
                    st.error(f"🔴 **Alerta Clínico: Previsão do Algoritmo indicou Risco de Malignidade**")
                else:
                    st.success(f"🟢 **Risco Clínico Baixo: Algoritmo indica Morfologia Benigna**")
                
                # Métricas separadas e formatadas
                m1, m2 = st.columns(2)
                with m1:
                    st.metric(label="Confiança do Algoritmo", value=f"{confidence:.2f}%")
                with m2:
                    st.markdown("""
                    <div style="font-size:0.85em; color:#94A3B8; padding-top:10px; border-left:3px solid #00CED1; padding-left:10px;">
                        <i>Nota Técnica: Modelo tunado via Algoritmo Genético focado em Alta Sensibilidade (Recall Máximo). 
                        Níveis de confiança moderados em achados malignos indicam detecção precoce celular, evitando Falsos Negativos.</i>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Exibindo métricas de peso
                st.markdown("##### 🧬 Principais Influenciadores Celulares")
                st.info(f"O motor genético identificou desvios axiais prioritários nas seguintes features: `{top_features_names[0]}` | `{top_features_names[1]}` | `{top_features_names[2]}`")
                
                # Texto Final do LLM
                st.markdown("##### 📄 Relatório de Explicabilidade Gerado via Inteligência Artificial")
                st.markdown(f'<div class="laudo-box">{laudo}</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Erro na conexão com a LLM. Verifique seu arquivo .env e chave API. \nDetalhe: {e}")
                
    else:
        st.markdown(
            "> Dica: Preencha os valores do exame lado esquerdo e clique em **Gerar Análise Assistida** "
            "para que o sistema processe as células e entregue um Laudo Médico automatizado.",
            help="Este sistema só gera os outputs consumindo a LLM do ambiente LlaMA 3"
        )
