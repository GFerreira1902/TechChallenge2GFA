# Tech Challenge Fase 2: Otimização Evolucionária e IA Generativa na Saúde da Mulher

**Integrante / Matrícula: Guilherme Ferreira de Arruda / rm373210**

## Visão Geral do Projeto

Este projeto apresenta um sistema avançado de diagnóstico de rastreio de câncer de mama pautado em otimização bio-inspirada. A arquitetura foi construída com a introdução de **Algoritmos Genéticos** para otimização de hiperparâmetros de um modelo preditivo (_Random Forest_), e a integração de **IA Generativa (LLM)** para traduzir outputs matemáticos complexos em laudos médicos humanizados e explicáveis.

O objetivo clínico do modelo otimizado foi maximizar a métrica de **Recall (Sensibilidade)**, garantindo a minimização de casos Falsos Negativos, que representam o maior risco, ainda mais quando falamos sobre o câncer.

## Arquitetura e Tecnologias

- **Motor de Machine Learning:** `Scikit-Learn` (Random Forest Classifier).
- **Otimizador Evolucionário:** Algoritmo Genético customizado (Crossover Uniforme, Mutação Aleatória e Seleção por Torneio).
- **Explicabilidade (LLM):** Integração com acesso nativo à API `Groq` rodando o modelo open-weight `Llama-3.3-70b-versatile`.
- **Interface Web (UI):** `Streamlit`, feito com CSS.
- **Qualidade e Validação:** `Pytest` (Testes automatizados na classe GA e mock da API LLM).

## Funcionalidades Principais

1. **Treinamento Genético:** Avaliação progressiva focada em função **fitness** parametrizada: `(0.7 * Recall) + (0.3 * Specificity)`.
2. **Fallback Seguro da LLM:** Sistema protegido contra interrupções de serviço. Se a credencial da API não estiver disponível no ambiente, o sistema ativa um Mock Mode que simula o laudo clínico para garantir a continuidade no fluxo web.
3. **Imputação Dinâmica Modular:** Preenchimento estatístico via interface das covariáveis com a mediana global do treino, focando a atenção humana apenas nas Top 5 features de maior impacto (_Feature Importance / SHAP_).

## Instruções de Instalação e Execução

### 1. Clonar o Repositório e Instalar Dependências

```bash
git clone https://github.com/GFerreira1902/TechChallenge2GFA.git
cd TechChallenge2GFA

# Recomendo a ativação de um ambiente virtual (venv)
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate # Linux/Mac

pip install -r requirements.txt
```

### 2. Configurar Autenticação LLaMA (Opcional)

Para garantir o processo dinâmico de geração de textos via IA Generativa, crie um arquivo `.env` na raiz do projeto contendo a credencial:

```env
GROQ_API_KEY=sua_API_key_aqui
```

> **Observação** Se optar por não configurar a chave localmente, foi criado um tratamento em que a interface assumirá o comportamento de _Fallback Seguro_, assim então evitando erros na tela.

### 3. Comando Para Rodar o Dashboard Clínico

```bash
streamlit run app.py
```

## Rodando a Suíte de Validação

O projeto inclui validações assertivas dos operadores bio-inspirados e segurança de integração:

```bash
python -m pytest tests/ -v
```

## Entregáveis do Projeto

- **Repositório GitHub Oficial:** [Acessar Código Fonte](https://github.com/GFerreira1902/TechChallenge2GFA)
  //TODO
- **Vídeo de Demonstração (YouTube):**

---

_Projeto elaborado para a Fase 2 do Tech Challenge - Pós-Graduação IA para Devs FIAP._
