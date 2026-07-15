# Tech Challenge Fase 2: OtimizaÃ§Ã£o EvolucionÃ¡ria e IA Generativa na SaÃºde da Mulher

**Integrante / MatrÃ­cula: Guilherme Ferreira de Arruda / rm373210**

## VisÃ£o Geral do Projeto

Este projeto representa a evoluÃ§Ã£o do sistema de diagnÃ³stico de rastreio de cÃ¢ncer de mama que executei na fase 1 do tech challenge. Agora na Fase 2, a arquitetura foi melhorada com a introduÃ§Ã£o de **Algoritmos GenÃ©ticos** para otimizaÃ§Ã£o de hiperparÃ¢metros de um modelo preditivo (_Random Forest_), e a integraÃ§Ã£o de **IA Generativa (LLM)** para traduzir outputs matemÃ¡ticos complexos em laudos mÃ©dicos humanizados e explicÃ¡veis.

O objetivo clÃ­nico do modelo otimizado foi maximizar a mÃ©trica de **Recall (Sensibilidade)**, garantindo a minimizaÃ§Ã£o de casos Falsos Negativos, que representam o maior risco, ainda mais quando falamos sobre o cÃ¢ncer.

## Arquitetura e Tecnologias

- **Motor de Machine Learning:** `Scikit-Learn` (Random Forest Classifier).
- **Otimizador EvolucionÃ¡rio:** Algoritmo GenÃ©tico customizado (Crossover Uniforme, MutaÃ§Ã£o AleatÃ³ria e SeleÃ§Ã£o por Torneio).
- **Explicabilidade (LLM):** IntegraÃ§Ã£o com acesso nativo Ã  API `Groq` rodando o modelo open-weight `Llama-3.3-70b-versatile`.
- **Interface Web (UI):** `Streamlit`, feito com CSS.
- **Qualidade e ValidaÃ§Ã£o:** `Pytest` (Testes automatizados na classe GA e mock da API LLM).

## Funcionalidades Principais

1. **Treinamento GenÃ©tico:** AvaliaÃ§Ã£o progressiva focada em funÃ§Ã£o **fitness** parametrizada: `(0.7 * Recall) + (0.3 * Specificity)`.
2. **Fallback Seguro da LLM:** Sistema protegido contra interrupÃ§Ãµes de serviÃ§o. Se a credencial da API nÃ£o estiver disponÃ­vel no ambiente, o sistema ativa um Mock Mode que simula o laudo clÃ­nico para garantir a continuidade no fluxo web.
3. **ImputaÃ§Ã£o DinÃ¢mica Modular:** Preenchimento estatÃ­stico via interface das covariÃ¡veis com a mediana global do treino, focando a atenÃ§Ã£o humana apenas nas Top 5 features de maior impacto (_Feature Importance / SHAP_).

## InstruÃ§Ãµes de InstalaÃ§Ã£o e ExecuÃ§Ã£o

### 1. Clonar o RepositÃ³rio e Instalar DependÃªncias

```bash
git clone https://github.com/GFerreira1902/TechChallenge2GFA.git
cd TechChallenge2GFA

# Recomendo a ativaÃ§Ã£o de um ambiente virtual (venv)
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate # Linux/Mac

pip install -r requirements.txt
```

### 2. Configurar AutenticaÃ§Ã£o LLaMA (Opcional)

Para garantir o processo dinÃ¢mico de geraÃ§Ã£o de textos via IA Generativa, crie um arquivo `.env` na raiz do projeto contendo a credencial:

```env
GROQ_API_KEY=sua_API_key_aqui
```

> **ObservaÃ§Ã£o** Se optar por nÃ£o configurar a chave localmente, foi criado um tratamento em que a interface assumirÃ¡ o comportamento de _Fallback Seguro_, assim entÃ£o evitando erros na tela.

### 3. Comando Para Rodar o Dashboard ClÃ­nico

```bash
streamlit run app.py
```

## Rodando a SuÃ­te de ValidaÃ§Ã£o

O projeto inclui validaÃ§Ãµes assertivas dos operadores bio-inspirados e seguranÃ§a de integraÃ§Ã£o:

```bash
python -m pytest tests/ -v
```

## EntregÃ¡veis do Projeto

- **RepositÃ³rio GitHub Oficial:** [Acessar CÃ³digo Fonte](https://github.com/GFerreira1902/TechChallenge2GFA)
  //TODO
- **VÃ­deo de DemonstraÃ§Ã£o (YouTube):**

---

_Projeto elaborado para a Fase 2 do Tech Challenge - PÃ³s-GraduaÃ§Ã£o IA para Devs FIAP._



