# Previsão de Evasão e Sucesso Acadêmico com MLP

> **Atividade 02 · Sistemas Inteligentes**
>
> Felipe Kadanos e Marcos José Ferrari de Oliveira

## Tema

Como os dados de matrícula e o desempenho nos dois primeiros semestres ajudam a prever o desfecho acadêmico de um estudante?

O projeto usa uma rede neural MLP para classificar os estudantes em três categorias: **evadido**, **ainda matriculado** ou **formado**. Os dados são do Instituto Politécnico de Portalegre e estão disponíveis no [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/697).

## Como funciona

1. **Preparação dos dados** — As categorias são codificadas e os atributos numéricos são padronizados usando o conjunto de treino.
2. **Treino e validação** — Uma MLP com três camadas ocultas é treinada com ReLU e Tanh. Os hiperparâmetros são comparados no conjunto de validação.
3. **Avaliação** — A melhor configuração é avaliada no conjunto de teste. O código gera métricas e gráficos de desempenho.
