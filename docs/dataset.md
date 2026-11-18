# Descrição do dataset e decisões sobre os atributos

Card A1 · Integrante A. Este documento alimenta a seção 2 do relatório.

> Os números abaixo vêm da execução de `notebooks/01_eda.ipynb` e de `python src/preprocessamento.py` sobre o `data.csv`. O período da coleta vem do artigo de Realinho et al. (2022).

## 1. Identificação

| Item | Informação |
|---|---|
| Nome | *Predict Students' Dropout and Academic Success* |
| Fonte | UCI Machine Learning Repository, id 697: https://archive.ics.uci.edu/dataset/697 |
| Origem | Instituto Politécnico de Portalegre (Portugal) |
| Licença | CC BY 4.0 |
| Período dos dados | 2008/2009 a 2018/2019 (Realinho et al., 2022) |
| Nº de amostras | 4.424 (`df.shape` = (4424, 37)) |
| Nº de atributos | 36 + o target |
| Valores ausentes | Nenhum (0 em todas as colunas). Também não há linhas duplicadas (0). |
| Arquivo | `data/raw/data.csv`, separador `;` |

REALINHO, Valentim *et al*. **Predict Students' Dropout and Academic Success**. UCI Machine Learning Repository, 2021. DOI: 10.24432/C5MC89. Disponível em: https://archive.ics.uci.edu/dataset/697. Acesso em: 18 nov. 2026.

## 2. Problema e target

- **Tipo:** classificação multiclasse (aprendizado supervisionado).
- **Target:** coluna `Target`, com 3 classes: `Dropout` (evasão), `Enrolled` (ainda matriculado) e `Graduate` (formado).
- **Codificação no código:** `Dropout` = 0, `Enrolled` = 1, `Graduate` = 2 (`NOMES_CLASSES` em `src/preprocessamento.py`).
- **Distribuição das classes (conjunto completo):**

  | Classe | Alunos | % |
  |---|---|---|
  | Dropout | 1.421 | 32,1% |
  | Enrolled | 794 | 17,9% |
  | Graduate | 2.209 | 49,9% |

  As classes são desbalanceadas: `Enrolled` tem quase três vezes menos alunos que `Graduate`. A acurácia sozinha engana, então usar F1 macro, recall por classe e matriz de confusão, e testar `class_weight`.

## 3. Tipos de atributo e tratamento

| Grupo | Atributos | Tratamento |
|---|---|---|
| **Categóricos (códigos sem ordem)** | `Marital status`, `Application mode`, `Course`, `Daytime/evening attendance`, `Previous qualification`, `Nacionality`, `Mother's qualification`, `Father's qualification`, `Mother's occupation`, `Father's occupation` | One-hot. Categorias com menos de 1% do treino viram "outros". |
| **Binários (0/1)** | `Displaced`, `Educational special needs`, `Debtor`, `Tuition fees up to date`, `Gender`, `Scholarship holder`, `International` | Mantidos como 0/1, sem padronizar. |
| **Numéricos** | `Application order`, `Previous qualification (grade)`, `Admission grade`, `Age at enrollment`, as 12 colunas `Curricular units 1st/2nd sem (...)`, `Unemployment rate`, `Inflation rate`, `GDP` | Padronização (média 0, desvio 1), com `fit` só no treino. |

Por que one-hot: números como o código do curso (ex.: 9254 vs. 9500) não indicam "mais" ou "menos". Se entrassem como número, a rede inferiria uma ordem inexistente.

O nome `Nacionality` está escrito assim no arquivo original (erro de grafia da fonte) e foi mantido no código. Nomes de coluna com espaço ou tab sobrando são limpos na leitura (`str.strip()`).

## 4. Decisão sobre cada atributo

Decisão padrão: **usar**. A coluna "Observação" registra riscos e o que a EDA precisa confirmar. A coluna "Decisão final" é do grupo, preenchida depois de ver a EDA.

### 4.1. Matrícula e perfil do aluno

| Atributo | Tipo | Decisão proposta | Observação |
|---|---|---|---|
| `Marital status` | categórico | Usar | **Sensível.** Ver seção 5. |
| `Application mode` | categórico | Usar | Modo de candidatura; muitas categorias, várias raras. |
| `Application order` | numérico (0 a 9) | Usar | Ordem de preferência do curso; tratado como número por ter ordem. |
| `Course` | categórico | Usar | Muitas categorias; checar desequilíbrio por curso. |
| `Daytime/evening attendance` | categórico (binário) | Usar | Diurno/noturno. |
| `Previous qualification` | categórico | Usar | Várias categorias raras. |
| `Previous qualification (grade)` | numérico | Usar | |
| `Nacionality` | categórico | Usar, **confirmar** | **Sensível.** Se a EDA mostrar que quase todos têm o mesmo valor, o grupo pode descartar (variância quase nula). Na EDA, 97,5% dos alunos têm a mesma nacionalidade e 20 das 21 categorias são raras (menos de 1%). |
| `Admission grade` | numérico | Usar | |
| `Displaced` | binário | Usar | |
| `Educational special needs` | binário | Usar | **Sensível.** Ver seção 5. |
| `Gender` | binário | Usar | **Sensível.** Ver seção 5. |
| `Age at enrollment` | numérico | Usar | **Sensível** (idade). Ver seção 5. |
| `International` | binário | Usar | **Sensível**; pode ser redundante com `Nacionality`. |

### 4.2. Contexto socioeconômico

| Atributo | Tipo | Decisão proposta | Observação |
|---|---|---|---|
| `Mother's qualification` | categórico | Usar | Muitas categorias raras. |
| `Father's qualification` | categórico | Usar | Idem. |
| `Mother's occupation` | categórico | Usar | Idem. |
| `Father's occupation` | categórico | Usar | Idem. |
| `Debtor` | binário | Usar | |
| `Tuition fees up to date` | binário | Usar | Pode estar muito ligado ao desfecho; discutir. |
| `Scholarship holder` | binário | Usar | |
| `Unemployment rate` | numérico | Usar | Indicador macroeconômico (varia por período, não por aluno). |
| `Inflation rate` | numérico | Usar | Idem. |
| `GDP` | numérico | Usar | Idem. |

### 4.3. Desempenho acadêmico (1º e 2º semestres)

Seis colunas por semestre, com o mesmo sufixo: `(credited)`, `(enrolled)`, `(evaluations)`, `(approved)`, `(grade)`, `(without evaluations)`. São 12 atributos numéricos, todos com decisão **Usar**.

**Cuidado de interpretação:** são medidos *depois* do início do curso (ver seção 5). Tendem a ser os atributos mais informativos. A EDA mostra quatro pares do 1º e do 2º semestre com correlação alta entre si: `credited` (0,945), `enrolled` (0,943), `approved` (0,904) e `grade` (0,837). Há redundância, mas os dois semestres foram mantidos (ver decisão final).

| Decisão final do grupo | |
|---|---|
| Atributos descartados | Nenhum (`COLUNAS_DESCARTADAS` vazia) |
| Justificativa | Em `Nacionality`, 97,5% têm o mesmo valor, mas o pré-processamento já agrupa as categorias raras em "outros", então ela vira uma coluna quase binária e inofensiva. Os pares de 1º e 2º semestre são correlacionados, mas a diferença entre semestres pode ser informativa (progressão do aluno) e a rede lida com entradas redundantes. **Decisão proposta; o grupo pode alterá-la.** |

Se algum atributo for descartado, listar em `COLUNAS_DESCARTADAS` em `src/preprocessamento.py`.

## 5. Pontos de atenção para o relatório

**Atributos sensíveis.** Gênero, nacionalidade, estado civil, necessidades educacionais especiais, idade e condição de estrangeiro descrevem características pessoais. Usá-los pode levar o modelo a tratar grupos de forma desigual, e a EDA (seção 6 do notebook) mostra só diferenças de taxa entre grupos, sem provar causa. No relatório, discutir: (a) se faz sentido usá-los, (b) o risco de decisões discriminatórias se o modelo for usado na prática e (c) que o objetivo aqui é acadêmico.

**Disponibilidade temporal.** As colunas do 1º e do 2º semestres só existem depois do início do curso. O modelo, portanto, não prevê evasão no momento da matrícula, e sim com o curso já em andamento. Isso deve ser dito como limitação, pois o desempenho obtido seria menor em um cenário de previsão precoce.

**Classe `Enrolled`.** Tende a ser a mais difícil de prever (aluno ainda em curso, sem desfecho final). Conferir nos resultados do parceiro.

## 6. Pré-processamento aplicado (resumo para a seção 2)

1. Leitura de `data/raw/data.csv` (`sep=";"`) e limpeza dos nomes de coluna.
2. Target de texto para inteiros (0, 1, 2).
3. Divisão **estratificada 70/15/15** (treino/validação/teste), `random_state=42`.
4. Categorias raras (menos de 1% do **treino**) agrupadas em "outros", depois one-hot.
5. Padronização das colunas numéricas, com média e desvio calculados **só no treino** e aplicados em validação e teste (evita vazamento de dados).
6. Resultado: `X` numérico, sem `NaN`; nº de atributos após one-hot **110** (saída de `python src/preprocessamento.py`).
7. Pesos de classe (`class_weight`) calculados só com o treino.

Tamanhos das partições: treino 3.096, validação 664 e teste 664. A proporção das classes foi preservada em cada parte:

| Partição | Dropout | Enrolled | Graduate |
|---|---|---|---|
| Treino | 994 (32,1%) | 556 (18,0%) | 1.546 (49,9%) |
| Validação | 213 (32,1%) | 119 (17,9%) | 332 (50,0%) |
| Teste | 214 (32,2%) | 119 (17,9%) | 331 (49,8%) |

Pesos de classe (treino): Dropout 1,04 · Enrolled 1,86 · Graduate 0,67.

## 7. Principais achados da EDA

- **Desempenho acadêmico é o sinal mais forte.** Correlação com o target (0/1/2): unidades aprovadas no 2º semestre 0,62; nota do 2º semestre 0,57; aprovadas no 1º semestre 0,53; nota do 1º semestre 0,49. Isso é uma correlação linear aproximada, não prova de importância.
- **Situação financeira:** mensalidades em dia (0,41) e bolsa (0,30) se associam a mais formados. Entre devedores, 62,0% evadiram contra 28,3% entre não devedores. Entre bolsistas, 12,2% evadiram contra 38,7% entre não bolsistas.
- **Idade:** a idade média na matrícula é 26,1 anos entre quem evadiu e cerca de 22 entre quem não evadiu (Enrolled 22,4; Graduate 21,8).
- **Gênero (atributo sensível):** com `Gender` = 1, 45,1% evadiram; com `Gender` = 0, 25,1%. Pela documentação do UCI, 1 = masculino e 0 = feminino. É uma diferença de taxas, não prova de causa, e entra na discussão ética.
- **Grupos pequenos:** só 51 alunos têm necessidades educacionais especiais e 110 são internacionais; qualquer conclusão sobre eles é frágil.
- **Categóricos com muitas categorias raras:** qualificação e ocupação dos pais têm de 21 a 34 categorias raras cada, o que justifica agrupá-las em "outros" (de 36 atributos originais se chega a 110 colunas após o one-hot).
