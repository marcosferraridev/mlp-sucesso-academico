# Previsão de Evasão e Sucesso Acadêmico com MLP

Guia de escopo e colaboração para os dois integrantes e suas IAs.
**Disciplina:** Sistemas Inteligentes – 2026/02 · **Profª:** Malga Costa · **Atividade 02** (peso 1,5) · **Entrega:** até 18/11/2026

> **Status:** projeto planejado, ainda não implementado. A estrutura, os contratos e os comandos abaixo descrevem a entrega esperada. Este documento deve acompanhar o código na raiz do repositório.

**Objetivo:** entregar a solução mais simples que cumpra todos os requisitos avaliados pela professora. Código simples, resultados reais e compreensão pelos integrantes têm prioridade. Não há garantia de nota.

## Regra obrigatória para as IAs

Implemente exclusivamente a tarefa atribuída ao integrante no **card** informado. Altere somente os arquivos autorizados no card. Não implemente tarefas do outro integrante, não antecipe etapas e não faça melhorias, refatorações ou correções fora da tarefa.

Se faltar uma parte do outro integrante, informe a dependência e pare no limite da sua tarefa. Não crie uma implementação substituta. Se precisar mudar um arquivo compartilhado ou um contrato, explique a necessidade e aguarde a combinação antes de fazê-lo.

Nunca invente resultados. Números, métricas e gráficos do relatório devem vir de execuções reais do código. Onde faltar dado, use o marcador `[PREENCHER]`.

Sem integrante, card e arquivos autorizados identificados, a IA deve pedir essas informações antes de escrever código. Este README dá contexto do projeto inteiro; não autoriza a IA a implementar tudo.

---

## 1. Objetivo e requisitos da professora

Aplicar **aprendizado supervisionado** e **MLP** a um dataset público real. O enunciado original é o `Descritivo_Atividade_02.docx` (guardar em `docs/`) e é a referência para as exigências.

**Entrega obrigatória**

1. **Dataset:** descrito (fonte, período, nº de amostras e atributos), com atributos justificados, dados normalizados/padronizados e target definido.
2. **Código da MLP** (`.py` ou `.ipynb`) em Python com Keras/TensorFlow ou PyTorch:
   - pelo menos **3 camadas ocultas**;
   - pelo menos **2 funções de ativação** testadas;
   - taxa de aprendizado, épocas e batch size **ajustados e justificados**;
   - **vetor de erros exibido a cada iteração**;
   - gráficos de desempenho (perda, acurácia etc.).
3. **Relatório técnico** (Word ou PDF, **ABNT**): Introdução e Contexto, Descrição do Dataset, Metodologia, Resultados, Discussão e Conclusão.

Sem relatório, a etapa vale 0,0. Trabalhos idênticos ou com indício de cópia recebem 0,0. Pontuação: 0,5 por etapa (dataset, MLP, relatório).

Fluxo esperado:

```
data.csv → pré-processamento → MLP → métricas e gráficos → relatório
```

### Dataset escolhido

*Predict Students' Dropout and Academic Success* (UCI, id 697): https://archive.ics.uci.edu/dataset/697

- Instituto Politécnico de Portalegre (Portugal), licença **CC BY 4.0**.
- 4.424 alunos e 36 atributos, sem valores ausentes segundo a documentação. Confirmem os números ao carregar o arquivo (algumas publicações citam 4.423).
- Mistura dados da matrícula (demográficos, socioeconômicos, trajetória) e desempenho do 1º e 2º semestres.
- **Problema:** classificação multiclasse. **Target:** coluna `Target` com 3 classes: `Dropout`, `Enrolled`, `Graduate`.
- Referência (rascunho ABNT, conferir): REALINHO, Valentim *et al*. **Predict Students' Dropout and Academic Success**. UCI Machine Learning Repository, 2021. DOI: 10.24432/C5MC89. Acesso em: [data].

**Pontos de atenção**

- Colunas como curso, modo de candidatura, estado civil, nacionalidade e qualificação/ocupação dos pais são **códigos inteiros sem ordem**: tratar com *one-hot* (ou agrupar categorias raras), não como números.
- Classes **desbalanceadas**: acurácia sozinha engana. Usar também F1 macro, recall por classe, matriz de confusão e testar `class_weight`.
- `Enrolled` tende a ser a classe mais difícil de prever.
- As notas do 1º e 2º semestres só existem depois do início do curso, e há atributos sensíveis (gênero, nacionalidade, estado civil, necessidades especiais). Ambos devem ser discutidos no relatório.
- O `data.csv` usa **ponto e vírgula** (`sep=";"`). Conferir nomes de colunas com espaços ou caracteres extras.

## 2. Decisões de simplicidade do grupo

- Python 3.11 (ou outra versão suportada pelo TensorFlow; conferir antes de instalar).
- Dependências diretas: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `jupyter` e `tensorflow` (usando a API **Keras**). Biblioteca de rede escolhida: **Keras (TensorFlow)**.
- Funções simples; sem classes, camadas de arquitetura ou abstrações desnecessárias.
- Experimentos manuais e pequenos (variar **um fator por vez**, mesma seed e mesmos dados).
- Seed fixa (`42`) em tudo, para reprodutibilidade.
- Comentários curtos em português explicando cada decisão, pois elas vão para o relatório.

**Fora do escopo:** API ou Flask, deploy, Docker, AutoML/Optuna, outros modelos além da MLP, técnicas de balanceamento além de `class_weight`, dashboards e qualquer funcionalidade "para melhorar" o trabalho.

Estrutura prevista na raiz do repositório:

```
README.md
requirements.txt
.gitignore
docs/
    Descritivo_Atividade_02.docx
    dataset.md
data/raw/data.csv
notebooks/01_eda.ipynb
src/
    preprocessamento.py
    modelo.py
resultados/
    metricas.csv
    metricas_teste.csv
    graficos/
relatorio/relatorio_tecnico.docx
```

O ambiente `.venv/` é local e não entra no Git. O `.gitignore` deve incluir `.venv/`, `__pycache__/`, `*.pyc` e `.ipynb_checkpoints/`.

## 3. Contratos de integração

Estes nomes e formatos devem ser respeitados por todos. Não renomear funções, arquivos ou colunas unilateralmente.

### 3.1. Dados: `src/preprocessamento.py` (Integrante A)

```python
NOMES_CLASSES = ["Dropout", "Enrolled", "Graduate"]   # índice = código da classe (0, 1, 2)

def carregar_dados() -> tuple:
    """Retorna X_train, X_val, X_test, y_train, y_val, y_test (arrays NumPy)."""

def carregar_pesos_classe() -> dict[int, float]:
    """Pesos de classe calculados só com o treino, para usar em class_weight."""
```

Regras:

- Ler `data/raw/data.csv` (`sep=";"`).
- Divisão **estratificada 70/15/15** com `random_state=42`.
- Normalização/padronização com `fit` **somente no treino**, aplicada em validação e teste.
- `X` totalmente numérico, sem `NaN`; `y` com inteiros 0, 1, 2.
- Colunas categóricas tratadas conforme a seção 1, com a decisão comentada.
- Mesmo resultado a cada execução.
- Não importar TensorFlow/PyTorch, não treinar e não usar o conjunto de teste para decisões.
- Ao executar `python src/preprocessamento.py`: imprimir os shapes, o **nº de atributos de entrada** e a distribuição das classes em cada partição.

### 3.2. Modelo: `src/modelo.py` (Integrante B)

```python
def construir_mlp(input_dim: int, ativacao: str, learning_rate: float):
    """MLP com 3+ camadas ocultas (inicial: 64-32-16), saída softmax de 3 neurônios."""

def treinar(modelo, dados: tuple, epocas: int, batch_size: int, class_weight=None) -> dict:
    """Treina, imprime a loss de treino e validação a cada época e devolve o histórico."""

def avaliar(modelo, X, y) -> dict:
    """Retorna acuracia, f1_macro, precisao/recall/f1 por classe e matriz_confusao."""
```

Regras:

- `dados` é a tupla devolvida por `carregar_dados()`.
- O histórico tem listas por época: `loss`, `val_loss`, `accuracy`, `val_accuracy`. O **vetor de erros** é exibido durante o treino e ao final.
- `ativacao` aceita pelo menos `"relu"` e `"tanh"`.
- Escolhas de hiperparâmetros são feitas **na validação**. O conjunto de teste só é usado na avaliação final (card B5).
- Dados sintéticos (`make_classification`) são permitidos **apenas** no card B1 e em verificações isoladas. O treino final usa sempre `carregar_dados()`.

Saídas combinadas:

| Arquivo | Conteúdo |
|---|---|
| `resultados/metricas.csv` | Um experimento por linha: `ativacao, learning_rate, epocas, batch_size, class_weight, loss_treino, loss_val, acc_val, f1_macro_val` |
| `resultados/metricas_teste.csv` | Métricas do modelo final no teste |
| `resultados/graficos/` | `perda_<ativacao>.png`, `comparacao_ativacoes.png`, `hiperparametros.png`, `matriz_confusao.png` |

### 3.3. Relatório: `relatorio/relatorio_tecnico.docx`

| Seção | Responsável |
|---|---|
| Capa, resumo, sumário, formatação ABNT e referências | A |
| 1 Introdução e Contexto · 2 Descrição do Dataset · 6 Conclusão | A |
| 3 Metodologia · 4 Resultados Obtidos · 5 Discussão | B |

Todas as figuras e tabelas têm título, numeração e fonte. A IA pode gerar rascunhos, mas o texto final deve ser **revisado e reescrito pelos integrantes** (autoria).

## 4. Divisão de tarefas

| Integrante | Responsabilidade | Arquivos da tarefa principal |
|---|---|---|
| **A** – _(nome)_ | Dataset, pré-processamento e relatório (seções 1, 2, 6 e formatação) | `docs/dataset.md`, `notebooks/01_eda.ipynb`, `src/preprocessamento.py` |
| **B** – _(nome)_ | MLP, experimentos e resultados (seções 3, 4 e 5) | `src/modelo.py`, `resultados/` |

A e B trabalham **em paralelo** até o card B2, que depende do A3. Sem datas fixas: cada um avança conforme sua disponibilidade, respeitando a ordem das dependências.

### Cards

**Card 0: Alinhamento (juntos)** · Arquivos: `requirements.txt`, `.gitignore`, `data/raw/data.csv`, `docs/`, este README
Conferir o dataset no UCI, baixar o `data.csv`, confirmar que o TensorFlow instala na versão de Python usada, criar a estrutura de pastas e as branches.
*Pronto quando:* repositório criado e `pip install -r requirements.txt` funciona.

**Integrante A**

| Card | Tarefa | Arquivos autorizados | Depende de | Pronto quando |
|---|---|---|---|---|
| A1 | Descrever o dataset, justificar atributos (usar/descartar), apontar atributos sensíveis e fazer a EDA (ausentes, distribuições, desbalanceamento, correlações) | `docs/dataset.md`, `notebooks/01_eda.ipynb`, `resultados/graficos/` | Card 0 | Todo atributo tem decisão justificada e o notebook roda do início ao fim |
| A2 | Limpeza e codificação: one-hot ou agrupamento dos códigos categóricos, target em 0/1/2 | `src/preprocessamento.py` | A1 | Dataset sem `NaN`, todo numérico, decisões comentadas |
| A3 | Normalização, divisão estratificada, `carregar_dados()` e `carregar_pesos_classe()` | `src/preprocessamento.py` | A2 | `python src/preprocessamento.py` roda e imprime shapes, nº de atributos e distribuição das classes. **Avisar o B.** |
| A4 | Relatório: estrutura ABNT, Introdução e Descrição do Dataset | `relatorio/relatorio_tecnico.docx` | A1 | Seções 1 e 2 escritas e revisadas, fontes nas referências |
| A5 | Relatório: Conclusão, referências e formatação final | `relatorio/relatorio_tecnico.docx` | A4, B6 | Sem `[PREENCHER]`, figuras e tabelas com legenda e fonte |

**Integrante B**

| Card | Tarefa | Arquivos autorizados | Depende de | Pronto quando |
|---|---|---|---|---|
| B1 | Esqueleto da MLP com dados sintéticos, vetor de erros por época e gráfico de perda | `src/modelo.py`, `resultados/graficos/` | Card 0 | Treina sem erro, 3+ camadas ocultas, vetor de erros impresso e gráfico gerado |
| B2 | Conectar a `carregar_dados()` e treinar com a 1ª ativação (ReLU) | `src/modelo.py`, `resultados/` | B1, **A3** | Treina com dados reais, curvas geradas, resultados anotados |
| B3 | 2ª ativação (Tanh) em comparação justa | `src/modelo.py`, `resultados/` | B2 | `metricas.csv` e `comparacao_ativacoes.png` com 2+ ativações |
| B4 | Ajuste de learning rate, batch size e épocas (e `class_weight` com/sem), com justificativa baseada nos resultados | `src/modelo.py`, `resultados/` | B3 | Cada hiperparâmetro final tem justificativa respaldada na tabela |
| B5 | Avaliação final no **teste** com a melhor configuração, mais baseline (classe majoritária) | `src/modelo.py`, `resultados/` | B4 | `metricas_teste.csv` e `matriz_confusao.png` gerados; roda do zero com os mesmos resultados |
| B6 | Relatório: Metodologia, Resultados e Discussão | `relatorio/relatorio_tecnico.docx` | B5 | Seções 3, 4 e 5 com números reais, revisadas. **Avisar o A.** |

**Card Final: revisão e entrega (juntos)**
Revisão cruzada, conferir coerência entre relatório e resultados reais, clonar em pasta limpa e rodar do zero, exportar o relatório em PDF e fazer o merge final.

Marcos para merge na `main`: **Marco 1** = A3 e B1 concluídos · **Marco 2** = B5 concluído · **Marco 3** = relatório completo (A5 + B6).

## 5. Fluxo de colaboração

Cada card deve ter responsável, objetivo, arquivos autorizados, dependências e critério de conclusão. A IA recebe autorização **somente** para o card informado. Uma lista no Trello (A fazer → Fazendo → Revisão → Concluído) é opcional.

1. Atualizar a cópia local a partir da `main` antes de iniciar.
2. Criar uma branch por card (ex.: `feat/preprocessamento`, `feat/mlp-ativacoes`, `docs/relatorio-secoes-1-2`).
3. Implementar exclusivamente o card, com commits pequenos e descritivos (`feat:`, `docs:`, `chore:`). Sugestão: um commit por card concluído.
4. Revisar o código gerado pela IA e executar as verificações possíveis da própria parte.
5. Abrir pull request (ou merge) para a `main`, descrevendo o que mudou, o que foi verificado e as dependências.
6. Revisar e integrar com o parceiro.

Não enviar `.venv`, caches ou arquivos temporários. Não sobrescrever o trabalho do outro nem fazer refatorações gerais. Se um contrato precisar mudar, combinar com o outro integrante e registrar a decisão neste README antes da integração (a edição do README também deve estar autorizada no card).

## 6. Prompt para cada integrante copiar para sua IA

Preencha todos os campos antes de enviar:

```
Você está colaborando em um trabalho acadêmico de duas pessoas (Sistemas Inteligentes,
Atividade 02: MLP para prever evasão e sucesso acadêmico de estudantes).

Integrante: [A ou B e nome]
Card: [código, título e descrição da tarefa]
Arquivos autorizados: [lista exata]
Critério de conclusão: [resultado verificável]
Dependências conhecidas: [partes necessárias do outro integrante]

Leia README.md e o código existente relacionado à tarefa. O README dá contexto do
projeto inteiro, mas sua autorização se limita ao card acima. Se integrante, card ou
arquivos autorizados não estiverem identificados, peça essas informações antes de
escrever código.

Implemente exclusivamente a tarefa deste card. Altere somente os arquivos autorizados.
Não implemente tarefas do outro integrante, não antecipe etapas e não faça melhorias,
refatorações ou correções fora da tarefa.

Se faltar uma parte do outro integrante, informe a dependência e pare no limite da sua
tarefa. Não crie uma implementação substituta. Se precisar alterar um arquivo
compartilhado ou um contrato, explique a necessidade e aguarde a combinação.

Priorize a solução mais simples que atenda aos requisitos da professora. Não adicione
funcionalidades, bibliotecas ou arquivos sem necessidade. O código deve ser curto,
legível, com comentários em português explicando as decisões. Respeite os nomes e
formatos definidos nos contratos do README. Use seed fixa (42).

Não invente dados, fontes ou resultados. Textos do relatório devem se basear somente
em números e gráficos reais que eu colar aqui; onde faltar dado, use [PREENCHER].
Dados sintéticos só são permitidos no card B1 e em verificações isoladas.

Verifique o que for possível da sua parte sem implementar partes alheias. Não afirme
ter executado testes que não executou.

Ao finalizar, informe o que foi alterado, como foi verificado e quais dependências
permanecem pendentes. Não continue com outras tarefas.
```

## 7. Downloads, instalação e execução previstas

### O que baixar e instalar (cada integrante)

| Item | Para quê | Onde obter | Obrigatório |
|---|---|---|---|
| **Python 3.11** (ou versão suportada pelo TensorFlow) | Executar o código. No Windows, marcar **"Add Python to PATH"** na instalação | python.org/downloads | Sim |
| **Git** | Versionamento e commits por card | git-scm.com/downloads | Sim |
| **Conta no GitHub** | Repositório compartilhado da dupla | github.com | Sim |
| **Dataset `data.csv`** | Base de dados do trabalho. Na página do UCI, clicar em *Download*, extrair o `.zip` e salvar em `data/raw/` | archive.ics.uci.edu/dataset/697 | Sim |
| **Pacotes Python** (via `requirements.txt`) | `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `tensorflow` e `jupyter` | `pip install -r requirements.txt` | Sim |
| **Word ou LibreOffice** | Escrever o relatório e exportar em PDF | office.com ou libreoffice.org | Sim |
| **VS Code** (extensões *Python* e *Jupyter*) | Editar código e abrir o notebook da EDA | code.visualstudio.com | Recomendado |

Observações:

- O `tensorflow` é um pacote grande (algumas centenas de MB), então instale com boa conexão. **Não é preciso GPU**: o treino roda na CPU.
- Instalar os pacotes **dentro do ambiente virtual** (`.venv`), nunca no Python global.
- O `data.csv` é pequeno (cerca de 500 KB) e entra no repositório, pois o dataset faz parte da entrega.
- Conferir a instalação antes de começar:
  `python --version` · `git --version` · `python -c "import tensorflow as tf; print(tf.__version__)"`

### Instalação e execução

Estes comandos serão utilizáveis quando os módulos e o `requirements.txt` estiverem implementados. Executar na raiz do repositório, que deve ter seu próprio ambiente virtual.

Conteúdo previsto de `requirements.txt`:

```
numpy
pandas
scikit-learn
matplotlib
tensorflow
jupyter
```

**Windows / PowerShell**

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src/preprocessamento.py
python src/modelo.py
```

**Linux / macOS**

```
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/preprocessamento.py
python src/modelo.py
```

`preprocessamento.py` valida os dados. `modelo.py` treina, imprime o vetor de erros por época e gera os arquivos em `resultados/`. A EDA fica em `notebooks/01_eda.ipynb` (abrir com `jupyter notebook` ou pelo VS Code).

## 8. Checklist de entrega

- [ ] Um clone do repositório funciona com ambiente virtual novo, seguindo este README.
- [ ] Dataset (`data.csv`) incluído, com fonte, período, nº de amostras e atributos descritos.
- [ ] Atributos justificados, dados normalizados e target definido.
- [ ] MLP com 3 ou mais camadas ocultas.
- [ ] Pelo menos 2 funções de ativação testadas e comparadas.
- [ ] Learning rate, épocas e batch size ajustados e justificados com experimentos.
- [ ] Vetor de erros exibido a cada época.
- [ ] Gráficos de desempenho gerados (perda, acurácia, matriz de confusão).
- [ ] Teste usado **só** na avaliação final; F1 macro e matriz de confusão reportados.
- [ ] Relatório com Introdução, Dataset, Metodologia, Resultados, Discussão e Conclusão, em ABNT, com referências e nomes na capa.
- [ ] Números do relatório coincidem com os resultados reais do código.
- [ ] Código entregue contém somente o necessário.
- [ ] Os dois integrantes revisaram e conseguem explicar o código, os resultados e o relatório.