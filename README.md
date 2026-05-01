# Tech Challenge – Modelos Preditivos de Classificação

Este projeto implementa pipelines de Machine Learning para **classificação supervisionada** em dois cenários de saúde:

- Predição de Diabetes
- Classificação de Câncer de Mama (Breast Cancer Wisconsin)

O foco é demonstrar boas práticas de pré-processamento, pipelines, avaliação e reprodutibilidade.

---

## Estrutura do Projeto

```
.
├── data/
│   └── raw/
├── notebooks/
│   ├── 01_eda_diabetes.ipynb
│   └── 02_eda_breast_cancer.ipynb
├── src/
│   └── project_name/
│       ├── data.py
│       ├── features.py
│       ├── pipelines.py
│       └── train.py
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Datasets

- Breast Cancer Wisconsin: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
- Diabetes Dataset: https://www.kaggle.com/datasets/mathchi/diabetes-data-set

Os arquivos CSV devem ser baixados e colocados em `data/raw/`.

---

## Execução Local

```bash
pip install -r requirements.txt
python src/project_name/train.py
```

---

## Execução com Docker

```bash
docker build -t techchallenge-ml .
docker run --rm techchallenge-ml
```

---

## Execução com Jupyter Notebook

```bash
pip install jupyter notebook ipykernel
jupyter notebook
```

Abra os notebooks da pasta `notebooks/` e selecione o kernel Python correto.

---

## Modelos Utilizados

- Logistic Regression
- Random Forest (Diabetes)
- SVM (Breast Cancer)

---

## Observação

Este projeto é educacional/experimental e **não deve ser usado para decisões clínicas reais**.
