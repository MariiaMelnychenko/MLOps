# MLOps Lab — Версіонування даних та ML-пайплайн з DVC та MLflow

## Огляд проєкту

Цей проєкт демонструє побудову відтворюваного пайплайну машинного навчання з використанням сучасних MLOps-інструментів:

- **Git** — контролю версій коду
- **DVC (Data Version Control)** — версіонування датасетів та оркестрація пайплайну
- **MLflow** — відстеження експериментів та логування моделей
- **Scikit-learn** — навчання моделей
- **Optuna** — гіперпараметрична оптимізація
- **Hydra** — конфігурація експериментів у YAML
- **Python** — обробка даних та скрипти пайплайну

Проєкт використовує датасет **Telco Customer Churn** для навчання класифікаційної моделі, що передбачає, чи клієнт залишить послугу.

## Структура проєкту

```
mlops1/
│
├── data/
│   ├── raw/
│   │   ├── telco.csv
│   │   └── telco.csv.dvc
│   │
│   ├── prepared/
│   │   ├── train.csv
│   │   └── test.csv
│   │
│   └── models/
│       ├── model.pkl
│       ├── metrics.json
│       ├── confusion_matrix.png
│       └── feature_importance.png
│
├── config/              # Hydra: config.yaml, model/, hpo/
├── src/
│   ├── prepare.py
│   ├── train.py
│   └── optimize.py      # Optuna + MLflow nested runs
├── tests/
│   ├── test_data.py     # Pre-train: валідація даних
│   └── test_model.py    # Post-train: артефакти + Quality Gate
├── scripts/
│   └── compare_samplers.py
├── .github/workflows/
│   └── cml.yaml         # CI/CD: GitHub Actions + CML
├── reports/
├── models/
├── notebooks/
│   └── 01_eda.ipynb
│
├── dvc.yaml
├── dvc.lock
├── requirements.txt
├── README.md
└── .gitignore
```

## Датасет

Проєкт використовує датасет **Telco Customer Churn**, який містить інформацію про клієнтів телеком-компанії та їх відтік.

- **Цільова змінна:** `Churn`
- **Задача:** Передбачення відтоку клієнтів

## Огляд пайплайну

```
Сирі дані
   │
   ▼
Етап: Підготовка
   │
   ▼
Підготовлені дані
   │
   ▼
Етап: Навчання
   │
   ▼
Навчена модель + метрики
```

### Етап 1 — Підготовка даних

**Скрипт:** `src/prepare.py`

**Функції:**

- завантаження сирих даних
- видалення пропущених значень
- кодування категоріальних змінних
- розділення на тренувальну та тестову вибірки

**Вихідні файли:**

- `data/prepared/train.csv`
- `data/prepared/test.csv`

### Етап 2 — Навчання моделі

**Скрипт:** `src/train.py`

**Функції:**

- завантаження підготовлених даних
- навчання RandomForestClassifier
- обчислення метрик якості
- логування в MLflow
- побудова графіків

**Вихідні файли:**

- `data/models/model.pkl`
- `data/models/metrics.json`
- `data/models/confusion_matrix.png`
- `data/models/feature_importance.png`

## Відстеження експериментів (MLflow)

MLflow використовується для логування:

**Параметрів:**

- `n_estimators`
- `max_depth`

**Метрик:**

- `train_accuracy`
- `test_accuracy`
- `train_f1`
- `test_f1`

**Артефактів:**

- confusion_matrix.png
- feature_importance.png
- навчена модель

Запуск MLflow UI:

```bash
mlflow ui
```

Відкрити: [http://127.0.0.1:5000](http://127.0.0.1:5000)

## HPO (Optuna) та Hydra

Перед запуском потрібні підготовлені дані (`dvc repro` або наявні `data/prepared/*.csv`).

```bash
python src/optimize.py
```

**Параметри з командного рядка (Hydra):**

```bash
python src/optimize.py hpo.n_trials=30
python src/optimize.py hpo=random
python src/optimize.py model=logistic_regression
python src/optimize.py hpo.use_cv=true
```

**Порівняння sampler TPE vs Random (однаковий `n_trials`):**

```bash
python scripts/compare_samplers.py
```

У MLflow з’являється **батьківський run** (study) і **вкладені child runs** для кожного trial; після HPO модель з найкращими параметрами перетреновується на повному train, оцінюється на **test**, зберігається як `models/best_model.pkl` і логуються `best_params.json`, `config_resolved.json`.

**Model Registry (Staging):** у `config/config.yaml` встановіть `mlflow.register_model: true` і запустіть MLflow Tracking Server з backend store (SQLite/PostgreSQL); з чисто файловим `./mlruns` реєстрація може бути недоступна.

## Версіонування даних (DVC)

DVC використовується для версіонування датасетів та оркестрації пайплайну.

- **Етапи пайплайну:** prepare, train
- **Запуск пайплайну:** `dvc repro`
- **Відправка даних у сховище:** `dvc push`
- **Перевірка статусу:** `dvc status`

## Встановлення

1. Клонувати репозиторій та перейти в папку проєкту:
  ```bash
   git clone <URL_репозиторію>
   cd mlops1
  ```
2. Створити віртуальне середовище:
  ```bash
   python -m venv venv
  ```
3. Активувати середовище (Windows):
  ```bash
   venv\Scripts\activate
  ```
4. Встановити залежності:
  ```bash
   pip install -r requirements.txt
  ```

## Запуск пайплайну

Виконати DVC-пайплайн:

```bash
dvc repro
```

Це автоматично:

- підготує датасет
- навчить модель
- згенерує артефакти

## CI/CD (GitHub Actions + CML)

При кожному **push** або **pull_request** автоматично:

1. Встановлюються залежності
2. Лінтинг коду (`flake8`, `black`)
3. **Pre-train тести** — валідація структури даних (`tests/test_data.py`)
4. Підготовка і навчання моделі (`prepare.py` → `train.py`)
5. **Post-train тести** — перевірка артефактів + Quality Gate за F1 (`tests/test_model.py`)
6. **CML-звіт** у PR: метрики, confusion matrix, feature importance

**Quality Gate:** модель має мати F1 >= поріг (за замовчуванням 0.50).

**CD:** при push у `main` — `model.pkl` та `metrics.json` зберігаються як workflow artifact.

Запуск тестів локально:

```bash
pytest tests/ -v
```

## Використані технології

- Python
- Scikit-learn
- MLflow
- DVC
- Optuna
- Hydra
- GitHub Actions + CML
- pytest
- Pandas
- NumPy
- Matplotlib
- Seaborn

