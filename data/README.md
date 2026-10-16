# Data

The dataset (~115 MB) is not committed to this repository.

1. Download the **Fake and Real News Dataset** from Kaggle:
   https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
2. Place `True.csv` and `Fake.csv` in this folder:

```
data/
├── Fake.csv
└── True.csv
```

With the Kaggle CLI:

```bash
kaggle datasets download -d clmentbisaillon/fake-and-real-news-dataset -p data --unzip
```
