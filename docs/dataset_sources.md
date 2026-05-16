# Churn Dataset Sources

Compatible or near-compatible labeled churn datasets to consider:

- Hugging Face mirror of the expanded IBM Telco churn dataset:
  https://huggingface.co/datasets/jason1966/aadityabansalcodes_telecommunications-industry-customer-churn-dataset
- Kaggle 100K synthetic Telco churn dataset:
  https://www.kaggle.com/datasets/dhrubangtalukdar/telco-customer-churn-data
- Kaggle telecommunications industry customer churn dataset:
  https://www.kaggle.com/datasets/aadityabansalcodes/telecommunications-industry-customer-churn-dataset
- Kaggle telecom customer churn analysis dataset:
  https://www.kaggle.com/datasets/asadullahcreative/telecom-customer-churn-analysis

Training ingestion supports CSVs dropped into:

```text
data/external/labeled_churn/
```

External files should contain compatible Telco churn fields. The training script maps
common variants such as `Customer ID`, `Tenure Months`, `Phone Service`, and
`Churn Label` onto the canonical training schema.
