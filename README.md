# HiRID data pre-processing

Repository for the HiRID data pre-processing (for irregular setting = imputation was not conducted)

[[Original Paper](https://www.nature.com/articles/s41591-020-0789-4)], [[Dataset Download](https://physionet.org/content/hirid/1.1.1/)]

# Setup

## conda env & code

```
# conda env
conda create --name ehr_preproc python=3.10.18
conda activate ehr_preproc
pip install jupyter
pip install pandas 
pip install numpy 
pip install matplotlib 
pip install scikit-learn

# code
git clone https://github.com/Jeong-Eul/HiRID_PREPROC
cd HiRID_PREPROC/hirid
```


# Fast Run

This repository preprocesses data in a way that maximally preserves the underlying causal structure.
When constructing multivariate time series across all patients, we adhere to the following principles:

1. $X_t$ is observed, and $A_t$ (the treatment or action) is determined based on it.
2. $Y_t$ is determined using $X_t$ and the previous action $A_{t-1}$.

Following these causal consistency rules, our code performs shock early prediction labeling within an 8-hour horizon, using the same annotation and labeling criteria as defined in the original paper.

```
python processing_hirid.py 
    --observation_path [your path] 
    --pharma_path [your path] 
    --general_path [your path] 
    --obs_feature_path [your path] 
    --pharma_feature_path [your path]
    --save_pth [your path]
    --labeled_save_pth [your path] 
    --Action default=True  
    --min_seq_len [seq length minimum] 
    --max_seq_len [seq length maximum] 
    --prediction_window [prediction window (hour)]
    --iqr_factor 1.5 
    --start_los [los min (hour)] 
    --end_los [los max (hour)]
```

- When you download the HiRID dataset, set the path corresponding to raw_stage/observation_tables as observation_path, and the path corresponding to raw_stage/pharma_records as pharma_path.;
- Prepare a list of the variable names and IDs you wish to include, and place it under obs_feature_path (do the same for the pharma variables under pharma_feature_path). ;
- Specify the directory where part-wise aggregation results will be saved (save_pth). ;
- Specify the directory where part-wise annotation and labeling results will be saved (labeled_save_pth). ;
- You can define the length of stay range for patients to be included in the analysis using start_los and end_los.  ;
- The argument Action is treatment behavior. for example, if vasopressor volumn is higher than previous time point, action is "up". if Action is False, treatment parameter is saved as binary indicator  ;

if you start with default setting, use bellow command:

```
python processing_hirid.py
```

# Data preprocessing step

This code performs preprocessing in the following order:
- Step 1 – Aggregation: Vital and laboratory indicators are integrated with pharmaceutical information to construct multivariate time series. During this process, data preprocessing is carried out while preserving the underlying causal structure.
- Step 2 – Annotation/Labeling: Based on the aggregated results, target labels for prediction are generated. This step includes setting the minimum and maximum sequence lengths and applying masking operations.


# Citing

Original paper:

```
@article{hyland2020early,
  title={Early prediction of circulatory failure in the intensive care unit using machine learning},
  author={Hyland, Stephanie L and Faltys, Martin and H{\"u}ser, Matthias and Lyu, Xinrui and Gumbsch, Thomas and Esteban, Crist{\'o}bal and Bock, Christian and Horn, Max and Moor, Michael and Rieck, Bastian and others},
  journal={Nature medicine},
  volume={26},
  number={3},
  pages={364--373},
  year={2020},
  publisher={Nature Publishing Group US New York}
}
```

