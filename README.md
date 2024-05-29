# HiRID_PREPROC  
HiRID 전처리 과정 정리  
tabularization.py: feature selection, tabularization  
imputation.py: data imputation  
aggregation_demographs.py: incorperate the separated parts and add demographic information

```python

# 실행 방법

from tabularization import *
from imp import reload
import imputation
reload(imputation)


parts_list=[]
for i in range(0, 250):
    parts_list.append(i)
    
data_preprocessing_pipeline(parts_list, 'hourly_intervals')
imputation.Imputation(parts_list)
aggregation_demographs.P_Aggregation_Demograph(parts_list)
```
