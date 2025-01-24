# MIND-EEG: Multi-granularity Integration Network with Discrete Codebook for EEG-based Emotion Recognition

## Prepare dataset
We use three publicly available datasets: SEED-IV, SEED-V, MPED. SEED-IV and SEED-V can be requested at https://bcmi.sjtu.edu.cn/ApplicationForm/apply_form/. MPED can be requested at https://github.com/Tengfei000/MPED.

Feel free to write your own code to split each dataset to training and test sets. The common way to separate each dataset is specified in the table:

|    Dataset   | Input_feacture |  num_classes  |  Test Scheme  |
|    :----:    |    :----:   |     :----:    |     :----:    |
| SEED-IV    |      DE       | 4 emotions        | train:test = 2:1   |
| SEED-V    |     DE        | 5 emotions        | 3-fold cross-validation   |
| MPED      |      STFT        |7 emotions       | train:test = 3:1  |



## Train model
Orgnize your datasets in a folder and set the path as follow:
```
--datapath your_datasets_folder_dir
```

Default values of parameters have been set in main_MIND.py. You can run with default with following simple comment:

```
python main_MIND.py
```



