Dataset:
    https://www.kaggle.com/uciml/human-activity-recognition-with-smartphones

LSTM for Human Activity Recognition:
    https://news.ycombinator.com/item?id=13049143
    
Keras:
    https://keras.io/getting-started/sequential-model-guide/
    
Results:
    1. Bagging
    
    Using 11 decision trees with 200 features having highest variance
    
    accuracy:  88.8361045131
    confusion matrix: (row=>true class; col=>predicted class)
                        LAYING  SITTING  STANDING  WALKING  WALKING_DOWNSTAIRS  \
    LAYING               537.0      0.0       0.0      0.0                 0.0   
    SITTING                0.0    391.0      42.0      0.0                 0.0   
    STANDING               0.0    100.0     490.0      0.0                 0.0   
    WALKING                0.0      0.0       0.0    465.0                16.0   
    WALKING_DOWNSTAIRS     0.0      0.0       0.0     11.0               359.0   
    WALKING_UPSTAIRS       0.0      0.0       0.0     20.0                45.0   

                        WALKING_UPSTAIRS  
    LAYING                           0.0  
    SITTING                          0.0  
    STANDING                         0.0  
    WALKING                         81.0  
    WALKING_DOWNSTAIRS              14.0  
    WALKING_UPSTAIRS               376.0  
    
    2. Random Forest
    
    Using same config as Bagging
    
    accuracy:  89.4129623346
    confusion matrix: (row=>true class; col=>predicted class)
                        LAYING  SITTING  STANDING  WALKING  WALKING_DOWNSTAIRS  \
    LAYING               537.0      1.0       0.0      0.0                 0.0   
    SITTING                0.0    418.0      49.0      0.0                 0.0   
    STANDING               0.0     72.0     483.0      0.0                 0.0   
    WALKING                0.0      0.0       0.0    455.0                30.0   
    WALKING_DOWNSTAIRS     0.0      0.0       0.0      9.0               350.0   
    WALKING_UPSTAIRS       0.0      0.0       0.0     32.0                40.0   

                        WALKING_UPSTAIRS  
    LAYING                           0.0  
    SITTING                          0.0  
    STANDING                         0.0  
    WALKING                         70.0  
    WALKING_DOWNSTAIRS               9.0  
    WALKING_UPSTAIRS               392.0  

    
