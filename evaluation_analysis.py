# I have a csv file in the following format:
# statement,dp_id,dp_label,label,assessment1,assessment2
# One in three women is sexually assaulted on the dangerous journey north.,15600,1,False,True,False

# Now, I want to generate a confusion matrix for the value of assessment1 and assessment2 with respect to 
# the agreement between dp_label and label. dp_label and label are agreeing if label is False and dp_label is 0, else not

import pandas as pd

df = pd.read_csv("assistant_evaluation/results.csv")

# Create a confusion matrix
df["agreement"] = df.apply(lambda row: True if row["dp_label"] == 0 and row["label"] == False else False, axis=1)
df["dp_label_bool"] = df["dp_label"].apply(lambda x: False if x == 0 else True)

print(df.head())

#create a confusion matrix for assessment1 and assessment2 with respect to agreement
confusion_matrix1 = pd.crosstab(df["agreement"], df["assessment1"], rownames=["agreement"], colnames=["assessment1"])
confusion_matrix2 = pd.crosstab(df["agreement"], df["assessment2"], rownames=["agreement"], colnames=["assessment2"])
confusion_matrix3 = pd.crosstab(df["label"], df["dp_label_bool"], rownames=["label"], colnames=["dp_label_bool"])

print(confusion_matrix1)
print(confusion_matrix2)
print(confusion_matrix3)



