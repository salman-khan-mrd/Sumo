# Prediction of Sumoylation Sites 
To guarantee proper use of our code, please follow all the steps in the presented order.
## Introduction
Sumoylation is a post-translation modification (PTM) mechanism that involves many critical biological processes, such as gene expression, localizing and stabilizing proteins, and replicating the genome. Moreover, the sumoylation sites are associated with different diseases, including Parkinson’s and Alzheimer’s. Due to its vital role in the biological process, identifying sumoylation sites in proteins is significant for monitoring protein functions and discovering multiple diseases. Therefore, in literature, several computational models utilizing conventional ML methods have been introduced to classify sumoylation sites. However, these models cannot accurately classify the sumoylation sites due to intrinsic limitations associated with the conventional learning methods. 
## Python Packages and Dependencies
The model relies on the following Python packages and their specific versions:
1. Pandas: 1.0.5
2. NumPy: 1.18.5
3. Pip: 20.1.1
4. SciPy: 1.4.1
5. scikit-learn: 0.23.1
6. TensorFlow: 2.3.1

Please ensure you have a compatible Python environment with these dependencies to use the model effectively.
## Create a Virtual Environment
To install the required dependencies for this project, you can create a virtual environment and use the provided requirements.txt file. Make sure your Python environment matches the specified versions above to ensure compatibility.
python -m venv myenv  # Create a virtual environment
source myenv/bin/activate  # Activate the virtual environment
pip install -r requirements.txt  # Install the project dependencies
## Download Protein Sequences--Dataset
In bioinformatics and machine learning, the acquisition or selection of a valid benchmark dataset is an essential step for developing an intelligent computational model. 

We have included a file named datset.fasta as an example in this repository
Once you have downloaded the sequence you can send the sequence as input.
## FEATURE EXTRACTION TECHNIQUES
To generate prominent, reliable, and variant statistical-based discriminative descriptors, several feature encoding approaches have been utilized for the formulation of proteins, RNA, and DNA sequences 20. The detailed overview of the proposed feature encoding schemes is presented as follow.
1. Mismatches (MisM)
2. Accumulated Nucleotide Frequency (ANF)
3. Position-Specific Trinucleotide Propensity Based on Single-Strand (PSTNPSS)
4. Adaptive Skip Dipeptide Composition (ASDC)
5. Dinucleotide-Based Auto Covariance (DAC)

Feature Extraction folder contains all the features extraction related necessary codes used in this study.
#### For Example: Position-Specific Trinucleotide Propensity Based on Single-Strand (PSTNPSS)
```bash
# Initialize dictionary to store trinucleotide counts for each position
trinucleotide_counts = defaultdict(lambda: defaultdict(int))

# Loop through each sequence
for seq in sequences:
    for i in range(len(seq) - 2):
        trinucleotide = seq[i:i+3]  # Extract trinucleotide
        trinucleotide_counts[i][trinucleotide] += 1  # Count trinucleotide at position i

# Convert counts to a DataFrame for easier viewing and manipulation
trinucleotide_df = pd.DataFrame(trinucleotide_counts).fillna(0)

# Normalize to get trinucleotide propensity (frequency)
trinucleotide_df = trinucleotide_df.div(trinucleotide_df.sum(axis=0), axis=1)

# Display the position-specific trinucleotide propensity
print(trinucleotide_df)
```
## Fused Feature Vector 
In this model, we applied five different feature encodings such as Mismatches MisM, ASDC, DAC, PSTNPSS, and ANF to capture the nucleotide-based features keeping their residue ordering information. Moreover, to generate the high discriminative model representing the multi-perspective features, we serially concatenated the extracted features to form an individual vector covering the weakness of the individual feature vector.  

## FEATURE SELECTION TECHNIQUES Using SHAP
SHAPley Additive Explanations (SHAP) uses cooperative game theory to distribute credit among the contributions of input features in machine learning algorithms. 

Feature Selection folder contains all the features selection related necessary codes used in this study.
```bash
# Load your data
# Assuming X is your feature matrix and y is your target variable
# X, y = your_data()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the XGBoost model
model = xgb.XGBClassifier()
model.fit(X_train, y_train)

# Create a SHAP explainer
explainer = shap.TreeExplainer(model)

# Calculate SHAP values
shap_values = explainer.shap_values(X_train)

# Plot the summary plot for feature importance
shap.summary_plot(shap_values, X_train)

# Optional: Select top features based on SHAP values
importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'shap_importance': np.abs(shap_values).mean(axis=0)
}).sort_values(by='shap_importance', ascending=False)

# Choose a threshold or a specific number of top features
top_features = importance_df['feature'].head(10)  # Example: top 10 features

# Filter the original dataset to include only these top features
X_train_selected = X_train[top_features]
X_test_selected = X_test[top_features]

# You can now retrain your model using X_train_selected and X_test_selected
model_selected = xgb.XGBClassifier()
model_selected.fit(X_train_selected, y_train)

# Evaluate your model
accuracy = model_selected.score(X_test_selected, y_test)
print(f"Model accuracy with selected features: {accuracy:.4f}")
```
## Machine Learning Algorithms
The performance of the proposed model with other widely used machine learning algorithms using hybrid features. 
1. Support Vector Machine using Radial Basis Function Kernel
2. Random Forests
3. KNN
4. XGBoost
5. NB
6. LR

## How to Run the Program
#### For Example: SVM Code
```bash
df2 = pd.read_csv(io.BytesIO(uploaded['PSTNPSS.csv']))
X = df2.iloc[:, 0:65].values
y = df2.iloc[:, 66].values
df2.head()

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.2, random_state=42)
print (len(X_train),len(X_test),len(y_train),len(y_test))

#Create a svm Classifier
#clf = svm.SVC(kernel='rbf',C=10, gamma=0.01,random_state=42)

#Train the model using the training sets
clf.fit(X_train, y_train)

scores = cross_val_score(clf, X_train,y_train,cv=5)
print("score = ", scores)
print("Mean ACC = ",scores.mean()*100)
```

## Any Questions?
If you need any help don't hesitate to get in touch with salman-mrd@gmail.com
