import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_eda(filepath):
    df = pd.read_csv(filepath)

    print("Data Info:")
    print(df.info())
    print()

    print("Missing values per column:")
    print(df.isnull().sum())
    print()

    print("Class distribution in target variable:")
    print(df['Sales Category'].value_counts())
    print()

    print("Feature data types:")
    print(df.dtypes)
    print()

    # 💡 Only include numeric columns for correlation
    numeric_df = df.select_dtypes(include=['int64', 'float64'])

    # ✅ Heatmap only on numeric features
    plt.figure(figsize=(10, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix of Numeric Features")
    plt.tight_layout()
    plt.savefig("correlation_matrix.png")
    plt.close()

    return df

def preprocess(df):
    # Encode target (label encoding)
    le = LabelEncoder()
    df['target_encoded'] = le.fit_transform(df['Sales Category'])
    
    # Features and target
    X = df.drop(columns=['Sales Category', 'target_encoded'])
    y = df['target_encoded']
    
    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split dataset into 80% train and 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    return X_train, X_test, y_train, y_test, scaler, le
