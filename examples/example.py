import pandas as pd
import matplotlib.pyplot as plt
from dataLibFarah import DataLib

def main():
    # 1. Data Manipulation
    print("1. Data Manipulation Demonstration")

    # Create new sample dataset
    data = {
        'hours_worked': [35, 40, 45, 50, 30, 60, 55, 25],
        'monthly_expenses': [2000, 2500, 2700, 3000, 1800, 3200, 3100, 1500],
        'savings_rate': [10, 15, 20, 25, 5, 30, 25, 2],
        'country': ['USA', 'UK', 'Canada', 'Germany', 'France', 'India', 'Japan', 'Australia']
    }
    df = pd.DataFrame(data)

    # Filter data
    filtered_df = df[df['hours_worked'] > 40]
    print("Filtered Data (Hours Worked > 40):")
    print(filtered_df)

    # Normalize data
    normalized_df = DataLib.normalize_data(df[['hours_worked', 'monthly_expenses']])
    print("\nNormalized Data:")
    print(normalized_df)

    # 2. Statistical Analysis
    print("\n2. Statistical Analysis Demonstration")

    # Descriptive statistics
    stats = DataLib.calculate_statistics(df[['hours_worked', 'monthly_expenses']])
    print("Descriptive Statistics:")
    print(stats)

    # Correlation analysis
    correlation_matrix = DataLib.correlation_matrix(df[['hours_worked', 'monthly_expenses', 'savings_rate']])
    print("\nCorrelation Matrix:")
    print(correlation_matrix)

    # 3. Data Visualization
    print("\n3. Data Visualization Demonstration")

    # Bar plot
    DataLib.plot_data(df, x_column='country', y_column='monthly_expenses', kind='bar')

    # Scatter plot
    DataLib.plot_data(df, x_column='hours_worked', y_column='monthly_expenses', kind='scatter')

    # Correlation heatmap
    DataLib.correlation_matrix(df[['hours_worked', 'monthly_expenses', 'savings_rate']])

    # 4. Advanced Analysis
    print("\n4. Advanced Analysis Demonstration")

    # Linear Regression
    X = df[['hours_worked', 'savings_rate']]
    y = df['monthly_expenses']

    model, coef, intercept = DataLib.linear_regression(X, y)
    print("Linear Regression Results:")
    print(f"Intercept: {intercept}")
    print(f"Coefficients: {coef}")

    # K-means Clustering
    clustering_results, centers = DataLib.kmeans_clustering(X, n_clusters=3)
    print("\nK-means Clustering Labels:")
    print(clustering_results)

    # PCA
    pca_results, explained_variance = DataLib.apply_pca(X)
    print("\nPCA Explained Variance Ratio:")
    print(explained_variance)

if __name__ == "__main__":
    main()
