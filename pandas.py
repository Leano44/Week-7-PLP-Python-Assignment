# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore

# Load Iris dataset with error handling
def load_dataset():
    try:
        df = pd.read_csv("iris_dataset.csv")  # Ensure the file is in the same directory
        print("\n✅ Iris Dataset Loaded Successfully!")
        return df
    except FileNotFoundError:
        print("\n❌ Error: File not found. Please check the file path.")
        return None
    except Exception as e:
        print(f"\n❌ Error loading file: {e}")
        return None

# Explore dataset
def explore_dataset(df):
    print("\n📌 First Few Rows:")
    print(df.head())

    print("\n📌 Dataset Info:")
    print(df.info())

    print("\n📌 Missing Values:")
    print(df.isnull().sum())

    # Fill missing values with column mean
    df.fillna(df.mean(), inplace=True)
    print("\n✅ Missing values handled!")

# Perform basic data analysis
def analyze_dataset(df):
    print("\n📊 Basic Statistics:")
    print(df.describe())

    # Group by species and compute mean of numerical columns
    print("\n📊 Average measurements per species:")
    print(df.groupby("target").mean())

# Visualizations
def visualize_dataset(df):
    # Bar Chart: Average petal length per species
    plt.figure(figsize=(8, 5))
    df.groupby("target")["petal length (cm)"].mean().plot(kind="bar", color='skyblue')
    plt.title("📊 Average Petal Length Per Species")
    plt.ylabel("Petal Length (cm)")
    plt.show()

    # Histogram: Sepal length distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df["sepal length (cm)"], bins=20, kde=True)
    plt.title("📊 Distribution of Sepal Length")
    plt.xlabel("Sepal Length (cm)")
    plt.show()

    # Scatter Plot: Sepal length vs Petal length
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=df["sepal length (cm)"], y=df["petal length (cm)"], hue=df["target"])
    plt.title("📊 Sepal Length vs Petal Length")
    plt.show()

# Main function
def main():
    # Load dataset
    df = load_dataset()
    if df is None:
        return  # Stop execution if dataset loading fails
    
    # Explore and analyze
    explore_dataset(df)
    analyze_dataset(df)
    
    # Visualize data
    visualize_dataset(df)

# Run the script
if __name__ == "__main__":
    main()
