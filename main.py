import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load dataset with real-world cleanup
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df.dropna(inplace=True)
        df = df[df["Size (sqft)"] > 0]
        print("\n--- Dataset Statistics ---")
        print(df.describe())
        return df
    except FileNotFoundError:
        print("Error: File not found.")
        exit()
    except Exception as e:
        print(f"Unexpected error: {e}")
        exit()

# Visualize data
def visualize_data(df):
    plt.scatter(df["Size (sqft)"], df["Price ($)"], color="blue", label="Data Points")  # Label was capitalized
    plt.title("House Size vs Price")
    plt.xlabel("Size (sqft)")
    plt.ylabel("Price ($)")  # Fixed "PRice" typo
    plt.xlim(df["Size (sqft)"].min() - 100, df["Size (sqft)"].max() + 100)
    plt.ylim(df["Price ($)"].min() - 5000, df["Price ($)"].max() + 5000)  # Fixed missing parentheses
    plt.legend()
    plt.grid(True)
    plt.show()

# Train linear regression model
def train_model(df):
    X = df[["Size (sqft)"]]
    y = df["Price ($)"]

    model = LinearRegression()  # Fixed wrong assignment using "-"
    model.fit(X, y)             # Was "s", should be "X"

    predictions = model.predict(X)
    r2 = r2_score(y, predictions)
    coef = model.coef_[0]
    intercept = model.intercept_

    print("\n--- Model Details ---")
    print(f"Coefficient (slope): {coef:.2f}")
    print(f"Intercept: {intercept:.2f}")
    print(f"R² score: {r2:.3f}")  # Fixed typo: "scoer" and wrong format

    return model 

# Visualize regression line
def visualize_model(df, model):
    X = df[["Size (sqft)"]]
    y = df["Price ($)"]
    predictions = model.predict(X)

    plt.scatter(X, y, color="blue", label="Data Points")  # Fixed typos: "bule", wrong variable "s"
    plt.plot(X, predictions, color="red", label="Regression Line")
    plt.title("House Size vs Price with Regression Line")
    plt.xlabel("Size (sqft)")
    plt.ylabel("Price ($)")
    plt.xlim(df["Size (sqft)"].min() - 100, df["Size (sqft)"].max() + 100)
    plt.ylim(df["Price ($)"].min() - 5000, df["Price ($)"].max() + 5000)
    plt.legend()
    plt.grid(True)
    plt.show()

# Predict price from user input
def predict_price(model):
    try:
        sqft = float(input("\nEnter house size (sqft): "))
        if sqft <= 0:
            print("Size must be a positive number.")
            return
        prediction = model.predict([[sqft]])
        rounded_price = round(prediction[0] / 100) * 100
        print(f"Estimated price: ${rounded_price:,.2f}")  # Fixed "Extimated"
    except ValueError:
        print("Invalid input. Please enter a numeric value.")

# Main function
def main():
    df = load_data("housing_data.csv")
    visualize_data(df)
    model = train_model(df)
    visualize_model(df, model)
    predict_price(model)

if __name__ == "__main__":  # Fixed: "if__name__" missing space and colon
    main()


# ---------------------------------------
# References (APA Style)
# ---------------------------------------

# Géron, A. (2019). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (2nd ed.). O'Reilly Media.

# Python Software Foundation. (n.d.). *pandas documentation*. Retrieved July 9, 2025, from https://pandas.pydata.org/docs

# Scikit-learn developers. (n.d.). *Scikit-learn: Machine Learning in Python*. Retrieved July 9, 2025, from https://scikit-learn.org/stable/

# Study.com. (n.d.). *Building Basic Linear Regression Models in Python* [Video lesson]. Instructor: [Insert Instructor Name]. Retrieved July 9, 2025, from https://study.com

# OpenAI. (2025). *ChatGPT* [Large language model]. https://chat.openai.com
