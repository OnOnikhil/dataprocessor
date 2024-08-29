from flask import Flask, request, render_template, redirect, url_for, flash
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib

# Set the Matplotlib backend to 'Agg' for non-interactive use
matplotlib.use('Agg')

app = Flask(__name__)
app.secret_key = 'supersecretkey'
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static/images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = STATIC_FOLDER

# Ensure the upload and static folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            file = request.files['file']
            if file:
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(file_path)
                graphs = process_file(file_path)
                return render_template('index.html', graphs=graphs)
        except Exception as e:
            # Log the error and flash a message to the user
            print(f"Error: {e}")
            flash(f"An error occurred: {e}")
            return redirect(url_for('index'))
    return render_template('index.html')

def process_file(file_path):
    try:
        df = pd.read_excel(file_path)

        # Remove blank rows and columns
        df.dropna(how='all', inplace=True)
        df.dropna(axis=1, how='all', inplace=True)

        # Define the correct column name for ppb values
        ppb_column = 'Pesticide concentration(PPB)'

        # Automatically detect unique ppb values
        ppb_values = df[ppb_column].unique()

        # Automatically detect parameter columns (excluding the ppb column)
        parameters = [col for col in df.columns if col != ppb_column]

        # Create a dictionary to store the average results for each ppb value
        average_results = {}

        # Filter the data and calculate the averages for each ppb value
        for ppb in ppb_values:
            filtered_df = df[df[ppb_column] == ppb]
            averages = filtered_df[parameters].mean()
            average_results[ppb] = averages

        # Create a DataFrame to store the average results
        average_df = pd.DataFrame(average_results).transpose()
        average_df.index.name = ppb_column

        # Perform linear regression and generate graphs
        def perform_linear_regression(df):
            columns = df.columns
            graphs = []

            for column in columns:
                x = df.index.values.reshape(-1, 1)
                y = df[column].values

                # Remove rows where either x or y is NaN
                mask = ~np.isnan(x).flatten() & ~np.isnan(y).flatten()
                x = x[mask]
                y = y[mask]

                if len(x) > 0 and len(y) > 0:
                    model = LinearRegression().fit(x, y)
                    r2 = model.score(x, y)

                    # Plotting
                    plt.figure(figsize=(8, 6))
                    plt.scatter(x, y, color='blue')
                    plt.plot(x, model.predict(x), color='red')
                    plt.xlabel(ppb_column)
                    plt.ylabel(column)
                    plt.title(f'Linear Regression: {ppb_column} vs {column}\nSlope: {model.coef_[0]:.2f}, Intercept: {model.intercept_:.2f}, R^2: {r2:.2f}')
                    
                    graph_path = os.path.join(app.config['STATIC_FOLDER'], f'{ppb_column}_vs_{column}.png')
                    plt.savefig(graph_path)
                    plt.close()
                    
                    graphs.append(url_for('static', filename=f'images/{ppb_column}_vs_{column}.png'))

            return graphs

        # Perform regression analysis and return graph paths
        graphs = perform_linear_regression(average_df)
        return graphs

    except Exception as e:
        # Log the error and re-raise it to be caught in the main route
        print(f"Error in process_file: {e}")
        raise

if __name__ == '__main__':
    app.run(debug=True)
