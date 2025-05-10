# AI-stock-predictor

A Python project to predict stock market prices using an LSTM (Long Short-Term Memory) neural network. This project fetches historical stock data, preprocesses it, trains an AI model, and visualizes the predictions.

## Setup and Installation

Follow these steps to set up the project environment:

1.  **Clone the Repository:**
    If you haven't already, clone the project to your local machine:
    ```bash
    git clone https://github.com/taniatitiriga/AI-stock-predictor
    cd AI-stock-predictor
    ```

2.  **Create a Python Virtual Environment:**
    It is highly recommended to use a virtual environment to isolate project dependencies. This ensures that the project uses the correct package versions and avoids conflicts with other Python projects.
    Navigate to the project's root directory (`AI-stock-predictor`) and run:
    ```bash
    python -m venv stock_env
    ```
    This will create a directory named `stock_env` for the virtual environment.

3.  **Activate the Virtual Environment:**
    Before installing dependencies, you need to activate the environment:
    *   **On macOS and Linux:**
        ```bash
        source stock_env/bin/activate
        ```
    *   **On Windows (Command Prompt):**
        ```bash
        stock_env\Scripts\activate.bat
        ```
    *   **On Windows (PowerShell):**
        ```bash
        .\stock_env\Scripts\Activate.ps1
        ```
        *(If you encounter an execution policy error on PowerShell, you might need to run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` for the current session and then try activating again.)*

    Your command prompt/terminal should now indicate that the `(stock_env)` is active.

4.  **Upgrade Pip (Optional but Recommended):**
    Ensure you have the latest version of pip within your activated environment:
    ```bash
    python -m pip install --upgrade pip
    ```

5.  **Install Required Packages:**
    With the virtual environment activated, install all the necessary Python packages listed in the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```
    This command reads the `requirements.txt` file and installs the specified versions of each package.

## Configuration

Before running the application, you can customize its behavior by editing the `src/config.py` file. Key parameters include:

*   `TICKER`: The stock symbol to analyze (e.g., 'AAPL', 'MSFT').
*   `START_DATE`, `END_DATE`: The historical period for fetching data.

## Running the Application

Once the setup is complete and the virtual environment is activated, you can run the main stock prediction pipeline:

1.  Navigate to the root directory of the project in your terminal.
2.  Execute the `main.py` script:
    ```bash
    python main.py
    ```

The script will perform the following actions:
*   Fetch historical stock data for the ticker specified in `config.py`.
*   Calculate and append technical indicators.
*   Preprocess the data for the LSTM model.
*   Build the LSTM model architecture.
*   Train the model using the training dataset.
*   Make predictions on both the training and test datasets.
*   Display plots comparing actual vs. predicted stock prices and the model's training/validation loss.
