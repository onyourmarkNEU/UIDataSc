
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix
# from j48 import J48DecisionTree
from j48 import DecisionNode
from matplotlib import cm
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

global statistics_frame


def visualize_data():
    global df
    if df is None:
        messagebox.showerror("Error", "Please load a CSV file first.")
        return

    # preprocess categorical variables
    for col in df.select_dtypes(include='object').columns:
        df[col] = pd.factorize(df[col])[0]

    # create swarm plot with color encoding for categorical variable
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.swarmplot(data=df, x='class_variable',
                  y='numeric_variable', hue='class_variable', ax=ax)

    canvas = FigureCanvasTkAgg(fig, master=visualize_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)


def display_statistics(event):
    global variable_listbox, df, graph_canvas

    selected_variable = variable_listbox.get(variable_listbox.curselection())
    if selected_variable in df.columns:
        variable_stats = df[selected_variable].describe()
        statistics_text.delete("1.0", tk.END)
        statistics_text.insert(tk.END, str(variable_stats))

        # call display_graph function
        display_graph()


def display_graph():
    global variable_listbox, df, graph_canvas

    selected_variable = variable_listbox.get(variable_listbox.curselection())
    if selected_variable in df.columns:
        variable_data = df[selected_variable]

        fig, ax = plt.subplots(figsize=(5, 4))

        if variable_data.dtype in [int, float, 'int64', 'float64']:
            variable_data.hist(ax=ax)
        else:
            group_data = df.groupby(
                [selected_variable, df.columns[-1]]).size().unstack()
            group_data.plot(kind='bar', stacked=True, ax=ax)

        graph_canvas.figure = fig
        graph_canvas.draw()


def open_csv():
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file_path:
        try:
            global df
            global variable_listbox
            df = pd.read_csv(file_path, encoding="utf-8")
            # messagebox.showinfo("Success", "CSV file loaded successfully!")
            variable_listbox.delete(0, tk.END)
            for col in df.columns:
                variable_listbox.insert(tk.END, col)
        except Exception as e:
            messagebox.showerror(
                "Error", f"Failed to open CSV file. Error: {e}")


def run_algorithm(algorithm_name, result_text):
    # print("Result text:", result_text)

    if df is None:
        messagebox.showerror("Error", "Please load a CSV file first.")
        return

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    if algorithm_name == "Linear Regression":
        model = LinearRegression()
    elif algorithm_name == "Logistic Regression":
        model = LogisticRegression()
    elif algorithm_name == "J48 Decision Tree":
        model = J48DecisionTree(max_depth=10, min_samples_split=2)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        print("Accuracy:", accuracy)
        messagebox.showinfo(
            "Results", f"J48 Decision Tree Accuracy: {accuracy}")
        return

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    if algorithm_name == "Linear Regression":
        mse = mean_squared_error(y_test, predictions)
        equation = "y = "
        for i, coef in enumerate(model.coef_):
            equation += f"{coef:.2f} * x{i+1} + "
        equation += f"{model.intercept_:.2f}"

        # messagebox.showinfo("Results", f"Mean Squared Error: {mse}\nRegression Equation: {equation}")
    else:
        report = classification_report(y_test, predictions)
        cm = confusion_matrix(y_test, predictions)
        # messagebox.showinfo("Results", f"Classification Report:\n{report}\nConfusion Matrix:\n{cm}")

    result_text.delete("1.0", tk.END)
    if algorithm_name == "Linear Regression":
        result_text.insert(
            tk.END, f"Mean Squared Error: {mse}\nRegression Equation: {equation}")
    else:
        result_text.insert(
            tk.END, f"Classification Report:\n{report}\nConfusion Matrix:\n{cm}")


def visualize_data():
    global df
    if df is None:
        messagebox.showerror("Error", "Please load a CSV file first.")
        return

    # preprocess categorical variables
    for col in df.select_dtypes(include='object').columns:
        df[col] = pd.factorize(df[col])[0]

    # determine name of class variable
    class_var = df.columns[-1]

    # create dictionary mapping unique class values to colors
    color_dict = {}
    for cls in df[class_var].unique():
        color_dict[cls] = np.random.rand(3,)

    # create swarm plot with color encoding for class variable
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.swarmplot(data=df, x=class_var, y='numeric_variable',
                  palette=color_dict, ax=ax)

    canvas = FigureCanvasTkAgg(fig, master=visualize_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)


def main():
    global df, text, variable_listbox, statistics_text, graph_canvas, statistics_frame, result_text
    df = None

    root = tk.Tk()
    root.title("CSV Machine Learning")

    notebook = ttk.Notebook(root)
    notebook.pack(pady=10)

    preprocess_frame = ttk.Frame(notebook)
    classify_frame = ttk.Frame(notebook)
    visualize_frame = ttk.Frame(notebook)

    notebook.add(preprocess_frame, text="Preprocess")
    notebook.add(classify_frame, text="Classify")
    notebook.add(visualize_frame, text="Visualize")

    # Preprocess Frame
    preprocess_left_frame = ttk.Frame(preprocess_frame)
    preprocess_left_frame.pack(side=tk.LEFT, padx=10, pady=10)

    open_csv_button = ttk.Button(
        preprocess_left_frame, text="Open CSV", command=open_csv)
    open_csv_button.pack(side=tk.TOP, padx=10, pady=10)

    variables_frame = ttk.Frame(preprocess_left_frame)
    variables_frame.pack(side=tk.TOP, pady=10)

    variables_label = ttk.Label(variables_frame, text="Variables")
    variables_label.pack()

    variable_listbox = tk.Listbox(variables_frame, width=30, height=10)
    variable_listbox.pack(pady=10)
    variable_listbox.bind("<<ListboxSelect>>", display_statistics)

    preprocess_right_frame = ttk.Frame(preprocess_frame)
    preprocess_right_frame.pack(side=tk.RIGHT, padx=10, pady=10)

    statistics_frame = ttk.Frame(preprocess_right_frame)
    statistics_frame.pack(side=tk.TOP, pady=10)

    statistics_label = ttk.Label(statistics_frame, text="Statistics")
    statistics_label.pack()

    statistics_text = tk.Text(statistics_frame, width=50, height=8)
    statistics_text.pack()

    graph_frame = ttk.Frame(preprocess_right_frame)
    graph_frame.pack(side=tk.BOTTOM, pady=10)

    graph_label = ttk.Label(graph_frame, text="Graphs")
    graph_label.pack()

    graph_canvas = FigureCanvasTkAgg(plt.figure(), master=graph_frame)
    graph_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Classify Frame
    algorithm_frame = ttk.Frame(classify_frame)
    algorithm_frame.pack(side=tk.LEFT, padx=10, pady=10)

    algorithm_label = ttk.Label(algorithm_frame, text="Select Algorithm:")
    algorithm_label.pack(pady=5)

    algorithm_combobox = ttk.Combobox(algorithm_frame, values=[
                                      "Linear Regression", "Logistic Regression", "J48 Decision Tree"])
    algorithm_combobox.set("Linear Regression")
    algorithm_combobox.pack(pady=5)

    test_options_frame = ttk.Frame(classify_frame)
    test_options_frame.pack(side=tk.LEFT, padx=10, pady=10)

    test_options_label = ttk.Label(test_options_frame, text="Test Options:")
    test_options_label.pack(pady=5)

    test_options_frame_left = ttk.Frame(test_options_frame)
    test_options_frame_left.pack(side=tk.LEFT)

    test_options_frame_right = ttk.Frame(test_options_frame)
    test_options_frame_right.pack(side=tk.RIGHT)

    test_options_algorithm = ttk.Label(
        test_options_frame_left, text="Algorithm:")
    test_options_algorithm.pack(pady=5)

    test_options_algorithm_entry = ttk.Entry(
        test_options_frame_right, width=30)
    test_options_algorithm_entry.pack(pady=5)

    test_options_seed = ttk.Label(test_options_frame_left, text="Seed:")
    test_options_seed.pack(pady=5)

    test_options_seed_entry = ttk.Entry(test_options_frame_right, width=30)
    test_options_seed_entry.pack(pady=5)

    test_options_cross_validation = ttk.Label(
        test_options_frame_left, text="Cross Validation Folds:")
    test_options_cross_validation.pack(pady=5)

    test_options_cross_validation_entry = ttk.Entry(
        test_options_frame_right, width=30)
    test_options_cross_validation_entry.pack(pady=5)

    test_options_percentage_split = ttk.Label(
        test_options_frame_left, text="Percentage Split:")
    test_options_percentage_split.pack(pady=5)

    test_options_percentage_split_entry = ttk.Entry(
        test_options_frame_right, width=30)
    test_options_percentage_split_entry.pack(pady=5)

    run_button = ttk.Button(classify_frame, text="Run Algorithm", command=lambda: run_algorithm(
        algorithm_combobox.get(), result_text))
    run_button.pack(side=tk.BOTTOM, pady=10)

    result_frame = ttk.Frame(classify_frame)
    result_frame.pack(side=tk.RIGHT, padx=10, pady=10)

    result_label = ttk.Label(result_frame, text="Results:")
    result_label.pack(pady=5)

    result_text = tk.Text(result_frame, width=50, height=8)
    result_text.pack(pady=10)

    # Visualize Frame
    visualize_button = ttk.Button(
        visualize_frame, text="Visualize Data", command=visualize_data)
    visualize_button.pack(pady=10)

    root.mainloop()


if __name__ == "__main__":
    main()
