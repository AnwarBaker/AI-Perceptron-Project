import os
import math
import random
import pickle
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

#configuration
DEFAULT_DATASET = "weather_data_linearly_separable.xlsx"
DEFAULT_MODEL = "perceptron_model.pkl"
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def detect_columns(df):

    cols = [c.strip() for c in df.columns]
    lc = [c.lower() for c in cols]


    temp_col = None
    hum_col = None
    wind_col = None
    label_col = None

    for i, name in enumerate(lc):
        if temp_col is None and ("temp" in name or "temperature" in name):
            temp_col = cols[i]
        if hum_col is None and ("hum" in name or "humidity" in name):
            hum_col = cols[i]
        if wind_col is None and ("wind" in name or "wind_speed" in name or "wind speed" in name):
            wind_col = cols[i]
        if label_col is None and ("safe" in name and "fly" in name) or ("safetofly" in name) or (name == "label"):
            label_col = cols[i]

    # fallback heuristics
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if temp_col is None and len(numeric_cols) >= 1:
        temp_col = numeric_cols[0]
    if hum_col is None and len(numeric_cols) >= 2:
        hum_col = numeric_cols[1]
    if wind_col is None and len(numeric_cols) >= 3:
        wind_col = numeric_cols[2]
    if label_col is None and len(numeric_cols) >= 4:
        label_col = numeric_cols[-1]

    return [temp_col, hum_col, wind_col], label_col


def train_and_save_perceptron(excel_path=DEFAULT_DATASET, model_path=DEFAULT_MODEL, force_retrain=False):

    model_path = Path(model_path)
    dataset_path = Path(excel_path)

    if model_path.exists() and not force_retrain:
        try:
            with open(model_path, "rb") as f:
                data = pickle.load(f)
            print(f"Loaded existing model from {model_path}")
            return data
        except Exception as e:
            print("Failed to load existing model:", e)

    if dataset_path.exists():
        print("Loading dataset:", dataset_path)
        df = pd.read_excel(dataset_path)
        feature_cols, label_col = detect_columns(df)
        if None in feature_cols or label_col is None:
            raise ValueError("Couldn't detect columns in dataset. Check your Excel headers.")
        X = df[list(feature_cols)].values
        y = df[label_col].values
        print(f"Detected feature columns: {feature_cols}, label: {label_col}")
    else:
        print("Dataset not found at", dataset_path)
        print("Generating synthetic linearly separable weather dataset for training.")
        n = 200
        temp = np.random.uniform(-5, 40, n)
        hum = np.random.uniform(10, 100, n)
        wind = np.random.uniform(0, 60, n)
        y = ((wind > 30) | ((hum > 80) & ((temp < 0) | (temp > 35)))).astype(int)
        X = np.vstack([temp, hum, wind]).T
        feature_cols = ["Temperature", "Humidity", "WindSpeed"]
        print("Synthetic data created.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = Perceptron(max_iter=1000, tol=1e-3, random_state=RANDOM_SEED)
    clf.fit(X_scaled, y)

    data = {"model": clf, "scaler": scaler, "feature_columns": feature_cols}
    with open(model_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Trained Perceptron saved to {model_path}")
    return data


def load_model(model_path=DEFAULT_MODEL):
    model_path = Path(model_path)
    if not model_path.exists():
        return None
    with open(model_path, "rb") as f:
        return pickle.load(f)


def predict_safe_flags(model_data, cities):
    if model_data is None:
        raise ValueError("Model data is None.")
    X = np.array([[c["temp"], c["hum"], c["wind"]] for c in cities])
    Xs = model_data["scaler"].transform(X)
    preds = model_data["model"].predict(Xs)
    return [bool(int(p)) for p in preds]

def euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def total_route_cost(route, coords, unsafe_flags, penalty=50):
    total = 0.0
    n = len(route)
    for i in range(n):
        a = coords[route[i]]
        b = coords[route[(i + 1) % n]]
        total += euclidean(a, b)
    # add penalty per unsafe city (or could add per visit; here each city visited once)
    for idx in route:
        if unsafe_flags[idx]:
            total += penalty
    return total


def simulated_annealing(coords, unsafe_flags, penalty=50, T0=1000.0, cooling_rate=0.995, iter_per_temp=200, seed=None, max_steps=None):

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    n = len(coords)
    if n <= 1:
        return list(range(n)), 0.0

    # initial route
    curr_route = list(range(n))
    random.shuffle(curr_route)
    curr_cost = total_route_cost(curr_route, coords, unsafe_flags, penalty)
    best_route = curr_route[:]
    best_cost = curr_cost
    T = T0
    steps = 0

    while T > 1e-3:
        for _ in range(iter_per_temp):
            # propose new route by swapping two nodes
            i, j = random.sample(range(n), 2)
            new_route = curr_route[:]
            new_route[i], new_route[j] = new_route[j], new_route[i]
            new_cost = total_route_cost(new_route, coords, unsafe_flags, penalty)
            delta = new_cost - curr_cost
            if delta < 0 or math.exp(-delta / T) > random.random():
                curr_route = new_route
                curr_cost = new_cost
                if curr_cost < best_cost:
                    best_cost = curr_cost
                    best_route = curr_route[:]
            steps += 1
            if max_steps and steps >= max_steps:
                break
        if max_steps and steps >= max_steps:
            break
        T *= cooling_rate
    return best_route, best_cost


class DronePlannerApp(tk.Tk):
    def __init__(self, model_data):
        super().__init__()
        self.title("Intelligent Delivery Drone Planner")
        self.geometry("1150x720")
        self.model_data = model_data
        # cities: list of dicts {'x','y','temp','hum','wind','unsafe'}
        self.cities = []
        self.penalty = 50.0
        self._last_initial = None
        self._last_optimized = None
        self._create_widgets()
        self._draw_empty_plot()

    def _create_widgets(self):
        # left control frame
        control = ttk.Frame(self)
        control.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)

        # Dataset & model controls
        ttk.Label(control, text="ML Model").pack(anchor="w")
        model_frame = ttk.Frame(control)
        model_frame.pack(fill=tk.X, pady=4)
        ttk.Button(model_frame, text="Load Model...", command=self._load_model_file).pack(fill=tk.X)
        ttk.Button(model_frame, text="Train / (Re)train Model", command=self._train_model_dialog).pack(fill=tk.X, pady=4)
        ttk.Button(model_frame, text="Show Model Info", command=self._show_model_info).pack(fill=tk.X)

        ttk.Separator(control).pack(fill=tk.X, pady=6)

        # Cities generation
        ttk.Label(control, text="Cities").pack(anchor="w")
        lbl = ttk.Label(control, text="Number of cities:")
        lbl.pack(anchor="w", pady=(6, 0))
        self.num_entry = ttk.Entry(control)
        self.num_entry.insert(0, "10")
        self.num_entry.pack(fill=tk.X)

        ttk.Button(control, text="Generate Random Cities", command=self._generate_random).pack(fill=tk.X, pady=4)
        ttk.Button(control, text="Edit Cities Table", command=self._open_edit_window).pack(fill=tk.X, pady=4)

        ttk.Button(control, text="Predict Safe/Unsafe (Perceptron)", command=self._predict_safety).pack(fill=tk.X, pady=4)

        ttk.Separator(control).pack(fill=tk.X, pady=6)

        # SA parameters
        ttk.Label(control, text="Simulated Annealing").pack(anchor="w")
        ttk.Label(control, text="Initial Temperature").pack(anchor="w")
        self.temp_entry = ttk.Entry(control)
        self.temp_entry.insert(0, "1000")
        self.temp_entry.pack(fill=tk.X)

        ttk.Label(control, text="Cooling Rate (0..1)").pack(anchor="w")
        self.cool_entry = ttk.Entry(control)
        self.cool_entry.insert(0, "0.995")
        self.cool_entry.pack(fill=tk.X)

        ttk.Label(control, text="Iterations per Temp").pack(anchor="w")
        self.iter_entry = ttk.Entry(control)
        self.iter_entry.insert(0, "200")
        self.iter_entry.pack(fill=tk.X)

        ttk.Label(control, text="Penalty for unsafe city").pack(anchor="w", pady=(6, 0))
        self.pen_entry = ttk.Entry(control)
        self.pen_entry.insert(0, "50")
        self.pen_entry.pack(fill=tk.X)

        ttk.Button(control, text="Show Initial Route", command=self.show_initial_route).pack(fill=tk.X, pady=6)
        ttk.Button(control, text="Run Simulated Annealing", command=self._run_sa).pack(fill=tk.X, pady=4)
        ttk.Button(control, text="Show Optimized Route", command=self.show_optimized_route).pack(fill=tk.X, pady=4)

        ttk.Separator(control).pack(fill=tk.X, pady=6)

        ttk.Button(control, text="Export Route as CSV", command=self._export_route_csv).pack(fill=tk.X, pady=(10, 2))

        # Right frame: plot + info
        right = ttk.Frame(self)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.info_label = ttk.Label(right, text="Status: Ready")
        self.info_label.pack(anchor="w", pady=(4, 0))

    def _load_model_file(self):
        path = filedialog.askopenfilename(title="Select model pickle", filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")])
        if path:
            try:
                with open(path, "rb") as f:
                    self.model_data = pickle.load(f)
                self.info_label.config(text=f"Loaded model: {os.path.basename(path)}")
            except Exception as e:
                messagebox.showerror("Load error", f"Failed to load model: {e}")

    def _train_model_dialog(self):
        # allow user to pick dataset or use default/synthetic
        res = messagebox.askquestion("Train model", "Do you want to load training data from an Excel file? (Yes = choose file, No = use synthetic)")
        if res == "yes":
            path = filedialog.askopenfilename(title="Select Excel dataset", filetypes=[("Excel files", "*.xlsx;*.xls"), ("All files", "*.*")])
            if not path:
                return
            # ask where to save model
            save_path = filedialog.asksaveasfilename(defaultextension=".pkl", initialfile=DEFAULT_MODEL, title="Save model to")
            if not save_path:
                return
            try:
                data = train_and_save_perceptron(excel_path=path, model_path=save_path, force_retrain=True)
                self.model_data = data
                messagebox.showinfo("Trained", f"Model trained and saved to {save_path}")
                self.info_label.config(text=f"Trained model saved: {os.path.basename(save_path)}")
            except Exception as e:
                messagebox.showerror("Training error", f"Failed to train: {e}")
        else:
            save_path = filedialog.asksaveasfilename(defaultextension=".pkl", initialfile=DEFAULT_MODEL, title="Save model to")
            if not save_path:
                return
            data = train_and_save_perceptron(excel_path="__none__", model_path=save_path, force_retrain=True)
            self.model_data = data
            messagebox.showinfo("Trained", f"Synthetic model trained and saved to {save_path}")
            self.info_label.config(text=f"Trained model saved: {os.path.basename(save_path)}")

    def _show_model_info(self):
        if not self.model_data:
            messagebox.showinfo("Model", "No model loaded.")
            return
        cols = self.model_data.get("feature_columns", ["?"])
        msg = f"Model loaded. Feature columns: {cols}\nPerceptron params: {self.model_data['model']}"
        messagebox.showinfo("Model info", msg)

    def _generate_random(self):
        try:
            n = int(self.num_entry.get())
            if n <= 0:
                raise ValueError("n must be > 0")
        except Exception:
            messagebox.showerror("Input error", "Invalid number of cities.")
            return
        self.cities = []
        for i in range(n):
            x = random.uniform(0, 100)
            y = random.uniform(0, 100)
            temp = random.uniform(-5, 40)
            hum = random.uniform(10, 100)
            wind = random.uniform(0, 60)
            self.cities.append({"x": x, "y": y, "temp": temp, "hum": hum, "wind": wind, "unsafe": False})
        self.info_label.config(text=f"Generated {n} random cities.")
        self.show_initial_route()

    def _open_edit_window(self):
        if not self.cities:
            messagebox.showinfo("No cities", "Generate cities first.")
            return
        win = tk.Toplevel(self)
        win.title("Edit Cities")
        frame = ttk.Frame(win)
        frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        canvas = tk.Canvas(frame)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.configure(yscrollcommand=scrollbar.set)
        inner = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=inner, anchor="nw")

        entries = []
        header = ["id", "x", "y", "temp", "hum", "wind", "unsafe"]
        for j, h in enumerate(header):
            ttk.Label(inner, text=h).grid(row=0, column=j, padx=2, pady=2)

        for i, c in enumerate(self.cities, start=1):
            ttk.Label(inner, text=str(i-1)).grid(row=i, column=0)
            xent = ttk.Entry(inner, width=8); xent.insert(0, f"{c['x']:.3f}"); xent.grid(row=i, column=1)
            yent = ttk.Entry(inner, width=8); yent.insert(0, f"{c['y']:.3f}"); yent.grid(row=i, column=2)
            tent = ttk.Entry(inner, width=8); tent.insert(0, f"{c['temp']:.2f}"); tent.grid(row=i, column=3)
            hent = ttk.Entry(inner, width=8); hent.insert(0, f"{c['hum']:.2f}"); hent.grid(row=i, column=4)
            went = ttk.Entry(inner, width=8); went.insert(0, f"{c['wind']:.2f}"); went.grid(row=i, column=5)
            var = tk.IntVar(value=1 if c.get("unsafe") else 0)
            chk = ttk.Checkbutton(inner, variable=var, text="")
            chk.grid(row=i, column=6)
            entries.append((xent, yent, tent, hent, went, var))

        def save_and_close():
            for idx, ent in enumerate(entries):
                try:
                    x = float(ent[0].get()); y = float(ent[1].get())
                    temp = float(ent[2].get()); hum = float(ent[3].get()); wind = float(ent[4].get())
                    unsafe = bool(ent[5].get())
                except Exception as e:
                    messagebox.showerror("Input error", f"Invalid values on row {idx}.")
                    return
                self.cities[idx].update({"x": x, "y": y, "temp": temp, "hum": hum, "wind": wind, "unsafe": unsafe})
            win.destroy()
            self.info_label.config(text="Cities updated.")
            self.show_initial_route()

        ttk.Button(win, text="Save", command=save_and_close).pack(side=tk.BOTTOM, pady=6)

        def on_config(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        inner.bind("<Configure>", on_config)

    def _predict_safety(self):
        if not self.cities:
            messagebox.showinfo("No cities", "Generate cities first.")
            return
        if not self.model_data:
            messagebox.showerror("No model", "No Perceptron model loaded or trained.")
            return
        try:
            flags = predict_safe_flags(self.model_data, self.cities)
            for i, f in enumerate(flags):
                # Perceptron label: 1 = Unsafe (as defined in project)
                self.cities[i]["unsafe"] = bool(int(f))
            self.info_label.config(text="Predicted safety for all cities.")
            self.show_initial_route()
        except Exception as e:
            messagebox.showerror("Prediction error", f"Failed to predict: {e}")

    def show_initial_route(self):
        if not self.cities:
            messagebox.showinfo("No cities", "Generate cities first.")
            return
        coords = [(c["x"], c["y"]) for c in self.cities]
        unsafe = [c["unsafe"] for c in self.cities]
        route = list(range(len(coords)))
        cost = total_route_cost(route, coords, unsafe, penalty=self.penalty)
        self._last_initial = {"route": route, "cost": cost, "coords": coords, "unsafe": unsafe}
        self._last_optimized = None
        self._plot_route(route, coords, unsafe, title=f"Initial route — cost: {cost:.2f}")

    def _run_sa(self):
        if not self.cities:
            messagebox.showinfo("No cities", "Generate cities first.")
            return
        try:
            T0 = float(self.temp_entry.get())
            cooling = float(self.cool_entry.get())
            iters = int(self.iter_entry.get())
            pen = float(self.pen_entry.get())
            self.penalty = pen
        except Exception:
            messagebox.showerror("Input error", "Invalid SA parameters.")
            return
        coords = [(c["x"], c["y"]) for c in self.cities]
        unsafe = [c["unsafe"] for c in self.cities]
        best_route, best_cost = simulated_annealing(coords, unsafe, penalty=pen, T0=T0, cooling_rate=cooling, iter_per_temp=iters, seed=RANDOM_SEED)
        self._last_optimized = {"route": best_route, "cost": best_cost, "coords": coords, "unsafe": unsafe}
        self.info_label.config(text=f"Optimized route cost: {best_cost:.2f}")
        self._plot_route(best_route, coords, unsafe, title=f"Optimized route — cost: {best_cost:.2f}")

    def show_optimized_route(self):
        if not self._last_optimized:
            messagebox.showinfo("No optimization", "Run Simulated Annealing first.")
            return
        r = self._last_optimized
        self._plot_route(r["route"], r["coords"], r["unsafe"], title=f"Optimized route — cost: {r['cost']:.2f}")

    def _plot_route(self, route, coords, unsafe, title="Route"):
        self.ax.clear()
        if not coords:
            self._draw_empty_plot()
            return
        xs = [coords[i][0] for i in route] + [coords[route[0]][0]]
        ys = [coords[i][1] for i in route] + [coords[route[0]][1]]
        self.ax.plot(xs, ys, linestyle="-", marker="o", linewidth=1.5)
        for i, (x, y) in enumerate(coords):
            color = "red" if unsafe[i] else "green"
            self.ax.scatter([x], [y], s=40)
            self.ax.annotate(f"{i}", (x + 0.5, y + 0.5), fontsize=9)
            circ = plt.Circle((x, y), 1.5, fill=False, edgecolor=color, linewidth=1.2)
            self.ax.add_patch(circ)
        self.ax.set_title(title)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()

    def _draw_empty_plot(self):
        self.ax.clear()
        self.ax.set_title("Route display")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.canvas.draw()

    def _export_route_csv(self):
        if not self._last_optimized and not self._last_initial:
            messagebox.showinfo("No data", "No route to export. Create cities and run SA or show initial route.")
            return
        which = "optimized" if self._last_optimized else "initial"
        data = self._last_optimized if which == "optimized" else self._last_initial
        route = data["route"]
        coords = data["coords"]
        unsafe = data["unsafe"]
        path = filedialog.asksaveasfilename(defaultextension=".csv", title="Save route CSV", initialfile=f"{which}_route.csv")
        if not path:
            return
        rows = []
        for idx in route:
            x, y = coords[idx]
            rows.append({"city_index": idx, "x": x, "y": y, "unsafe": int(unsafe[idx])})
        with open(path, "w") as f:
            f.write("city_index,x,y,unsafe\n")
            for r in rows:
                f.write(f"{r['city_index']},{r['x']},{r['y']},{r['unsafe']}\n")
        messagebox.showinfo("Saved", f"Route exported to {path}")
        self.info_label.config(text=f"Route exported to {os.path.basename(path)}")
def main():
    model_data = None
    if os.path.exists(DEFAULT_MODEL):
        try:
            model_data = load_model(DEFAULT_MODEL)
            print("Loaded model:", DEFAULT_MODEL)
        except Exception as e:
            print("Error loading model:", e)

    if model_data is None:
        model_data = train_and_save_perceptron(excel_path=DEFAULT_DATASET, model_path=DEFAULT_MODEL)
    app = DronePlannerApp(model_data)
    app.mainloop()


if __name__ == "__main__":
    main()
