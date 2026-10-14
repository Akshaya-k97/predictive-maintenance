# Dataset setup

This project uses the **AI4I 2020 Predictive Maintenance Dataset** (UCI Machine Learning Repository):
https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

1. Open the link above and download the dataset (a ZIP containing `ai4i2020.csv`).
2. Extract it and place the file here as `data/ai4i2020.csv`.
3. Check licensing/citation terms on the UCI page before redistributing. The CSV is **not** committed to
   Git (see `.gitignore`); each user downloads it locally.

Expected columns: `UDI, Product ID, Type, Air temperature [K], Process temperature [K], Rotational speed [rpm],
Torque [Nm], Tool wear [min], Machine failure, TWF, HDF, PWF, OSF, RNF`.
`src/data_preprocessing.py` verifies these and stops with a clear message if any is missing.
The dataset is synthetic (generated to mimic real industrial data), which is stated in its UCI description.
