import pandas as pd
import matplotlib.pyplot as pl

# =======================================
# MAKE SURE TO ADD "ep" to header of csv
# =======================================

csv_filename = "../../real_DQN_boxed_water_medium_11-30-2019-213120.csv"
label = "DQN"
out_path = "out/"

graphing = {
	'% time spent exploring': ["Time", "upper right", "Figure 7. Percentage of Time Spent Exploring"],
	'delta_t': ["Steps", "upper left", "Figure 8. Number of Steps per Episode"],
	'mean episode reward': ["Mean Reward", "lower left", "Figure 9. Mean Episode Reward"],
	'reward for this episode': ["Reward", "lower left", "Figure 10. Reward per Episode"],
}

def plot_stats(csv_filename, label, out_path):
	df = pd.read_csv(csv_filename)
	eps = df['ep'].values.tolist()

	for key, val in graphing.items():
		toPlot = df[[key]].values.tolist()

		pl.plot(eps, toPlot, 'b', label = label)
		pl.xlabel("Episodes", fontsize=18)
		pl.ylabel(val[0], fontsize=16)
		pl.legend(loc=val[1], fontsize=16)
		pl.title(val[2], fontsize=20)
		pl.savefig(out_path + "_" + val[0] + ".png")
		pl.close('all')

plot_stats(csv_filename, label, out_path)
