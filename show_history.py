import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('A4C/core_3_episode_990_history.csv')


plt.plot(df.iloc[:, 0], df['rewards'])
plt.waitforbuttonpress()