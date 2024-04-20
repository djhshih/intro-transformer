from plotnine import ggplot, geom_line, aes, theme_classic, scale_color_discrete
import pandas as pd

def plot_loss(losses_train, losses_test):
	epochs = [i for i in range(1, len(losses_test)+1)]
	epochs_train = [i * len(losses_test) / len(losses_train) for i in range(1, len(losses_train)+1)]

	losses_pd = pd.concat([
		pd.DataFrame({
			'epoch': epochs_train,
			'loss': losses_train,
			'type': 'train'
		}),
		pd.DataFrame({
			'epoch': epochs,
			'loss': losses_test,
			'type': 'test'
		}),
	])

	return (ggplot(losses_pd, aes(x='epoch', y='loss', colour='type')) + 
		geom_line(alpha=0.7) +
		theme_classic())

