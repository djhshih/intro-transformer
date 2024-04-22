import torch
from tqdm import tqdm

def train(model, optim, data_train, data_test, config, n_epochs):
	losses_train = []
	losses_test = []

	for epoch in tqdm(range(n_epochs)):
		# iterate through training data
		model.train()
		for step in range(len(data_train) // config.batch_size):
			idx = [i for i in
				torch.randint(len(data_train), (config.batch_size,))]
			X, Y = data_train[idx]
			# evaluate loss
			logits, loss = model(X, Y)
			# backpropagation of errors
			loss.backward()
			# update weights
			optim.step()
			# clear gradient
			optim.zero_grad(set_to_none=True)
			losses_train.append(loss.item())
		# evaluate test loss
		model.eval()
		X, Y = data_test[:]
		logits, loss_test = model(X, Y)
		losses_test.append(loss_test.item())

	return model, losses_train, losses_test

