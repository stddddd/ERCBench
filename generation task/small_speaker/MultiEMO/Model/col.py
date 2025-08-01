import scipy.stats
import numpy as np
import torch
import pickle

def KL(x, y):
	return scipy.stats.entropy(x, y)

def JS(x, y):
	x = np.array(x)
	y = np.array(y)
	z = (x+y)/2.0
	js = 0.5*KL(x,z)+0.5*KL(y,z)
	return js

def fairness(predictions):
	predictions = np.array(predictions)
	n_classes = predictions.shape[1]
	uni = [1 / n_classes for _ in range(n_classes)]
	ret = 0
	for prediction in predictions:
		ret += JS(prediction, uni)
	ret = ret / predictions.shape[0] * 100
	return  round(ret, 2)

def fairness2(predictions, label_ids):
	predictions = np.array(predictions)
	n_classes = predictions.shape[1]
	preds = np.argmax(predictions, axis=1)
	ret = 0
	ans = [torch.FloatTensor([0 for _ in range(n_classes)]) for _ in range(n_classes)]
	cnt = [0 for _ in range(n_classes)]
	for i in range(len(preds)):
		ans[label_ids[i]] += torch.FloatTensor(predictions[i])
		cnt[label_ids[i]] += 1
	for i in range(n_classes):
		ans[i] /= cnt[i]
		ret += JS(ans[i], [(x==i) for x in range(n_classes)])
	return  round(ret / n_classes * 100, 2)