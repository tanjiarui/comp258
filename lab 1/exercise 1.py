from sklearn.linear_model import Perceptron
from adaline import *


def padding(letter, pad):
	for index in pad:
		letter[index] = 1
	return letter


# generate dataset
letters = []
pixel = np.array([-1 for i in range(63)])
letters.append(padding(pixel.copy(), [2, 3, 10, 17, 23, 25, 30, 32, 36, 37, 38, 39, 40, 43, 47, 50, 54, 56, 57, 58, 60, 61, 62]))
letters.append(padding(pixel.copy(), [0, 1, 2, 3, 4, 5, 8, 13, 15, 20, 22, 27, 29, 30, 31, 32, 33, 36, 41, 43, 48, 50, 55, 56, 57, 58, 59, 60, 61]))
letters.append(padding(pixel.copy(), [2, 3, 4, 5, 6, 8, 13, 14, 21, 28, 35, 42, 50, 55, 58, 59, 60, 61]))
letters.append(padding(pixel.copy(), [0, 1, 2, 3, 4, 8, 12, 15, 20, 22, 27, 29, 34, 36, 41, 43, 48, 50, 54, 56, 57, 58, 59, 60]))
letters.append(padding(pixel.copy(), [0, 1, 2, 3, 4, 5, 6, 8, 13, 15, 22, 24, 29, 30, 31, 36, 38, 43, 50, 55, 56, 57, 58, 59, 60, 61, 62]))
letters.append(padding(pixel.copy(), [3, 4, 5, 6, 12, 19, 26, 33, 40, 43, 47, 50, 54, 58, 59, 60]))
letters.append(padding(pixel.copy(), [0, 1, 2, 5, 6, 8, 11, 15, 17, 22, 23, 29, 30, 36, 38, 43, 46, 50, 54, 56, 57, 58, 61, 62]))
letters.append(padding(pixel.copy(), [3, 10, 17, 23, 25, 30, 32, 36, 40, 43, 44, 45, 46, 47, 50, 54, 57, 61]))
letters.append(padding(pixel.copy(), [0, 1, 2, 3, 4, 5, 7, 13, 14, 20, 21, 27, 28, 29, 30, 31, 32, 33, 35, 41, 42, 48, 49, 55, 56, 57, 58, 59, 60, 61]))
letters.append(padding(pixel.copy(), [2, 3, 4, 8, 12, 14, 20, 21, 28, 35, 42, 48, 50, 54, 58, 59, 60]))
letters.append(padding(pixel.copy(), [0, 1, 2, 3, 4, 7, 12, 14, 20, 21, 27, 28, 34, 35, 41, 42, 48, 49, 54, 56, 57, 58, 59, 60]))
letters.append(padding(pixel.copy(), [0, 1, 2, 3, 4, 5, 6, 7, 14, 21, 28, 29, 30, 31, 32, 35, 42, 49, 56, 57, 58, 59, 60, 61, 62]))
letters.append(padding(pixel.copy(), [5, 12, 19, 26, 33, 40, 43, 47, 50, 54, 58, 59, 60]))
letters.append(padding(pixel.copy(), [0, 7, 14, 21, 28, 35, 42, 49, 56, 5, 11, 17, 23, 29, 37, 45, 53, 61]))
letters.append(padding(pixel.copy(), [3, 10, 16, 18, 23, 25, 29, 33, 36, 37, 38, 39, 40, 42, 48, 49, 55, 56, 57, 61, 62]))
letters.append(padding(pixel.copy(), [0, 1, 2, 3, 4, 5, 8, 13, 15, 20, 22, 23, 24, 25, 26, 29, 34, 36, 41, 43, 48, 50, 55, 56, 57, 58, 59, 60, 61]))
letters.append(padding(pixel.copy(), [2, 3, 4, 6, 8, 12, 13, 14, 20, 21, 28, 35, 42, 48, 50, 54, 58, 59, 60]))
letters.append(padding(pixel.copy(), [0, 1, 2, 3, 4, 8, 12, 15, 20, 22, 27, 29, 34, 36, 41, 43, 48, 50, 54, 56, 57, 58, 59, 60]))
letters.append(padding(pixel.copy(), [0, 1, 2, 3, 4, 5, 6, 8, 13, 15, 18, 22, 23, 24, 25, 29, 32, 36, 43, 50, 55, 56, 57, 58, 59, 60, 61, 62]))
letters.append(padding(pixel.copy(), [4, 5, 6, 12, 19, 26, 33, 40, 47, 50, 54, 58, 59, 60]))
letters.append(padding(pixel.copy(), [0, 1, 2, 5, 6, 8, 12, 15, 18, 22, 24, 29, 30, 36, 38, 43, 46, 50, 54, 56, 57, 58, 61, 62]))
letters = np.array(letters)
target = [-1 for j in range(21)]
target[3], target[10], target[17] = 1, 1, 1

model = Perceptron(eta0=0.01)
model.fit(letters, target)
print(model.score(letters, target))  # 1
model = AdalineGD()
model.fit(letters, target)
print(model.score(letters, target))  # .86