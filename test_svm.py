import numpy as np
import svm
import kernel as k

# Test AND gate
clsfyr = svm.train([[1,1],[1,-1],[-1,1],[-1,-1]], [1,-1,-1,-1], k.linear)
# should be [0,0,0,1,1,0,0,0]
print("classified: " + str(svm.classify(clsfyr, [[-1,-1],[1,-1],[-1,1],[1,1],[1,1],[1,-1],[-1,-1],[-1,1]])))
print("\n\n\n\n\n")

X = np.array([[1.0,0.0], [2.0,0.0], [3.0,0.0], [-1.0,0.0], [-2.0,0.0], [-3.0,0.0]])
y = np.array([[1.0], [1.0], [1.0], [-1.0], [-1.0], [-1.0]])
clsfyr = svm.train(X, y, k.linear)
print("classified: " + str(svm.classify(clsfyr, X)))

