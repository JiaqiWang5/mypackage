import numpy as np
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties

class LinearRegression:
	
	def model(self,filePath,alpha=0.01,num_iters=500):
		"""
		The linear regression model
		input the dataset, alpha and number of iterations
		return the parameters after training

		"""
		data = self.loadtxtAndcsv_data(filePath,",",np.float64)#read the data from the given file path
		X = data[:,0:-1]                   
		y = data[:,-1]    
		m = len(y)         
		col = data.shape[1]  
		X,mu,sigma = self.featureNormaliza(X)    # normalize
		self.plot_X1_X2(X)         # plot the normalized graph
		X = np.hstack((np.ones((m,1)),X))

		print("implementing decent gradient algorithm")

		theta = np.zeros((col,1))
		y = y.reshape(-1,1)
		theta,J_history = self.gradientDescent(X, y, theta, alpha, num_iters)
		self.plotJ(J_history, num_iters)
		return mu,sigma,theta

    # read file functions
	def loadtxtAndcsv_data(self,fileName,split,dataType):
		return np.loadtxt(fileName,delimiter=split,dtype=dataType)

	def loadnpy_data(self,fileName):
		return np.load(fileName)
    
	
	def featureNormaliza(self,X):
		"""
		normalized function
		imput array
		return the normalized array
		"""
		X_norm=np.array(X)
		mu=np.zeros((1,X.shape[1]))
		sigma=np.zeros((1,X.shape[1]))
		mu=np.mean(X_norm,0)
		sigma=np.std(X_norm,0)
		for i in range(X.shape[1]):
			X_norm[:,i]=(X_norm[:,i]-mu[i])/sigma[i]
		return X_norm,mu,sigma

	# gradien descent
	def gradientDescent(self,X,y,theta,alpha,num_iters):
		'''
		gradient descent algorithm
		input theta,alpha,number if iterations
		returns the theta
		'''
		m = len(y)      
		n = len(theta)
    
		temp = np.matrix(np.zeros((n,num_iters)))
    
    
		J_history = np.zeros((num_iters,1))
    
		for i in range(num_iters):    
			h = np.dot(X,theta) 
			temp[:,i] = theta - ((alpha/m)*(np.dot(np.transpose(X),h-y))) 
			theta = temp[:,i]
			J_history[i] = self.computerCost(X,y,theta) 
			print('.', end=' ')      
		return theta,J_history  

	# cost function
	def computerCost(self,X,y,theta):
		m = len(y)
		J = 0
    
		J = (np.transpose(X*theta-y))*(X*theta-y)/(2*m)
		return J


	#plot functions
	def plot_X1_X2(self,X):
		plt.scatter(X[:,0],X[:,1])
		plt.show()

	def plotJ(self,J_history,num_iters):
		x = np.arange(1,num_iters+1)
		plt.plot(x,J_history)
		plt.xlabel(u"number of iterations")
		plt.ylabel(u"cost value")
		plt.title(u"cost and the number of iterations")
		plt.show()

	# predict function
	def predict(self,mu,sigma,theta):
		'''
		predict function predict the final result according to the parameters
		input mu, sigma, theta
		returns the final result

		'''
		result = 0
		predict = np.array([1650,3])
		norm_predict = (predict-mu)/sigma
		final_predict = np.hstack((np.ones((1)),norm_predict))
		result = np.dot(final_predict,theta) 
		return result

