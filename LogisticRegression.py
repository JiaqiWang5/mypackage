import numpy as np

class LogisricRegression:
	#def model(self):



	
	def costFunction(self,initial_theta,X,y,inital_lambda):
		'''
		cost function, returns the J value

		'''
		m = len(y)
		J = 0
    
		h = self.sigmoid(np.dot(X,initial_theta))    # computer the h(z)
		theta1 = initial_theta.copy()     
		theta1[0] = 0   
    
		temp = np.dot(np.transpose(theta1),theta1)
		J = (-np.dot(np.transpose(y),np.log(h))-np.dot(np.transpose(1-y),np.log(1-h))+temp*inital_lambda/2)/m   # cost equation
		return J


	def gradient(self,initial_theta,X,y,inital_lambda):
		'''
		gradient fucntion, returns the gradient

		'''
		m = len(y)
		grad = np.zeros((initial_theta.shape[0]))
    
		h = self.sigmoid(np.dot(X,initial_theta))# computer the h(z)
		theta1 = initial_theta.copy()
		theta1[0] = 0

		grad = np.dot(np.transpose(X),h-y)/m+inital_lambda/m*theta1 #normalize the gradient
		return grad

   
	def sigmoid(self,z):
		'''
		s function returns the value to make the output between 0 and 1
		'''
		h = np.zeros((len(z),1))  
    
		h = 1.0/(1.0+np.exp(-z))
		return h
