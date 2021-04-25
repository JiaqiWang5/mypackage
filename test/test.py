from ..LinearRegression import LinearRegression


if __name__ == "__main__":
	model=LinearRegression()
	mu,sigma,theta = model.model("data.csv",0.01,400)
	print(mu)
	print(sigma)
	print(theta)
	print(model.predict(mu,sigma,theta))