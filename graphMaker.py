import matplotlib.pyplot as plt

generations = [0,5,10,15,20,25,30,35,40,45,50]
breastCancerAcc = [31.64, 64.2, 78.4, 83.13, 86.23, 89.76, 90.91, 91.45, 92.8, 93.45, 95.81]
creditCardAcc = [21.24,78.26,78.26,78.26,78.26,78.41,78.41,78.41,78.41,78.76,78.76]

plt.plot(generations, breastCancerAcc)
plt.xlabel("Number of generations")
plt.ylabel("Testing accuracy (%)")
plt.show()
plt.plot(generations, creditCardAcc)
plt.xlabel("Number of generations")
plt.ylabel("Testing accuracy (%)")
plt.show()