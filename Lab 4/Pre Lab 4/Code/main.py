import matplotlib.pyplot as plt

class1 = [(1, 1), (-1, -1)]
class2 = [(1, -1), (-1, 1)]

plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='black', linewidth=1)
plt.grid(True)
plt.scatter(*zip(*class1), c='r', label='class 1', s=100)
plt.scatter(*zip(*class2), c='b', label='class -1', s=100)
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter plot of two classes')
plt.legend(loc='right')

# Save the plots
plt.savefig('data_points.png')

plt.show()
