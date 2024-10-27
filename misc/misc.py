import matplotlib
import platform
if platform.system() == 'Darwin':
    matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
plt.ion()
plt.plot(range(10))
plt.show()
