import matplotlib.pyplot as plt
import time

'''
params:
    c : if True the return value contain color
    m : marker
    l : line style

return:
      a generator to produce different color, marker and line style for matplotlib.pyplot.plot. when the default color, marker and line style runs up (totally 273), it will loop from the beginning.
'''

def color_marker_line(c = True, m = True, l = True):
    colors = list('bgrcmy')
    markers = list('.o^>v<sp*+xDd1234Hh|_')
    lines = '- -- : -.'.split(' ')
    cmls = [' ']
    while(1):
        if c:
            cmls = colors
        if m:
            cmls = [(i + j).strip() for i in cmls for j in markers]
        if l:
            cmls = list([(i + j).strip() for i in cmls for j in lines])
        for cml in cmls:
            yield cml
        print('the cml list has been used up! loop it from the beginning!')

if __name__ == '__main__':
    i = 0
    cml = color_marker_line()
    for j in cml:
        print(type(j))
        plt.plot([1,2],[3,4],j)
        i += 1
        print(i)
        if i > 500:
            break
    plt.show()
