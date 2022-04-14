# https://blog.csdn.net/weixin_41561539/article/details/94294828
from pymouse import PyMouse
import time

#(85,47)
if __name__ == '__main__':
    mouse = PyMouse()
    
    #position = mouse.position()    #获取当前坐标的位置
    #mouse.move(600, 250)   #鼠标移动到(x,y)位置
    #print(position)
    #525,239 942,633
    while(True):
        x,y = mouse.position()
        print(x,y)
        #mouse.click(x,y)
        #mouse.click(x,y)
        time.sleep(1)
        #mouse.click(85,47)  #移动并且在(x,y)位置左击
