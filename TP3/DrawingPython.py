import Tkinter as tk
import numpy
import sys

class ImageGenerator:
    def __init__(self,parent,posx,posy,*kwargs):
        self.parent = parent
        self.posx = posx
        self.posy = posy
        self.sizex = 280
        self.sizey = 280
        self.b1 = "up"
        self.xold = None
        self.yold = None 
        self.drawing_area=tk.Canvas(self.parent,width=self.sizex,height=self.sizey)
        self.drawing_area.place(x=self.posx,y=self.posy)
        self.drawing_area.bind("<Motion>", self.motion)
        self.drawing_area.bind("<ButtonPress-1>", self.b1down)
        self.drawing_area.bind("<ButtonRelease-1>", self.b1up)
        self.button=tk.Button(self.parent,text="Submit",width=10,bg='white',command=self.save)
        self.button.place(x=self.sizex/7,y=self.sizey+20)
        self.button1=tk.Button(self.parent,text="Clear",width=10,bg='white',command=self.clear)
        self.button1.place(x=(self.sizex/7)+80,y=self.sizey+50)

        self.pixelArray = numpy.ones((self.sizex, self.sizey))

    def save(self):
        #make putin some poutine
        tempArray = numpy.zeros((1, 28*28))
        for i in range(0, self.sizex):
            for j in range(0, self.sizey):
                tempArray[0, ((j // 10) + (28 * (i // 10)))] += (1 - self.pixelArray[i, j])

        tempArray /= 100.
        return tempArray

    def clear(self):
        self.drawing_area.delete("all")
        self.pixelArray = numpy.ones((self.sizex, self.sizey))

    def b1down(self,event):
        self.b1 = "down"

    def b1up(self,event):
        self.b1 = "up"
        self.xold = None
        self.yold = None

    def motion(self,event):
        if self.b1 == "down":
            if self.xold is not None and self.yold is not None:
                event.widget.create_line(self.xold,self.yold,event.x,event.y,smooth=True,width=7,fill='black')
                if (event.x >= 1 and event.x < self.sizex - 1 and event.y >= 1 and event.y < self.sizey - 1):
                    for i in range(-3, 4):
                        for j in range(-3, 4):
                            self.pixelArray[event.y + i, event.x + j] = 0

        self.xold = event.x
        self.yold = event.y

if __name__ == "__main__":
    root=tk.Tk()
    root.wm_geometry("%dx%d+%d+%d" % (400, 400, 10, 10))
    root.config(bg='white')
    ImageGenerator(root,10,10)
    root.mainloop()
