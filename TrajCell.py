
class TrajCell:
    def __init__(self,r,c):
        self.row=r
        self.column=c
        self.trajs=[]

    def setTrajs(self,input):
        self.trajs=input