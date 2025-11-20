

class SubTraj:
    def __init__(self,initLat,initLon,cutLat,cutLon,points,isFromInit):
        self.initLat=initLat
        self.initLon=initLon
        self.points=points
        self.isFromInit=isFromInit
        self.cutLat=cutLat
        self.cutLon=cutLon
        self.streetPoints=[]
