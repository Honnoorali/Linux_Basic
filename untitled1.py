class Rect:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
    
    def bottom(self):
        return self.y + self.h
    
    def right(self):
        return self.x + self.w
    
    def area(self):
        return self.w * self.h

    def union(self, b):
        posX = min(self.x, b.x)
        posY = min(self.y, b.y)
        
        return Rect(posX, posY, max(self.right(), b.right()) - posX, max(self.bottom(), b.bottom()) - posY)
    
    def intersection(self, b):
        posX = max(self.x, b.x)
        posY = max(self.y, b.y)
        
        candidate = Rect(posX, posY, min(self.right(), b.right()) - posX, min(self.bottom(), b.bottom()) - posY)
        if candidate.w > 0 and candidate.h > 0:
            return candidate
        return Rect(0, 0, 0, 0)
    
    def ratio(self, b):
        return self.intersection(b).area() / self.union(b).area()


r1 = Rect(1306,1110,1324,1038)
r2 = Rect(1386,1110,1490,1058)

r3 = r1.intersection(r2)
print(r3.x, r3.y, r3.w, r3.h)    
#assert Rect(1, 1, 1, 1).ratio(Rect(1, 1, 1, 1)) == 1
#assert round(Rect(1, 1, 2, 2).ratio(Rect(2, 2, 2, 2)), 4) == 0.1111


