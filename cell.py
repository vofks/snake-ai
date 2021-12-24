class Cell:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        if(other is None):
            return False

        return self.x == other.x and self.y == other.y

    def __str__(self):
        return f'Cell x: {self.x} y: {self.y}'
