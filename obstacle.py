import gameConst as gc
import random
import arcade

class Obstacle(arcade.Sprite):

    def __init__(self, x, y):
        super().__init__("images/obstext.png")
        self.center_x = x + gc.OBSTACLE_W / 2
        self.bottom = y
        self.width = gc.OBSTACLE_W
        if y == 0:
            self.top = random.randrange(start=gc.MIN_HEIGHT, stop=(gc.GAME_HEIGHT - gc.GAP_SIZE - gc.MIN_HEIGHT))
        else:
            self.angle = 180
    
    def update(self):
        self.center_x -= gc.WORLD_SPEED
        if self.right <= 0:
            super().kill()

    def inBound(self):
        return self.center_x + self.width / 2 >= 0
        