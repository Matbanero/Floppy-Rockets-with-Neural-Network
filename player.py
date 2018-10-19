
import gameConst as gc
import flappyNet as fn
import obstacle
import arcade

class Player(arcade.Sprite):

    def __init__(self, best_score=1, curr_frame=1):
        super().__init__(filename="images/spaceship2.png")
        self.h = gc.PLAYER_H
        self.center_x = gc.PLAYER_START_X
        self.center_y = gc.GAME_HEIGHT / 2
        self.w = gc.PLAYER_W
        self.v = 0
        self.alive = True
        self.score = 0
        self.score_to_beat = best_score
        self.net = fn.Network([5, 40, 1])
        self.inp = fn.np.array([])
        self.closest_obst = obstacle.Obstacle
        self.curr_frame = curr_frame

    def move(self):
        if self.v > gc.MAXSPEED:
            self.v = gc.MAXSPEED
        if self.v > -gc.MAXSPEED:
            self.v += gc.ACC
        self.center_y += self.v

    def jump(self):
        if self.net.feedforward(self.inp, self.gamma()) == 1:
            self.v += gc.DRAG
    
    def prep_in(self, obst1):        
        pos = self.center_y - gc.GAME_HEIGHT / 2
        top = obst1.top
        dist = obst1.center_x - self.center_x
        bottom = top + gc.GAP_SIZE
        self.inp = fn.np.array([pos, top, bottom, dist, self.v])

    def in_bound(self):
        return self.center_y > 0 and self.center_y + self.h / 2 < gc.GAME_HEIGHT
    
    def update(self):
        self.prep_in(self.closest_obst)
        self.jump()
        self.score += round(1 + self.lmbd(), 1)
        self.move()

    def gamma(self):
        # self.frame, given from the world, which counts the frames from the very beginning, generation is not important
        result = -0.000002 * self.curr_frame + 1
        if result < 0.1:
            result = 0.1
        return result

    def lmbd(self):
        return (1 + self.score_to_beat * 0.01) ** (100 / self.score_to_beat) - 1