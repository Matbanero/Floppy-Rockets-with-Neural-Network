
import gameConst as gc
import obstacle as ob
import player as pl
import random
import arcade
import flappyNet as fn
import json

GAME_RUNNING = 1
GAME_OVER = 2

class World(arcade.Window):

    def __init__(self, width, height):
        super().__init__(width, height)
        self.background = None
        arcade.set_background_color(arcade.color.NAVY_BLUE)
        self.generation = 0
        self.best_score = 1
        self.curr_frame = 0
        
    def setup(self):  
        self.pop_list = arcade.SpriteList()
        self.obst_list = arcade.SpriteList()
        self.curr_state = GAME_RUNNING
        self.generation += 1
        self.training_data = {"input": [],
                              "output": []}
        self.spawnPopulation()
        self.addText()
        self.background = arcade.load_texture("images/bkg.png")

    def spawnPopulation(self):
        p = pl.Player(best_score=self.best_score, curr_frame=self.curr_frame)
        try:
            p.net = fn.load("currNet.txt")
            self.training_data = fn.load_data()
            self.best_score, self.frame = load_score()
            # Add hypparam picker here...
            p.net.SGD(self.training_data, 30, 15, 0.02, 0.1)
        except:
            pass
        self.pop_list.append(p)
            


    def addText(self):
        start_x = gc.GAME_WIDTH - 100
        start_y = gc.GAME_HEIGHT - 40
        arcade.draw_text(f"Generation: {self.generation:1d}\nBest Fit: {self.best_score:1f}",
                             start_x, start_y, arcade.color.WHITE_SMOKE, 10)


    def update(self, delta_time): 
        self.curr_frame += 1
        if  self.curr_state == GAME_RUNNING:
            if (not self.obst_list) or (self.obst_list[-1].center_x <= (self.width - gc.OBSTACLE_GEN_OFF)):
                obstacle1 = ob.Obstacle(gc.GAME_WIDTH, 0)
                obstacle2 = ob.Obstacle(gc.GAME_WIDTH, (obstacle1.top+gc.GAP_SIZE))
                self.obst_list.append(obstacle1)
                self.obst_list.append(obstacle2)
            self.obst_list.update()

            for p in self.pop_list:
                collision = arcade.check_for_collision_with_list(p, self.obst_list)
                if (not p.in_bound()) or any(collision):
                    
                    if p.score > (self.best_score * 0.7):
                        p.net.save_data(self.training_data)
                    
                    if p.score * 0.96 > self.best_score:
                        self.best_score = p.score
                        p.net.save("bestNet.txt")

                    self.save_score()
                    p.net.save("currNet.txt")
                    p.kill()
                    
                if self.obst_list[0].center_x - p.center_x > 0:
                    p.closest_obst = self.obst_list[0]
                else:
                    p.closest_obst = self.obst_list[2]
                p.update()

            if len(self.pop_list) == 0:
                self.curr_state = GAME_OVER
                arcade.finish_render()
        
        elif self.curr_state == GAME_OVER:
            self.setup()


    def on_draw(self):
        arcade.start_render()
        arcade.draw_texture_rectangle(gc.GAME_WIDTH // 2, gc.GAME_HEIGHT // 2,
                                      gc.GAME_WIDTH, gc.GAME_HEIGHT, self.background)
        self.obst_list.draw()
        self.pop_list.draw()
        self.addText()

    def save_score(self):
        data = {"best_score": self.best_score,
                "frame": self.curr_frame}
        f = open("bestScore.txt", "w")
        json.dump(data, f)
        f.close()

def load_score():
    f = open("bestScore.txt", "r")
    data = json.load(f)
    f.close()
    return (data["best_score"], data["frame"])

def main():
    world = World(gc.GAME_WIDTH, gc.GAME_HEIGHT)
    world.setup()
    arcade.run()


if __name__ == "__main__":
    main()

 

    
