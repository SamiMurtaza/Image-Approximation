import numpy as np


from PIL import Image as plImage
from PIL import ImageDraw as plDraw

from numpy.random import randint

from random import random
from time import perf_counter
from heapq import nsmallest
from random import choice

fname = "lisa.png"
#fname = "bat.jpg"
#fname = "half.jpg"
#fname = "tick.jpg"
#fname = "balakot(1).jpg"
#fname = "mashallah.jpg"
#fname = "wallpaper (2).jpg"
#fname = "class.jpg"
#fname = "copy.jpg"
#fname = "good.jpg"
n_shapes = 100
min_points, max_points = 3,6
internal_res = 160

size = 100
population = 30
offspring = 10
iterations = 10
generations =1000001
mutate = 0.7
constant = 10000000000

original = plImage.open(fname).convert('RGBA')
Ori = np.asarray(original, dtype=int)

class chromosome():

    def __init__(self, shapes=None, fill=None):
        if shapes is None:
            self.im = plImage.new("RGBA", original.size)
            self.ima = plDraw.Draw(self.im)
            self.shapes = []
            self.fill=[]
            for i in range(n_shapes):
                edges = randint(min_points, max_points)
                p1 = (randint(0,original.size[0]), randint(0,original.size[1]))
                s = [(p1[0]+randint(-size,size),p1[1]+randint(-size,size))  for i in range(edges)]            
                self.shapes.append(s)
                #self.fill.append((randint(0,255) for i in range(3))+(randint(0,128)))
                self.fill.append((randint(0,255),randint(0,255),randint(0,255),randint(0,255)))
                self.ima.polygon(s, self.fill[i] )
        else:
            self.im = plImage.new("RGBA", original.size)
            self.ima = plDraw.Draw(self.im)
            self.shapes = shapes
            self.fill = fill
            for i in range(len(shapes)):
                self.ima.polygon(self.shapes[i], fill= self.fill[i] )

        self.his = np.asarray(self.im, dtype=int)
        self.score = self.error_abs()
        self.fitness = self.update_fitness()
        
        
    def show(self):
        self.im.show()
        
##        self.score = self.error_abs()
##        self.fitness = constant/ self.score

    def error_abs(self): #previously known as distance_tour
        return np.sqrt(((Ori - self.his)**2).sum(axis=-1)).sum()

    def update_score(self): #previouslt known as update_lenght
        
        self.im = plImage.new("RGBA", original.size)
        self.ima = plDraw.Draw(self.im)

        for i in range(n_shapes):
            self.ima.polygon(self.shapes[i], self.fill[i] )
            
        self.his = np.asarray(self.im, dtype=int)
        self.score = self.error_abs()
        self.fitness = self.update_fitness()
        
        

    def update_fitness(self):
        self.fitness = constant/self.score

    def error_percent(self):
        '''Calculates human-readable % of error from absolute error'''
        return self.score / (chromosome.internal.shape[0] * chromosome.internal.shape[1] * 255 * 3) * 100

class pool():
    def __init__(self):
        self.population = [chromosome() for i in range(population)]

    avg = lambda self: sum([i.score for i in self.population])/population
    best = lambda self: min([i.score for i in self.population])

    def random_kill(self):
        while len(self.population) > population:
            self.population.pop(randint(0,len(self.population)-1))

    def random_parent(self):
        return self.population[randint(0,len(self.population)-1)], self.population[randint(0,len(self.population)-1)]

    def fitness_proportional_killing(self):
        while len(self.population) > population:
            tot = sum([self.population[i].score for i in range(population) ] )
            prop = [(self.population[i].score/tot) for i in range(population)]
            ran = random()
            current = 0
            for k in range(len(prop)):
                current += prop[k]
                if current >= ran:
                    self.population.pop(k)
                    break

    def fitness_proportional_parent(self):
        
        tot = sum([self.population[i].fitness for i in range(population) ] )
        prop = [(self.population[i].fitness/tot) for i in range(population)]
        
        ran = random()
        current = 0
        
        for i in range(len(prop)):
            current += prop[i]
            if current >= ran or current > tot:
                parent1 = self.population[i]
                break
            
        ran = random()
        current = 0
        
        for i in range(len(prop)):
            current += prop[i]
            if current >= ran or current > tot:
                parent2 = self.population[i]
                break
        return (parent1,parent2)


    def rank_based_parent(self):
        self.population = sorted(self.population,key=lambda pop:pop.score)
        tot = sum([j for j in range(population) ] )
        prop = [(j/tot) for j in range(population)]
        ran = random()
        current = 0
        
        for i in range(len(prop)):
            current += prop[i]
            if current >= ran:
                parent1 = self.population[i]
                break 
        ran = random()
        current = 0
        for i in range(len(prop)):
            current += prop[i]
            if current >= ran:
                parent2 = self.population[i]
                break
        return (parent1,parent2)

    def rank_based_kill(self):
        self.population = sorted(self.population,key=lambda pop:pop.score)
        
        while len(self.population) > population:
            tot = sum([j+1 for j in range(population) ] )
            prop = [j+1/tot for j in range(population)]
            ran = random()
            current = 0
            for k in range(len(prop)):
                current += prop[k]
                if current >= ran:
                    self.population.pop(k)
                    break
        

    def binary_tournament_parent(self):
        player1,player2= choice(self.population),choice(self.population)
        if player1.score < player2.score:
            parent1 = player1
        else:
            parent1 = player2
            
        player1,player2= choice(self.population),choice(self.population)
        if player1.score < player2.score:
            parent2 = player1
        else:
            parent2 = player2

        return parent1,parent2
        

    def binary_tournament_kill(self):
        while len(self.population) > population:
            player1,player2= choice(self.population),choice(self.population)
            if player1.score >player2.score:
                self.population.pop(self.population.index(player1))
            else:
                self.population.pop(self.population.index(player2))


    def truncation_parent(self):
        #nsmallest(2, numbers)
        ret = [i.score for i in self.population]
        ind = nsmallest(2,ret)
        return self.population[ret.index(ind[0])],self.population[ret.index(ind[1])]
        
    def truncation_kill(self):
        while len(self.population) > population:
            ret = [i.score for i in self.population]
            self.population.pop(ret.index(max(ret)))

    def add(self, child1, child2):
        self.population.append(child1)
        self.population.append(child2)
        
    def crossover(self,ma, ba):
        
        cut1, cut2 = randint(0,n_shapes-1),randint(0,n_shapes-1)
        factor = int (max(original.size) / internal_res)
        mom = chromosome(ma.shapes, ma.fill)
        dad = chromosome(ba.shapes, ba.fill)

        child1_shapes, child2_shapes = [-1 for i in range(n_shapes)], [-1 for i in range(n_shapes)]
        child1_fill, child2_fill = [-1 for i in range(n_shapes)], [-1 for i in range(n_shapes)]
        
        child1_shapes[min(cut1,cut2):max(cut1,cut2)] = mom.shapes[min(cut1,cut2):max(cut1,cut2)]
        child1_fill[min(cut1,cut2):max(cut1,cut2)] = mom.fill[min(cut1,cut2):max(cut1,cut2)]
        
        child2_shapes[min(cut1,cut2):max(cut1,cut2)] = dad.shapes[min(cut1,cut2):max(cut1,cut2)]
        child2_fill[min(cut1,cut2):max(cut1,cut2)] = dad.fill[min(cut1,cut2):max(cut1,cut2)]

        
        
        for i in range(n_shapes):
            if type(child1_shapes[i]) == int:
                
                child1_shapes[i] = dad.shapes[i]
                child1_fill[i] = dad.fill[i]
               

            if type(child2_shapes[i]) == int:
                child2_shapes[i] = mom.shapes[i]
                child2_fill[i] = mom.fill[i]
            
        child1=chromosome(child1_shapes, child1_fill)
        child2=chromosome(child2_shapes, child2_fill)

        
        
        return child1, child2
        

    def mutation(self):

        
        for i in self.population:
            ran = random()
            
            def col(index):
                i.fill[index] = (randint(0,255),randint(0,255),randint(0,255),randint(0,255))

            def locx(index):
                fac = randint(-original.size[0]/4,original.size[0]/4)
                i.shapes[index] = [(i[0]+fac, i[1]) for i in i.shapes[index]]

            def locy(index):
                fac = randint(-original.size[1]/4,original.size[1]/4)
                i.shapes[index] = [(i[0], i[1]+fac) for i in i.shapes[index]]
                
            def order(index):
                to = randint(0,n_shapes)
                i.shapes[index],i.shapes[to] = i.shapes[to], i.shapes[index]
                
            def change(index):
                edges = randint(min_points, max_points)
                p1 = (randint(0,original.size[0]), randint(0,original.size[1]))
                i.shapes[index]  = [(p1[0]+randint(-size,size),p1[1]+randint(-size,size))  for i in range(edges)]             
                    
            if ran <= mutate:
                index, func = randint(n_shapes), np.random.choice([change, col,  order])
                func(index)
                i.update_score()
                i.update_fitness()
                
    
class EA():
    def __init__(self):
        self.people = pool()
        
    def evolve(self, parent, kill):
        x=perf_counter()
        
        for i in range(generations):
            

                
            if i%250==0:
                print (i,"\t avg=",int(self.people.avg()), "\t best=",int(self.people.best()), "\t",round(perf_counter()-x,2),"sec")
                s=self.people.best()
                for j in self.people.population:
                  if j.score == s:
                    j.im.save('CLisa'+str(i)+'.png', 'PNG')
                    break
                
                    
            for j in range(population,population+offspring,2):

                if parent==0: mom, dad = self.people.fitness_proportional_parent()
                if parent==1: mom, dad = self.people.rank_based_parent()
                if parent==2: mom, dad = self.people.binary_tournament_parent()
                if parent==3: mom, dad = self.people.truncation_parent()
                if parent==4: mom, dad = self.people.random_parent()

                boy, girl = self.people.crossover(mom, dad)
                self.people.add(boy, girl)

            if kill==0:self.people.fitness_proportional_killing()
            if kill==1:self.people.rank_based_kill()   
            if kill==2:self.people.binary_tournament_kill()
            if kill==3:self.people.truncation_kill()
            if kill==4:self.people.random_kill()
            
            self.people.mutation()
            
        print (i, int(self.people.avg()), int(self.people.best()))


play = EA()
play.evolve(3,3)
