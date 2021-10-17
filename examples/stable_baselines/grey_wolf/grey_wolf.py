"""
Grey Wolf Optimization.

https://www.geeksforgeeks.org/implementation-of-grey-wolf-optimization-gwo-algorithm/
"""

import random
import copy
from datetime import datetime
import time


from typing import List, Dict
from matplotlib import pyplot as plt


class Wolf:
    """
        Wolf class.
    """

    def __init__(self, fitness, intervals: List[List[float]], seed):
        self.rnd = random.Random(seed)
        self.position = [0.0 for i in range(len(intervals))]

        for index, key in enumerate(intervals):
            maximum = intervals[key][1]
            minimum = intervals[key][0]

            self.position[index] = ((maximum - minimum) * self.rnd.random() + minimum)


        print("lolol")
        print("jlskdjflskjdf")
        print("jlskdjflskjdf")
        print("jlskdjflskjdf")
        print("jlskdjflskjdf")
        print("jlskdjflskjdf")
        print(self.position)

        self.fitness = fitness(self.position, seed, 998) # curr fitness


class GreyWolfOptimizer:
    """
        Grey Wolf Optimizer class.
    """

    def __init__(self,
        objective_function,
        max_iteration_number,
        num_wolves,
        intervals: Dict[str, List[float]],
        logdir,
        subprocesses
    ):
        self.objective_function = objective_function
        self.max_iteration_number = max_iteration_number
        self.num_wolves = num_wolves
        self.intervals = intervals
        self.logdir = logdir
        self.subprocesses = subprocesses

    # def _f(self, hyperparameters):
    #     return self.objective_function(hyperparameters)

    def evaluate_fitness(self, position, wolf_number, iteration_number):
        """
            Evaluate the fitness of a wolf given its position.
        """
        # fitness_value = 0.0

        # for i in enumerate(position):
        #     fitness_value += self._f(position[i])

        # return fitness_value
        return self.objective_function(position, wolf_number, iteration_number, self.logdir, self.subprocesses)

    def plot_population(self, population):
        """
            Plot the positions of wolfs on a chart.j

            :param population: (list) List of Wolf objects.
        """
        plt.clf()
        # curve_x = np.arange(minx, maxx, 0.01)
        # curve_y = [_f(x) for x in curve_x]

        positions = [w.position for w in population]
        _x = [p[0] for p in positions]
        _y = [p[1] for p in positions]

        # plt.plot(curve_x, curve_y, color="green")
        plt.scatter(_x, _y)
        plt.pause(0.01)

    def run_gwo(self):
        """
            Run Grey Wolf optimization.
        """
        rnd = random.Random(0)

        # create n random wolves
        # population = [ wolf(fitness, dim, minx, maxx, i) for i in range(n)]

        population = []

        for i in range(self.num_wolves):
            print(f"Creating wolf {i}")

            seed = time.mktime((datetime.now()).timetuple()) + i * 1024

            population.append(
                Wolf(self.evaluate_fitness, self.intervals, seed)
            )

            print(f"Created wolf {i}")

        # self.plot_population(population)

        # On the basis of fitness values of wolves
        # sort the population in asc order
        population = sorted(population, key = lambda temp: temp.fitness)

        # best 3 solutions will be called as
        # alpha, beta and gama
        alpha_wolf, beta_wolf, gamma_wolf = copy.copy(population[: 3])


        # main loop of gwo
        iteration_number = 0
        while iteration_number < self.max_iteration_number:

            # after every 10 iteration_numberations
            # print iteration_numberation number and best fitness value so far
            if iteration_number % 10 == 0 and iteration_number > 1:
                print(
                    f"iteration_number = {str(iteration_number)} best fitness = %.3f"
                    % alpha_wolf.fitness
                )

            # linearly decreased from 2 to 0
            _a = 2*(1 - iteration_number/self.max_iteration_number)

            # updating each population member with the help of best three members
            for i in range(self.num_wolves):
                a_1, a_2, a_3 = _a * (2 * rnd.random() - 1), _a * (2 * rnd.random() - 1), _a * (2 * rnd.random() - 1)
                c_1, c_2, c_3 = 2 * rnd.random(), 2 * rnd.random(), 2 * rnd.random()

                number_of_dimensions = len(self.intervals)

                x_1 = [0.0 for i in range(number_of_dimensions)]  # for each dimension
                x_2 = [0.0 for i in range(number_of_dimensions)]
                x_3 = [0.0 for i in range(number_of_dimensions)]
                x_new = [0.0 for i in range(number_of_dimensions)]

                for j in range(number_of_dimensions):
                    x_1[j] = alpha_wolf.position[j] - a_1 * abs(c_1 * alpha_wolf.position[j] - population[i].position[j])
                    x_2[j] = beta_wolf.position[j] - a_2 * abs(c_2 * beta_wolf.position[j] - population[i].position[j])
                    x_3[j] = gamma_wolf.position[j] - a_3 * abs(c_3 * gamma_wolf.position[j] - population[i].position[j])
                    x_new[j] += x_1[j] + x_2[j] + x_3[j]

                    if x_new[j] < 0:
                        x_new[j] *= -1

                for j in range(number_of_dimensions):
                    x_new[j] /= 3.0

                # fitness calculation of new solution
                fnew = self.evaluate_fitness(x_new, i, iteration_number)

                # greedy selection
                if fnew < population[i].fitness:
                    population[i].position = x_new
                    population[i].fitness = fnew

            # On the basis of fitness values of wolves
            # sort the population in asc order
            population = sorted(population, key = lambda temp: temp.fitness)

            # self.plot_population(population)

            # best 3 solutions will be called as
            # alpha, beta and gama
            alpha_wolf, beta_wolf, gamma_wolf = copy.copy(population[:3])
            print(f"ALPHA: {alpha_wolf.position}")

            iteration_number += 1
        # end-while

        plt.show()

        # returning the best solution
        return alpha_wolf.position
	