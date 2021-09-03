import os
import numpy as np
from collections import namedtuple
import pickle

#I0 is the distribution of infected people at time t=0, if None then randomly choose inf infections people

with open(os.path.abspath("covData/Yerevan_population.pkl"), "rb") as population:
    mumbaiPopulation = pickle.load(population)[:,:531]

# print(population.shape)
# The data procured has a shape of (84,531). The 84 represents a different time period (each time period is 2hrs, so 84 = 1 week). 531 refers to the cells

with open(os.path.abspath("covData/Yerevan_OD_matrices.pkl"), "rb") as originDestinationMatrix:
    originDestinationMatrix = pickle.load(originDestinationMatrix)[:,:531,:531]

# print(originDestinationMatrix.shape)
# The data procured has a shape of (84,531, 531) where people can move from one cell to another and we can track common trends in movements.
# After observing the data, most of the numbers are the same suggesting most people stay in one cell. This makes sense since the majority of the population will have their work/house/park/supermarket all in the same area.

# Maths Behind Model
par = namedtuple("Parameter", "r0Number DE infectiousPeriod initialRandomInfections HospitalisationRate timeInHospital")

def seirModel(par, distr, flow, lockdownStrictness, iterations, initialInfections):
    # Based on OriginDestinationMatrix Model (so time step = 84, cells = 531, total population is the sum of all people in figure)
    timeStep = flow.shape[0]
    cells = flow.shape[1]
    totalPopulation = distr[0].sum()  # total population, we assume that N = sum(flow)

    # The initial susceptible is the total population, the rest of the parameters are 0
    initialSusceptible = distr[0].copy()
    initialExposed = np.zeros(cells)
    initialInfectious = np.zeros(cells)
    initialRecovered = np.zeros(cells)

    #  If there are no initial infections, we need to make some random infections
    # We create a new numpy array, with all cells as 0. Then we randomly generate some cells and add 1 to these cells to act as an infected individual.
    if par.initialRandomInfections is None:
        initial = np.zeros(cells)
        # randomly choose inf infections
        for i in range(initialInfections):
            location = np.random.randint(cells)
            if initialSusceptible[location] > initial[location]:
                initial[location] += 1.0

    else:
        initial = par.infectiousPeriod

    # Now, we can take away these people from the susceptible group
    initialSusceptible -= initial
    # And add them to the infectious group
    initialInfectious += initial

    # The result of this will be an array with the number of iterations (how long the epidemic will last) x 5 parameters (S,E,I,R,Hospitalised People)
    result = np.zeros((iterations, 5))
    # There are no hospitalised people (hence will be 0)
    result[0, :] = [initialSusceptible.sum(), initialExposed.sum(), initialInfectious.sum(), initialRecovered.sum(), 0]

    # THIS IS BAD PRACTICE::::  - instead we can use broadcasting
    # for j in range(timeStep):
    #     for i in range(cells):
    #         realFlow[j][i] /= realFlow[j][i].sum()

    realFlow = flow.copy()  # copy!
    realFlow = realFlow / realFlow.sum(axis=2)[:, :, np.newaxis]
    realFlow = lockdownStrictness * realFlow

    # Modelling the Equations
    # history of each of the different parameters:

    # At time 0, all the parameters will be 0

    history = np.zeros((iterations, 5, cells))
    history[0, 0, :] = initialSusceptible
    history[0, 1, :] = initialExposed
    history[0, 2, :] = initialInfectious
    history[0, 3, :] = np.zeros(cells)

    eachIter = np.zeros(iterations + 1)

    # run simulation
    for iteration in range(0, iterations - 1):
        realOriginDestinationMatrix = realFlow[iteration % timeStep]

        # The population density at each time step
        popDensityTime = distr[iteration % timeStep] + 1

        if (popDensityTime > totalPopulation + 1).any():  # Assertion!
            print("Uh Oh! Something's Wrong!")  # Has to be smaller than total population
            return result, history
        # Total Population = S + E + I + R

        # From the SEIR Model Equations
        newExposed = initialSusceptible * initialInfectious / popDensityTime * par.r0Number / par.infectiousPeriod
        newInfectious = initialExposed / par.DE
        newRecovered = initialInfectious / par.infectiousPeriod

        initialSusceptible -= newExposed
        initialSusceptible = (initialSusceptible
                + np.matmul(initialSusceptible.reshape(1, cells), realOriginDestinationMatrix)
                - initialSusceptible * realOriginDestinationMatrix.sum(axis=1)
                )
        initialExposed = initialExposed + newExposed - newInfectious
        initialExposed = (initialExposed
                + np.matmul(initialExposed.reshape(1, cells), realOriginDestinationMatrix)
                - initialExposed * realOriginDestinationMatrix.sum(axis=1)
                )

        initialInfectious = initialInfectious + newInfectious - newRecovered
        initialInfectious = (initialInfectious
                + np.matmul(initialInfectious.reshape(1, cells), realOriginDestinationMatrix)
                - initialInfectious * realOriginDestinationMatrix.sum(axis=1)
                )

        initialRecovered += newRecovered
        initialRecovered = (initialRecovered
                + np.matmul(initialRecovered.reshape(1, cells), realOriginDestinationMatrix)
                - initialRecovered * realOriginDestinationMatrix.sum(axis=1)
                )

        result[iteration + 1, :] = [initialSusceptible.sum(), initialExposed.sum(), initialInfectious.sum(), initialRecovered.sum(), 0]
        eachIter[iteration + 1] = newInfectious.sum()
        result[iteration + 1, 4] = eachIter[max(0, iteration - par.timeInHospital): iteration].sum() * par.HospitalisationRate

        history[iteration + 1, 0, :] = initialSusceptible
        history[iteration + 1, 1, :] = initialExposed
        history[iteration + 1, 2, :] = initialInfectious
        history[iteration + 1, 3, :] = initialRecovered

    return result, history