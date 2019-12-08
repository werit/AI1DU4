using System;
using System.Collections.Generic;
using System.Linq;
using LocalSearch;

namespace TSP
{
    class GASolverManager : TSPSolver
    {
        GaSolver<PermutationStandard, TSPInput> solver;
        private TSPVisualizer visualizer;

        public GASolverManager(TSPVisualizer vis, Random random)
        {
            this.visualizer = vis;
            var initializationOperator = new PermutationRandomizer(random);
            var xoverOperator = new XoverOperator<PermutationStandard, TSPInput,PermutationStandardFactory>(PermutationStandardFactory.GetInstance(), random);
            //var initializationOperator = new PermutationRandomizer2(r);

            var mutationOperator =
                new SimpleSwapOperator<PermutationStandard, TSPInput, PermutationStandardFactory>(PermutationStandardFactory.GetInstance(), random);
            //var modificationOperator = new Rotate3Values<PermutationStandard, TSPInput, PermutationStandardFactory>(PermutationStandardFactory.GetInstance(), r);

            this.solver = new GaSolver<PermutationStandard, TSPInput>(random,initializationOperator, xoverOperator, mutationOperator);
        }

        public TSPSolution solve(TSPInput input)
        {
            Console.WriteLine("GA started");
            var res = solver.Solve(input, printProgress);
            Console.WriteLine("GA search ended");
            return res.convertToTSPSol();
        }

        private void printProgress(PermutationStandard currentBest, int steps)
        {
            var sol = currentBest.convertToTSPSol();
            visualizer.draw(sol);
            Console.WriteLine("Steps: " + steps + " Best distance: " + sol.totalDistance);
        }

    }

    internal class GaSolver<T, P>where T : CandidateSolution<P> where P : LocalSearchProblem
    {
        private Random random;
        private T current;
        private double currentBestVal;
        private const double MutationThreshold = 0.9;

        private LSBinaryOperator<T, P> xoverOperator;
        private LSUnaryOperator<T, P> mutationOperator;
        private LSNullaryOperator<T, P> initializationOperator;
        private const int PopulationSize= 100;
        private const int EvolutionSteps = 1000;

        public GaSolver(Random random,LSNullaryOperator<T, P> initializationOperator, XoverOperator<PermutationStandard, TSPInput,PermutationStandardFactory> xoverOperator, SimpleSwapOperator<PermutationStandard, TSPInput, PermutationStandardFactory> mutationOperator)
        {
            this.random = random;
            this.initializationOperator = initializationOperator;
            this.mutationOperator = (LSUnaryOperator<T, P>) mutationOperator;
            this.xoverOperator = (LSBinaryOperator<T, P>) xoverOperator;

        }
        private IEnumerable<T> initialize(P input)
        {
            var population = Enumerable.Repeat(0, PopulationSize).Select(x=>initializationOperator.Apply(input));
            return population;
        }

        public T Solve(P input, Action<T, int> onImprovementCallback = null)
        {
            currentBestVal = 0;
            var population = initialize(input).ToList();

            for (var j = 0; j < EvolutionSteps; j++)
            {
                var popualtionFitness = population.Select(ind => 1 / ind.Evaluate()).ToList();
                var bestIndivValue = popualtionFitness.Max();
                var bestIndiv = population[popualtionFitness.IndexOf(bestIndivValue)];
                if (bestIndivValue > currentBestVal)
                {
                    onImprovementCallback?.Invoke(bestIndiv, j);
                    currentBestVal = bestIndivValue;
                    current = bestIndiv;
                }

                var tempPopSum = popualtionFitness.Sum();
                var populationProbabilityDistribution = popualtionFitness.Select(x => x / tempPopSum).ToList();
                var individualSplittingPoints = RouletteSelectorSpliitingPOints(populationProbabilityDistribution);

                var getXoverPairs = Enumerable.Range(0, population.Count).Select(x => (
                    first: GetIndexOfNextGreater(individualSplittingPoints, random.NextDouble()),
                    second: GetIndexOfNextGreater(individualSplittingPoints, random.NextDouble())));

                var newGeneration = getXoverPairs.Select(pair =>
                    xoverOperator.Apply(population[pair.first], population[pair.second])).ToList();
                var mutatedNewGeneration = newGeneration.Select(ind =>
                    random.NextDouble() > MutationThreshold ? mutationOperator.Apply(ind) : ind).ToList();
                population = mutatedNewGeneration;
            }

            return current;
        }

        private static List<double> RouletteSelectorSpliitingPOints(List<double> populationProbabilityDistribution)
        {
            var individualSplittingPoints = new List<double>(populationProbabilityDistribution.Count)
            {
                populationProbabilityDistribution[0]
            };
            for (var i = 1; i < populationProbabilityDistribution.Count; i++)
            {
                individualSplittingPoints.Add(individualSplittingPoints[i - 1] +
                                              populationProbabilityDistribution[i]);
            }

            return individualSplittingPoints;
        }

        private int GetIndexOfNextGreater(List<double> individualSplittingPoints, double next)
        {
            //TODO Improve to logarithmic complexity
            var index = 0;
            while (next > individualSplittingPoints[index] && index < individualSplittingPoints.Count - 1)
            {
                index++;
            }

            return index;
        }

    }

    public class XoverOperator<T, P, F> : LSBinaryOperator<T, P>
        where T : IntegerSequenceCandidateSolution<P>
        where P : LocalSearchProblem
        where F : IntegerSequenceCandidateSolutionFactory<T, P>
    {
        Random r;
        F factory;
        public XoverOperator(F factory, Random r)
        {
            this.r = r;
            this.factory = factory;
        }
        public T Apply(T candidateSolution1, T candidateSolution2)
        {
            var coordinates = selectXoverCoordinates(candidateSolution1);
            var newData = xover(candidateSolution1,candidateSolution2, coordinates.Item1, coordinates.Item2);
            return factory.Create(newData, candidateSolution1.problem);
        }
        protected (int, int) selectXoverCoordinates(T candidate)
        {
            var firstCoord = r.Next(candidate.data.Length);
            var secondCoord = r.Next(candidate.data.Length);
            while (secondCoord == firstCoord)
                secondCoord = r.Next(candidate.data.Length);
            return (firstCoord, secondCoord);
        }

        protected int[] xover(T candidateSolution1,T candidateSolution2, int coord1, int coord2)
        {
            int startInd;
            int endInd;
            if (coord1 < coord2)
            {
                startInd = coord1;
                endInd = coord2;
            }
            else
            {
                startInd = coord2;
                endInd = coord1;
            }

            var fixedGene = candidateSolution1.data.Skip(startInd).Take(endInd - startInd+1);
            var restOfSecondParent = candidateSolution2.data.Where(num => !fixedGene.Contains(num)).ToList();
            var finalArray = restOfSecondParent.Take(startInd).Concat(fixedGene)
                .Concat(restOfSecondParent.Skip(startInd));
            return finalArray.ToArray();
        }
    }
}