#include "TSearch.h"
#include "LeggedAgent.h"
#include "CTRNN.h"
#include "random.h"

#define PRINTOFILE

// Task params
const double StepSizeTF = 0.05;
const double RunDurationTF = 220.0;

const double StepSizeAF = 0.01;
const double MaxWaitTime = 500.0;

// EA params
const int POPSIZE = 100;
const int GENS = 1000;
const double MUTVAR = 0.1;
const double CROSSPROB = 0.5;
const double EXPECTED = 1.1;
const double ELITISM = 0.1;

// Nervous system params
const int N = 2;
const double WR = 16.0;
const double BR = 16.0;
const double TMIN = 0.5;
const double TMAX = 10.0;

int	VectSize = N*N + 2*N;

// ------------------------------------
// Genotype-Phenotype Mapping Functions
// ------------------------------------
void GenPhenMapping(TVector<double> &gen, TVector<double> &phen)
{
	int k = 1;
	// Time-constants
	for (int i = 1; i <= N; i++) {
		phen(k) = MapSearchParameter(gen(k), TMIN, TMAX);
		k++;
	}
	// Bias
	for (int i = 1; i <= N; i++) {
		phen(k) = MapSearchParameter(gen(k), -BR, BR);
		k++;
	}
	// Weights
	for (int i = 1; i <= N; i++) {
			for (int j = 1; j <= N; j++) {
				phen(k) = MapSearchParameter(gen(k), -WR, WR);
				k++;
			}
	}
}

// ------------------------------------
// Fitness function
// ------------------------------------
double TruncatedFitnessFunction(TVector<double> &genotype, RandomState &rs)
{
		// Map genootype to phenotype
		TVector<double> phenotype;
		phenotype.SetBounds(1, VectSize);
		GenPhenMapping(genotype, phenotype);

		// Create the agent
		LeggedAgent Insect;

		// Instantiate the nervous system
		Insect.NervousSystem.SetCircuitSize(N);
		int k = 1;
		// Time-constants
		for (int i = 1; i <= N; i++) {
			Insect.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
			k++;
		}
		// Bias
		for (int i = 1; i <= N; i++) {
			Insect.NervousSystem.SetNeuronBias(i,phenotype(k));
			k++;
		}
		// Weights
		for (int i = 1; i <= N; i++) {
				for (int j = 1; j <= N; j++) {
					Insect.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
					k++;
				}
		}

		Insect.Reset(0, 0, 0);

    // Run the agent
    for (double time = 0; time < RunDurationTF; time += StepSizeTF) {
        Insect.Step1CPG(StepSizeTF);
    }

    // Finished
    return Insect.cx/RunDurationTF;
}

double AsymptoticFitnessFunction(TVector<double> &genotype, RandomState &rs)
{
	// Map genootype to phenotype
	TVector<double> phenotype;
	phenotype.SetBounds(1, VectSize);
	GenPhenMapping(genotype, phenotype);

	// Create the agent
	LeggedAgent Insect;

	// Instantiate the nervous system
	Insect.NervousSystem.SetCircuitSize(N);
	int k = 1;
	// Time-constants
	for (int i = 1; i <= N; i++) {
		Insect.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
		k++;
	}
	// Bias
	for (int i = 1; i <= N; i++) {
		Insect.NervousSystem.SetNeuronBias(i,phenotype(k));
		k++;
	}
	// Weights
	for (int i = 1; i <= N; i++) {
		for (int j = 1; j <= N; j++) {
			Insect.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
			k++;
		}
	}

	Insect.Reset(0, 0, 0);

	// Run the agent for a transient period
	double pastFootState,newFootState;
	for (double time = 0; time < RunDurationTF; time += StepSizeAF) {
		Insect.Step1CPG(StepSizeAF);
		pastFootState = newFootState;
		newFootState = Insect.Leg.FootState;
	}

	// Run the agent until you find a crossing point
	double timer = 0.0;
	while (! ((newFootState < 0.5) && (pastFootState > 0.5)))
	{
		Insect.Step1CPG(StepSizeAF);
		pastFootState = newFootState;
		newFootState = Insect.Leg.FootState;
		timer += StepSizeAF;
		if (timer > MaxWaitTime)
		{
			return 0.0;
		}
	}

	// Run the agent until you find the crossing point again
	timer = 0.0;
	double startingx,endingx;
	startingx = Insect.cx;
	Insect.Step1CPG(StepSizeAF);
	pastFootState = newFootState;
	newFootState = Insect.Leg.FootState;
	while (! ((newFootState < 0.5) && (pastFootState > 0.5)))
	{
		Insect.Step1CPG(StepSizeAF);
		pastFootState = newFootState;
		newFootState = Insect.Leg.FootState;
		timer += StepSizeAF;
		if (timer > MaxWaitTime)
		{
			return 0.0;
		}
	}
	endingx = Insect.cx;

	// If found, return velocity
	return (endingx-startingx)/timer;
}

// ------------------------------------
// Display functions
// ------------------------------------
void EvolutionaryRunDisplay(int Generation, double BestPerf, double AvgPerf, double PerfVar)
{
	cout << Generation << " " << BestPerf << " " << AvgPerf << " " << PerfVar << endl;
}

void ResultsDisplay(TSearch &s)
{
	TVector<double> bestVector;
	ofstream BestIndividualFile;
	TVector<double> phenotype;
	phenotype.SetBounds(1, VectSize);

	// Save the genotype of the best individual
	bestVector = s.BestIndividual();
	BestIndividualFile.open("best.gen.dat");
	BestIndividualFile << bestVector << endl;
	BestIndividualFile.close();

	// Also show the best individual in the Circuit Model form
	BestIndividualFile.open("best.ns.dat");
	GenPhenMapping(bestVector, phenotype);
	LeggedAgent Insect;
	// Instantiate the nervous system
	Insect.NervousSystem.SetCircuitSize(N);
	int k = 1;
	// Time-constants
	for (int i = 1; i <= N; i++) {
		Insect.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
		k++;
	}
	// Bias
	for (int i = 1; i <= N; i++) {
		Insect.NervousSystem.SetNeuronBias(i,phenotype(k));
		k++;
	}
	// Weights
	for (int i = 1; i <= N; i++) {
			for (int j = 1; j <= N; j++) {
				Insect.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
				k++;
			}
	}
	BestIndividualFile << Insect.NervousSystem;
	BestIndividualFile.close();
}

// ------------------------------------
// Stage finish functions
// ------------------------------------
int FinishedTruncatedFitness(int Generation,double BestPerf,double AvgPerf,double PerfVar){
    //if (BestPerf > 0.5) return 1;
		if (BestPerf > 0.615) return 1;
    else return 0;
}
int FinishedAsymptoticFitness(int Generation,double BestPerf,double AvgPerf,double PerfVar){
    if (Generation > GENS) return 1;
    else return 0;
}

// ------------------------------------
// The main program
// ------------------------------------
int main (int argc, const char* argv[]) {

	long IDUM=-time(0);
	TSearch s(VectSize);

#ifdef PRINTOFILE
	ofstream file;
	file.open("evol.dat");
	cout.rdbuf(file.rdbuf());
#endif

  // Configure the search
  s.SetRandomSeed(IDUM);
	s.SetSearchResultsDisplayFunction(ResultsDisplay);
	s.SetPopulationStatisticsDisplayFunction(EvolutionaryRunDisplay);
  s.SetSelectionMode(RANK_BASED);
  s.SetReproductionMode(GENETIC_ALGORITHM);
  s.SetPopulationSize(POPSIZE);
  s.SetMaxGenerations(GENS);
  s.SetCrossoverProbability(CROSSPROB);
  s.SetCrossoverMode(UNIFORM);
	s.SetMutationVariance(MUTVAR);
  s.SetMaxExpectedOffspring(EXPECTED);
  s.SetElitistFraction(ELITISM);
  s.SetSearchConstraint(1);

  // Run Stage 1
	s.SetEvaluationFunction(TruncatedFitnessFunction);
	s.SetSearchTerminationFunction(FinishedTruncatedFitness);
  s.ExecuteSearch();

	// Run Stage 2
	s.SetEvaluationFunction(AsymptoticFitnessFunction);
	s.SetSearchTerminationFunction(FinishedAsymptoticFitness);
  s.ExecuteSearch();

  return 0;
}
