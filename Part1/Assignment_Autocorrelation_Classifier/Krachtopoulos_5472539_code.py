# import statements
import numpy as np
import matplotlib.pyplot as plt
#import scipy.io
from scipy.stats import norm

## Exercise 1
print("===========================================================================================")
print("Exercise 1")
print("===========================================================================================")

# Set the precision
PRECISION = 3

# Exercise 1.1
# No code

# Exercise 1.2
numTs = 10
numPoints = 100
mean = 0
variance = 1

# The following file contains the realizations that were randomly generated after the first generation.
realTable = np.load("realTable.npy")

plt.figure('Timeseries realizations')
#realTable = np.zeros((numTs, numPoints))
for ts in range(numTs):
	#    for point in range(numPoints):
	#       realTable[ts, point] = np.random.normal(loc=mean, scale=np.sqrt(variance), size=None)
	plt.plot(realTable[ts,:], label="Line"+str(ts+1));

plt.rcParams["figure.figsize"] = (20,10)
plt.title('Timeseries realizations', fontsize=20)
plt.xlabel("Time", fontsize=18)
plt.ylabel("Value",fontsize=18)
plt.xlim(0,numPoints+10)
plt.legend()
plt.show(block=False)
#np.save("realTable", realTable)

# Exercise 1.3
# No code

# Exercise 1.4.a
print("Exercise 1.4.a")
# Autocorrelation for 1st realisation
var1 = np.var(realTable[0,:])
print("Autocorrelation for the first realisation: ", round(var1, PRECISION))

# Exercise 1.4.b
print("Exercise 1.4.b")
# Autocorrelation for the remaining realisations
varss = np.var(realTable[1:,:], axis=0)
print("Autocorrelation for the remaining realisations:")
for ts in range(numTs-1):
	print(round(varss[ts], PRECISION))

# Difference between the autocorrelations
diffs = (varss - var1)/var1
print("% Difference of the realisations")
for ts in range(numTs-1):
	print(round(diffs[ts], PRECISION)*100)

# Improving the autocorrelation with ergodicity
autocor1 = sum(realTable[0, :]**2)/numPoints
print("Ergodic autocorrelation for first realisation: ",round(autocor1, PRECISION))
autocors = np.sum(realTable[1:]**2, axis=1)/numPoints
print("Ergodic autocorrelation for remaining realisations: ")
for ts in range(1, numTs-1):
	print(round(autocors[ts], PRECISION))

# NEW difference between the autocorrelations
diffsNew = (autocors - autocor1)/autocor1
print("% NEW Difference of the realisations")
for ts in range(numTs-1):
	print(round(diffsNew[ts], PRECISION)*100)

# Question 1.5
plt.figure("Realizations' autocorrelation")
kRange = 101
xaxis = np.arange(-kRange+1, kRange)
def corr_func(ts):
	correlation = []
	kappas = np.arange(kRange)
	for k in kappas:
		correlation.append(sum(ts[:len(ts) - k]*ts[k:])/len(ts))
	# Add the symmetrical part
	return list(reversed(correlation[1:])) + correlation

corrs = []
for idx,ts in enumerate(realTable):
	correlation = corr_func(ts)
	corrs.append(correlation)
	plt.plot(xaxis, correlation, label="Line"+str(idx+1))

plt.rcParams["figure.figsize"] = (20,10)
plt.title("Realizations' autocorrelation", fontsize=20)
plt.xlabel("Time", fontsize=18)
plt.ylabel("Autocorrelation",fontsize=18)
plt.xlim(-kRange+1, kRange - 1)
plt.legend()
plt.show(block=False)

# Exercise 2
print("==========================================================================================")
print("Exercise 2")
print("==========================================================================================")
# Load the data by transforming the .mat files to numpy arrays
#nscan = scipy.io.loadmat("trainscans.mat")["Nscan"]
#iscan = scipy.io.loadmat("trainscans.mat")["Iscan"]
#scan = scipy.io.loadmat("testscans.mat")["scan"]

#np.save("nscan", nscan)
#np.save("iscan", iscan)
#np.save("scan", scan)

# Unlabelled test scans
scan = np.load("scan.npy")
# Scans from normal people (L=0)
nscan = np.load("nscan.npy")
# Scans from ILD people (L=1)
iscan = np.load("iscan.npy")

# Exercise 2.1
plt.figure("Autocorrelation plot of the samples")
kRange = 101
xaxis = np.arange(-kRange+1, kRange)
def corr_func2(ts):
	correlation = []
	kappas = np.arange(kRange)
	for k in kappas:
		correlation.append(sum(ts[:len(ts) - k]*ts[k:])/len(ts))
	# Add the symmetrical part
	return list(reversed(correlation[1:])) + correlation

def plotCorrs(scans, col, sampleLabel="test"):
	corrs = []
	for idx, ts in enumerate(scans):
		correlation = corr_func2(ts)
		corrs.append(correlation)
		plt.plot(xaxis, correlation, color=col, label=sampleLabel+str(idx+1))
	return np.array(corrs)


corrsI = plotCorrs(iscan, "red", sampleLabel="ILD")
corrsN = plotCorrs(nscan, "blue", sampleLabel="Normal")
plt.legend(prop={'size': 6})
plt.title("Autocorrelation plot of the samples", fontsize=20)
plt.xlabel("k", fontsize=18);
plt.ylabel("Rx", fontsize=18);
plt.show(block=False)

# Exercise 2.2
print("Exercise 2.2")
def compute_k_star(corrsI, corrsN):
	avgCorrsI = np.average(corrsI, axis=0)
	avgCorrsN = np.average(corrsN, axis=0)
	dist = np.abs(avgCorrsI - avgCorrsN)
	k_star = np.argmax(dist)
	return k_star
k_star = compute_k_star(corrsI, corrsN)
print("Proposed value based on visual inspection: ", k_star)

# Exersise 2.3
print("Exercise 2.3")
def plotHistograms(corrsI, corrsN, k_star):
	plt.hist(corrsI[:,k_star], alpha=0.5, color="red", label="ILD");
	plt.hist(corrsN[:,k_star], alpha=0.5, color="blue", label="Normal");
	plt.legend()
	plt.title("Autocorrelation histogram", fontsize=20)
	plt.xlabel("Rx", fontsize=18)
	plt.ylabel("Frequency", fontsize=18)
	plt.legend()

plt.figure("Autocorrelation histogram")
plotHistograms(corrsI, corrsN, k_star)
print("The two classes can't be seperated perfectly")
plt.show(block=False)

# Exercise 2.4
print("Exercise 2.4")
def computeAPrioriProbs(nscan, iscan):
	pN = len(nscan)/(len(nscan) + len(iscan))
	pI = len(iscan)/(len(nscan) + len(iscan))
	return pN, pI
pN, pI = computeAPrioriProbs(nscan, iscan)
print("A priori probability of a Normal: ",pN)
print("A priori probability of a ILD: ",pI)

# Exercise 2.5
print("Exercise 2.5")
# Using sample mean:
def computeNormalDistr(corrs, k_star, color, sampleLabel="test"):
	#import ipdb; ipdb.set_trace()
	mean = np.average(corrs[:, k_star])
	std = np.std(corrs[:, k_star])
	x = np.linspace(mean - 4*std, mean + 4*std, 100)
	y = norm(mean, std)
	plt.plot(x, y.pdf(x), color=color, label=sampleLabel + " Distribution")
	return mean, std, y

plt.figure("Normal Distribution of autocorrelation values at k*")
# Compute distribution Rx|L=1
meanI, stdI, yI = computeNormalDistr(corrsI, k_star, "blue", sampleLabel="ILD")
meanN, stdN, yN = computeNormalDistr(corrsN, k_star, "orange", sampleLabel="Normal")

print("ILD Gaussian Distribution: Mean: {}, Variance: {}".format(meanI, stdI**2))
print("Normal Gaussian Distribution: Mean: {}, Variance: {}".format(meanN, stdN**2))
# Plot the density histograms of exercise 2.3
plt.hist(corrsI[:,k_star], density=True, alpha=0.5, color="blue", label="ILD Samples histogram");
plt.hist(corrsN[:,k_star], density=True, alpha=0.5, color="red", label="Normal Samples histogram");
plt.title("Normal Distribution of autocorrelation values at k*", fontsize=20)
plt.xlabel("Rx", fontsize=18)
plt.ylabel("Frequency", fontsize=18)
plt.legend()
plt.grid()
plt.show(block=False)

# Exercise 2.6
print("Exercise 2.6")
# Calculate the probabilities P(Rx_k = r)
def calcProbs(r):
	# Calculate the probability for the correlation value (demominator)
	corr_k = yI.cdf(r)*pI + yN.cdf(r)*pN

	pIR = yI.cdf(r)*pI/corr_k
	pNR = yN.cdf(r)*pN/corr_k
	return pIR, pNR

def classify(corrs, k_star):
	countI = 0
	countN = 0
	for r in corrs[:, k_star]:
		pIR, pNR = calcProbs(r)
		if pNR >= pIR:
			countN += 1
		else:
			countI += 1
	return countN, countI

# Normal Samples
print("Normal Samples validation:")
for idx, corr in enumerate(corrsN):
	normal, ild = classify(np.array([corr]), k_star)
	if normal == 1:
		pred = "Normal"
	else:
		pred = "ILD"
	print("Sample {} is estimated to be {}".format(idx+1, pred))

# ILD Samples
print("ILD Samples validation:")
for idx, corr in enumerate(corrsI):
	normal, ild = classify(np.array([corr]), k_star)
	if normal == 1:
		pred = "Normal"
	else:
		pred = "ILD"
	print("Sample {} is estimated to be {}".format(idx+1, pred))

# Exercise 2.7
print("Exercise 2.7")
# Calculate the correlations of the samples
plt.figure("Autocorrelation plot of the test samples")
corrsTest = plotCorrs(scan, "green", sampleLabel="Test")
plt.legend()
plt.title("Autocorrelation plot of the test samples", fontsize=20)
plt.xlabel("k", fontsize=18)
plt.ylabel("Rx", fontsize=18)
plt.show(block=False)
# Classify the test samples
print("Test samples estimations:")
for idx, corr in enumerate(corrsTest):
	normal, ild = classify(np.array([corr]), k_star)
	if normal == 1:
		pred = "Normal"
	else:
		pred = "ILD"
	print("Sample {} is estimated to be {}".format(idx+1, pred))
trueTest, falseTest = classify(corrsTest, k_star)
plt.show(block=False)
plt.figure("Autocorrelation histogram including test data")


plotHistograms(corrsI, corrsN, k_star)
plt.hist(corrsTest[:, k_star], color="grey", alpha=0.7, label="Test");
plt.legend()
plt.show()

