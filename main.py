# Cem OZAN - 250201003
import numpy as np

import warnings

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

import math

from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D

def calculate_risk(test_data,MU_matrix_1, bivariate_covariance_1,MU_matrix_2, bivariate_covariance_2):
    # 1 - P(Ci|x)
    # P(Ci|x) = P(x|Ci)P(Ci)/P(x)
    # P(x) = Toplam[(P(x|Ck)*P(Ck))]
    MU_matrix_1 = np.array([mu_attribute1_1, mu_attribute2_1])
    MU_matrix_2 = np.array([mu_attribute1_2, mu_attribute2_2])
    
    bivariate_covariance_1 = np.array([
                                    [cov_11_1 * cov_11_1, cov_12_1], 
                                    [cov_12_1, cov_22_1 * cov_22_1]])
    bivariate_covariance_2 = np.array([
                                    [cov_11_2 * cov_11_2, cov_12_2], 
                                    [cov_12_2, cov_22_2 * cov_22_2]])
    for i in test_data:
        x_vector = np.array([i[15],i[17]])
        x = 0
        x = multivariate_gaussian(x_vector, MU_matrix_1, bivariate_covariance_1, 0.7) * 0.7
        x += multivariate_gaussian(x_vector, MU_matrix_2, bivariate_covariance_2, 0.3) * 0.3
        posterior1 = (multivariate_gaussian(x_vector, MU_matrix_1, bivariate_covariance_1, 0.7) * 0.7) / x
        
        posterior2 = (multivariate_gaussian(x_vector, MU_matrix_2, bivariate_covariance_2, 0.3) * 0.3) / x
        
        risk1 = 1 - posterior1
        risk2 = 1 - posterior2
        
        print("Risk of selection Class 1: ", risk1, " for the data: ", x_vector)
        print("Risk of selection Class 2: ", risk2, " for the data: ", x_vector)

def multivariate_gaussian(pos, mu, Sigma, prior):    

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)
    result = np.exp(-fac / 2) / N #likelihood
    
    lp = math.log(prior)
    return  result + lp

#The best graph were captured with 16-18 columns in data set. So I choose them.
def plot_gaussian(MU_matrix_1, bivariate_covariance_1,MU_matrix_2, bivariate_covariance_2):
    #Create grid and multivariate normal
    x = np.linspace(0,3,100)
    y = np.linspace(0,3,100)
    X, Y = np.meshgrid(x,y)

    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X; pos[:, :, 1] = Y
    rv = multivariate_normal(MU_matrix_1, bivariate_covariance_1)

    #Make a 3D plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, rv.pdf(pos),cmap='viridis',linewidth=0)
    ax.set_title("For Clas 1")
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    # --------------------------------------------------------------------
    x = np.linspace(0,3,100)
    y = np.linspace(0,3,100)
    X, Y = np.meshgrid(x,y)

    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X; pos[:, :, 1] = Y
    rv = multivariate_normal(MU_matrix_2, bivariate_covariance_2)

    #Make a 3D plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, rv.pdf(pos),cmap='viridis',linewidth=0)
    ax.set_title("For Clas 2")
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.show()
    
    
def find_max_likelihood(g1, g2):
    likelihood_list = [g1, g2]
    return max(likelihood_list)

    
def do_classification(test_data, mu_attribute2_1, mu_attribute2_2, mu_attribute1_1, mu_attribute1_2, cov_12_1, cov_12_2, cov_11_1, cov_11_2, cov_22_1, cov_22_2):
    decision_list = []  # to store classification results to calculate accuracy later.
    MU_matrix_1 = np.array([mu_attribute1_1, mu_attribute2_1])
    MU_matrix_2 = np.array([mu_attribute1_2, mu_attribute2_2])
    
    bivariate_covariance_1 = np.array([
                                    [cov_11_1 * cov_11_1, cov_12_1], 
                                    [cov_12_1, cov_22_1 * cov_22_1]])
    bivariate_covariance_2 = np.array([
                                    [cov_11_2 * cov_11_2, cov_12_2], 
                                    [cov_12_2, cov_22_2 * cov_22_2]])
    
    prior1 = 0.7
    prior2 = 0.3
    for i in test_data:
        
        x_vector = np.array([i[15],i[17]])
        g1 = multivariate_gaussian(x_vector, MU_matrix_1, bivariate_covariance_1, prior1)
        g2 = multivariate_gaussian(x_vector, MU_matrix_2, bivariate_covariance_2, prior2)
        # g1 = multivariate_normal(x_vector,MU_matrix_1, bivariate_covariance_1)
        # g2 = multivariate_normal(x_vector,MU_matrix_2, bivariate_covariance_2)
        
        max_likelihood = find_max_likelihood(g1,g2)
        if max_likelihood == g1:
            decision_list.append(1)
        else:
            decision_list.append(2)
        
    return decision_list


data = pd.read_csv('german.data',names=["Status of existing checking account", "Duration in month","Credit history","Purpose","Credit amount","Savings account/bonds","Present employment since","Installment rate in percentage of disposable income","Personal status and sex","Other debtors / guarantors","Present residence since","Property","Age in years","Other installment plans","Housing","Number of existing credits at this bank","Job","Number of people being liable to provide maintenance for","Telephone","foreign worker","class"], sep='\s+')

#There are total 700 rows in Class 1, and 300 rows in Class 2. P(C1) = 0.7, P(C2) = 0.3 => Priors

warnings.filterwarnings("ignore")

accuracy = 0  # total accuracy
# program will run for 500 times to find average accuracy
accuracies=[] # to store individual accuracies
print("...Calculating Discriminants and Plotting Graphs...")
for i in range(0, 500):
    # split data train & test
    X = np.array(data.iloc[:, 0:21])
    y = np.array(data['class'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    
    
    # calculate MUs and COVs(MLE parameters) of each class
    attribute1_class1, attribute1_class2, attribute2_class1, attribute2_class2 = 0, 0, 0, 0
    
    cov_12_1, cov_12_2, cov_11_1, cov_11_2, cov_22_1, cov_22_2 = 0, 0, 0, 0, 0, 0
    
    X_train_1 = []
    X_train_2 = []
    
    for i in range(len(X_train)):
        if X_train[i][20] == 1:
            X_train_1.append(X_train[i])
            
    for i in range(len(X_train)):
        if X_train[i][20] == 2:
            X_train_2.append(X_train[i])
    
    
    for k in range(len(X_train_1)):
        attribute1_class1 += float(X_train_1[k][15])
        
    for k in range(len(X_train_1)):
        attribute2_class1 += float(X_train_1[k][17])

    for k in range(len(X_train_2)):
        attribute1_class2 += float(X_train_2[k][15])
    
    for k in range(len(X_train_2)):
        attribute2_class2 += float(X_train_2[k][17])
    

    mu_attribute1_1 = attribute1_class1 / len(X_train_1)
    mu_attribute1_2 = attribute1_class2 / len(X_train_2)
    
    mu_attribute2_1 = attribute2_class1 / len(X_train_1)
    mu_attribute2_2 = attribute2_class2 / len(X_train_2)
    
    
    for k in range(len(X_train_1)):
        cov_12_1 += (float(X_train_1[k][17]) - mu_attribute2_1) * (float(X_train_1[k][15]) - mu_attribute1_1)
    cov_12_1 = cov_12_1 / len(X_train_1)
    
    for k in range(len(X_train_2)):
        cov_12_2 += (float(X_train_2[k][17]) - mu_attribute2_2) * (float(X_train_2[k][15]) - mu_attribute1_2)
    cov_12_2 = cov_12_2 / len(X_train_2)
    
    for k in range(len(X_train_1)):
        cov_11_1 += (float(X_train_1[k][17]) - mu_attribute2_1) * (float(X_train_1[k][17]) - mu_attribute2_1)
    cov_11_1 = cov_11_1 / len(X_train_1)
    
    for k in range(len(X_train_2)):
        cov_11_2 += (float(X_train_2[k][17]) - mu_attribute2_2) * (float(X_train_2[k][17]) - mu_attribute2_2)
    cov_11_2 = cov_11_2 / len(X_train_2)
    
    for k in range(len(X_train_1)):
        cov_22_1 += (float(X_train_1[k][15]) - mu_attribute1_1) * (float(X_train_1[k][15]) - mu_attribute1_1)
    cov_22_1 = cov_22_1 / len(X_train_1)
    
    for k in range(len(X_train_2)):
        cov_22_2 += (float(X_train_2[k][15]) - mu_attribute1_2) * (float(X_train_2[k][15]) - mu_attribute1_2)
    cov_22_2 = cov_22_2 / len(X_train_2)
    
    
    # classification_result is an array that stores the result of classification
    classification_result = do_classification(X_test, mu_attribute2_1, mu_attribute2_2, mu_attribute1_1, mu_attribute1_2, cov_12_1, cov_12_2, cov_11_1, cov_11_2, cov_22_1, cov_22_2)
    # compare classification results with y_test
    individual_accuracy = accuracy_score(y_test, classification_result)
    accuracy = accuracy + individual_accuracy
    accuracies.append(individual_accuracy)
    
    cost = 0
    for i in range(len(y_test)):
        if(y_test[i] == 1 and classification_result[i] == 2):
            cost += 1
        elif(y_test[i] == 2 and classification_result[i] == 1):
            cost += 5




MU_matrix_1 = np.array([mu_attribute1_1, mu_attribute2_1])
MU_matrix_2 = np.array([mu_attribute1_2, mu_attribute2_2])

bivariate_covariance_1 = np.array([
                                [cov_11_1 * cov_11_1, cov_12_1], 
                                [cov_12_1, cov_22_1 * cov_22_1]])
bivariate_covariance_2 = np.array([
                                [cov_11_2 * cov_11_2, cov_12_2], 
                                [cov_12_2, cov_22_2 * cov_22_2]])

calculate_risk(X_test,MU_matrix_1, bivariate_covariance_1,MU_matrix_2, bivariate_covariance_2)
print("Average Accuracy = %", (accuracy/500.0)*100)
print("Max Accuracy = %", max(accuracies)*100)
print("Cost of choosing wrong class: ", cost)
plot_gaussian(MU_matrix_1, bivariate_covariance_1, MU_matrix_2, bivariate_covariance_2)
