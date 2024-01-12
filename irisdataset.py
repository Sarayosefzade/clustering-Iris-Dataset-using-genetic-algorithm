import numpy as np
import random
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

iris = load_iris()

# Set the random seed for reproducibility
random.seed(42)
np.random.seed(42)

def initialize_population(population_size, num_clusters, num_samples):
    population = []
    for _ in range(population_size):
        genotype = [random.randint(0, num_clusters-1) for _ in range(num_samples)]
        population.append(genotype)
    return population

def calculate_centroids(data, labels, num_clusters):
    centroids = []
    for cluster in range(num_clusters):
        cluster_data = data[labels == cluster]
        if len(cluster_data) == 0:
            # Handle empty clusters by reallocating a random data point to it
            random_index = random.randint(0, len(data)-1)
            cluster_data = data[random_index]
        centroid = np.mean(cluster_data, axis=0)
        centroids.append(centroid)
    return centroids

def assign_clusters(data, centroids):
    labels = []
    for point in data:
        if len(centroids) == 0:
            # Handle the case when there are no centroids
            cluster = random.randint(0, len(data)-1)
        else:
            distances = [np.linalg.norm(point - centroid) for centroid in centroids]
            cluster = np.argmin(distances)
        labels.append(cluster)
    return labels

def evaluate_fitness(population, data, num_clusters):
    fitness_scores = []
    for genotype in population:
        centroids = calculate_centroids(data, genotype, num_clusters)
        predicted_labels = assign_clusters(data, centroids)
        accuracy = accuracy_score(iris.target, predicted_labels)
        fitness_scores.append(accuracy)
    return fitness_scores

def selection(population, fitness_scores, num_parents):
    selected_indices = np.argsort(fitness_scores)[-num_parents:]
    selected_population = [population[i] for i in selected_indices]
    return selected_population

def crossover(parents, population_size):
    num_parents = len(parents)
    num_offsprings = population_size - num_parents
    offsprings = []
    for _ in range(num_offsprings):
        parent1, parent2 = random.sample(parents, 2)
        offspring = []
        for i in range(len(parent1)):
            if random.random() < 0.5:
                offspring.append(parent1[i])
            else:
                offspring.append(parent2[i])
        offsprings.append(offspring)
    return parents + offsprings
def mutation(offsprings, mutation_rate, num_clusters):
    mutated_offsprings = []
    for offspring in offsprings:
        mutated_offspring = []
        for gene in offspring:
            if random.random() < mutation_rate:
                mutated_gene = random.randint(0, num_clusters-1)
                mutated_offspring.append(mutated_gene)
            else:
                mutated_offspring.append(gene)
        mutated_offsprings.append(mutated_offspring)
    return mutated_offsprings
def genetic_algorithm(data, num_clusters, population_size, num_generations, num_parents, mutation_rate):
    population = initialize_population(population_size, num_clusters, len(data))

    best_accuracy = 0.0
    best_genotype = None
    
    for _ in range(num_generations):
        fitness_scores = evaluate_fitness(population, data, num_clusters)
        max_fitness = max(fitness_scores)
        max_index = np.argmax(fitness_scores)
        
        if max_fitness > best_accuracy:
            best_accuracy = max_fitness
            best_genotype = population[max_index]
        
        parents = selection(population, fitness_scores, num_parents)
        population = crossover(parents, population_size)
        population = mutation(population, mutation_rate, num_clusters)
    
    centroids = calculate_centroids(data, best_genotype, num_clusters)
    predicted_labels = assign_clusters(data, centroids)
    accuracy = accuracy_score(iris.target, predicted_labels)
    
    return best_genotype, accuracy, predicted_labels

def perform_kmeans(data, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(data)
    return kmeans.labels_

# Setting up the parameters
num_clusters = 3

# Running K-means clustering separately
kmeans_labels = perform_kmeans(iris.data, num_clusters)
kmeans_accuracy = accuracy_score(iris.target, kmeans_labels)

print("K-means Accuracy:", kmeans_accuracy)


# population_size = 1000
# num_generations = 150
# num_parents = 10
# mutation_rate = 0.1
population_size = 1000
num_generations = 200
num_parents = 10
mutation_rate = 0.3

best_genotype, accuracy, _ = genetic_algorithm(iris.data, num_clusters=num_clusters, population_size=population_size,
                                            num_generations=num_generations, num_parents=num_parents, mutation_rate=mutation_rate)

print("Best Genotype:", best_genotype)
print("Genetic Algorithm Accuracy:", accuracy)