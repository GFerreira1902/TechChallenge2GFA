import numpy as np
import random
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

class GeneticAlgorithmOptimizer:
    def __init__(self, X_train, y_train, population_size=10, generations=5, mutation_rate=0.1):
        self.X_train = X_train
        self.y_train = y_train
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        
        # Espaço de busca (Genes) para o Random Forest
        # Expandir de acordo com a necessidade
        self.param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [None, 5, 10, 20],
            'min_samples_split': [2, 5, 10],
            'class_weight': ['balanced', 'balanced_subsample', None]
        }

    def generate_individual(self):
        """Cria um indivíduo (cromossomo) com hiperparâmetros aleatórios."""
        return {
            'n_estimators': random.choice(self.param_grid['n_estimators']),
            'max_depth': random.choice(self.param_grid['max_depth']),
            'min_samples_split': random.choice(self.param_grid['min_samples_split']),
            'class_weight': random.choice(self.param_grid['class_weight'])
        }

    def generate_population(self):
        """Cria a população inicial."""
        return [self.generate_individual() for _ in range(self.population_size)]

    def calculate_fitness(self, individual):
        """
        Avalia o desempenho do modelo (Fitness).
        Configuração focal para Otimização Híbrida: Recall/Sensibilidade
        """
        model = RandomForestClassifier(**individual, random_state=42)
        skf = StratifiedKFold(n_splits=3)
        
        recalls = []
        specificities = []
        f1_scores = []
        
        for train_idx, val_idx in skf.split(self.X_train, self.y_train):
            X_fold_train, X_fold_val = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
            y_fold_train, y_fold_val = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]
            
            model.fit(X_fold_train, y_fold_train)
            preds = model.predict(X_fold_val)
            
            # Métricas
            recalls.append(recall_score(y_fold_val, preds))
            f1_scores.append(f1_score(y_fold_val, preds))
            
            # Cálculo de Especificidade
            tn, fp, fn, tp = confusion_matrix(y_fold_val, preds, labels=[0,1]).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            specificities.append(specificity)
            
        avg_recall = np.mean(recalls)
        avg_specificity = np.mean(specificities)
        avg_f1 = np.mean(f1_scores)
        
        # FUNÇÃO DE FITNESS CUSTOMIZADA
        # Minimização de falsos negativos com Recall (70%) e False Positives (30%)
        # Permissão para implementações secundárias (Fairness ou F1)
        fitness = (0.7 * avg_recall) + (0.3 * avg_specificity)
        
        return fitness, avg_recall, avg_specificity, avg_f1

    def select_parents(self, population, fitness_scores):
        """Método de Seleção por Torneio para seleção em subgrupos."""
        tournament_size = 3
        selected_parents = []
        
        for _ in range(len(population)):
            tournament_indices = random.sample(range(len(population)), tournament_size)
            best_idx = max(tournament_indices, key=lambda idx: fitness_scores[idx])
            selected_parents.append(population[best_idx])
            
        return selected_parents

    def crossover(self, parent1, parent2):
        """Crossover Uniforme: Mistura os hiperparâmetros de dois pais."""
        child1 = {}
        child2 = {}
        for key in self.param_grid.keys():
            if random.random() > 0.5:
                child1[key] = parent1[key]
                child2[key] = parent2[key]
            else:
                child1[key] = parent2[key]
                child2[key] = parent1[key]
        return child1, child2

    def mutate(self, individual):
        """Mutação: Altera aleatoriamente um hiperparâmetro com base na taxa de mutação."""
        mutated = individual.copy()
        for key in self.param_grid.keys():
            if random.random() < self.mutation_rate:
                mutated[key] = random.choice(self.param_grid[key])
        return mutated

    def run(self):
        """Executa o Algoritmo Genético."""
        print(f"Iniciando Otimização GA ({self.generations} Gerações | População: {self.population_size})")
        population = self.generate_population()
        best_individual_overall = None
        best_fitness_overall = -1
        
        for generation in range(self.generations):
            print(f"--- Geração {generation + 1} ---")
            
            # 1. Avaliação (Fitness)
            fitness_results = [self.calculate_fitness(ind) for ind in population]
            fitness_scores = [res[0] for res in fitness_results]
            
            # Pega o melhor da geração
            best_idx = np.argmax(fitness_scores)
            best_fitness, best_rec, best_spec, best_f1 = fitness_results[best_idx]
            best_individual = population[best_idx]
            
            print(f"Melhor Fitness da Geração: {best_fitness:.4f} (Recall: {best_rec:.4f}, Spec: {best_spec:.4f})")
            
            if best_fitness > best_fitness_overall:
                best_fitness_overall = best_fitness
                best_individual_overall = best_individual
                
            # 2. Seleção
            parents = self.select_parents(population, fitness_scores)
            
            # 3. Crossover & Mutação (Nova População)
            next_generation = []
            for i in range(0, len(parents), 2):
                parent1 = parents[i]
                parent2 = parents[i+1] if (i+1) < len(parents) else parents[0] # Padronização de tamanho
                
                child1, child2 = self.crossover(parent1, parent2)
                
                next_generation.append(self.mutate(child1))
                next_generation.append(self.mutate(child2))
                
            # Mantém o tamanho da população original
            population = next_generation[:self.population_size]
            
        print("\n=== Fim da Otimização ===")
        print(f"Melhor Conjunto de Hiperparâmetros: {best_individual_overall}")
        print(f"Melhor Score Fitness: {best_fitness_overall:.4f}")
        
        return best_individual_overall

if __name__ == '__main__':
    # Exemplo de instância interna
    import os
    print("Módulo carregado com sucesso...")
    pass
