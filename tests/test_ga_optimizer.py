import pytest
import pandas as pd
import numpy as np
from src.ga_optimizer import GeneticAlgorithmOptimizer

@pytest.fixture
def mock_data():
    """Cria um dataset semente pequeno para os testes do Algoritmo Genético."""
    np.random.seed(42)
    X_train = pd.DataFrame(np.random.rand(100, 5), columns=['F1', 'F2', 'F3', 'F4', 'F5'])
    y_train = pd.Series(np.random.randint(0, 2, 100))
    return X_train, y_train

def test_ga_initialization(mock_data):
    """Valida se a inicialização do otimizador define corretamente os atributos da classe."""
    X_train, y_train = mock_data
    ga = GeneticAlgorithmOptimizer(X_train, y_train, population_size=4, generations=2)
    
    assert ga.population_size == 4
    assert ga.generations == 2
    assert len(ga.param_grid.keys()) == 4

def test_generate_population(mock_data):
    """Verifica se a população de parâmetros gerada aleatoriamente possui as chaves apropriadas."""
    X_train, y_train = mock_data
    ga = GeneticAlgorithmOptimizer(X_train, y_train, population_size=5)
    
    population = ga.generate_population()
    
    assert len(population) == 5
    for individual in population:
        assert 'n_estimators' in individual
        assert 'max_depth' in individual
        assert 'min_samples_split' in individual
        assert 'class_weight' in individual

def test_crossover(mock_data):
    """Valida se o evento de crossover mistura corretamente os hiperparâmetros de dois indivíduos."""
    X_train, y_train = mock_data
    ga = GeneticAlgorithmOptimizer(X_train, y_train)
    
    parent1 = {'n_estimators': 50, 'max_depth': 5, 'min_samples_split': 2, 'class_weight': None}
    parent2 = {'n_estimators': 300, 'max_depth': 20, 'min_samples_split': 10, 'class_weight': 'balanced'}
    
    child1, child2 = ga.crossover(parent1, parent2)
    
    assert list(child1.keys()) == list(parent1.keys())
    assert list(child2.keys()) == list(parent2.keys())
    
    # Os atributos dos filhos devem vir de um dos dois pais
    assert child1['n_estimators'] in [50, 300]
    assert child2['class_weight'] in [None, 'balanced']
