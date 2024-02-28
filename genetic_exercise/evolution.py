import random
import os
import chardet
from typing import List, Dict

def allocate_time_genetic(input_data: str, population_size: int = 100, generations: int = 100, mutation_rate: float = 0.1) -> List[str]:
    """
    Allocates time using a genetic algorithm based on the provided input_data.

    Parameters:
    - input_data (str): Input data in the specified format.
    - population_size (int): Population size for the genetic algorithm.
    - generations (int): Number of generations in the genetic algorithm.
    - mutation_rate (float): Mutation rate in the genetic algorithm.

    Returns:
    - List[str]: List of strings representing the resulting time allocation.
    """
    def fitness(solution: Dict[str, List[str]]) -> int:
        """
        Evaluates the fitness of a solution.

        Parameters:
        - solution (Dict[str, List[str]]): Solution to be evaluated.

        Returns:
        - int: Fitness score.
        """
        conflicts = 0
        for machine_allocations in solution.values():
            allocated_students = set()
            for student in machine_allocations:
                if student in allocated_students:
                    conflicts += 1
                allocated_students.add(student)
        return conflicts

    def generate_initial_population(population_size: int, students: Dict[str, Dict[str, int]]) -> List[Dict[str, List[str]]]:
        """
        Generates the initial population for the genetic algorithm.

        Parameters:
        - population_size (int): Population size.
        - students (Dict[str, Dict[str, int]]): Dictionary representing machines and students.

        Returns:
        - List[Dict[str, List[str]]]: Initial population.
        """
        machines = list(students.keys())
        initial_population = []
        for _ in range(population_size):
            solution = {machine: list(students[machine].keys()) for machine in machines}
            for machine in machines:
                random.shuffle(solution[machine])
            initial_population.append(solution)
        return initial_population

    def crossover(parent1: Dict[str, List[str]], parent2: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        Performs crossover of two parents to generate a child.

        Parameters:
        - parent1 (Dict[str, List[str]]): First parent.
        - parent2 (Dict[str, List[str]]): Second parent.

        Returns:
        - Dict[str, List[str]]: Child solution.
        """
        crossover_point = random.randint(1, len(parent1) - 1)
        child = {}
        for key in parent1:
            if random.random() < 0.5:
                child[key] = parent1[key][:crossover_point] + parent2[key][crossover_point:]
            else:
                child[key] = parent2[key][:crossover_point] + parent1[key][crossover_point:]
        return child

    def mutate(individual: Dict[str, List[str]], students: Dict[str, Dict[str, int]]) -> Dict[str, List[str]]:
        """
        Mutates an individual solution.

        Parameters:
        - individual (Dict[str, List[str]]): Individual solution.
        - students (Dict[str, Dict[str, int]]): Dictionary representing machines and students.

        Returns:
        - Dict[str, List[str]]: Mutated individual solution.
        """
        mutation_probability = 0.1
        mutated_individual = individual.copy()
        for machine, allocations in mutated_individual.items():
            if random.random() < mutation_probability:
                if len(allocations) > 1:
                    index1, index2 = random.sample(range(len(allocations)), 2)
                    allocations[index1], allocations[index2] = allocations[index2], allocations[index1]
        return mutated_individual

    def genetic_algorithm(population_size: int, generations: int, mutation_rate: float, students: Dict[str, Dict[str, int]]) -> Dict[str, List[str]]:
        """
        Executes the genetic algorithm.

        Parameters:
        - population_size (int): Population size.
        - generations (int): Number of generations.
        - mutation_rate (float): Mutation rate.
        - students (Dict[str, Dict[str, int]]): Dictionary representing machines and students.

        Returns:
        - Dict[str, List[str]]: Best solution.
        """
        population = generate_initial_population(population_size, students)
        for _ in range(generations):
            fitness_scores = [fitness(individual) for individual in population]

            if all(fit == 0 for fit in fitness_scores):
                parents = random.choices(population, k=population_size)
            else:
                weights = [1/fit if fit != 0 else 1 for fit in fitness_scores]
                parents = random.choices(population, weights=weights, k=population_size)

            if len(parents) < 2:
                continue

            offspring = []
            while len(offspring) < population_size:
                parent1 = random.choice(parents)
                parent2 = random.choice(parents)
                child = crossover(parent1, parent2)
                offspring.append(child)

            for i in range(population_size):
                if random.random() < mutation_rate:
                    offspring[i] = mutate(offspring[i], students)

            population = offspring

        best_solution = max(population, key=fitness)
        return best_solution

    def convert_solution_to_output(best_solution: Dict[str, List[str]]) -> List[str]:
        """
        Converts the best solution to the desired output format.

        Parameters:
        - best_solution (Dict[str, List[str]]): Best solution.

        Returns:
        - List[str]: Output lines.
        """
        output_lines = []
        for machine, allocations in best_solution.items():
            output_line = f"{machine}:"
            for student in allocations:
                output_line += f"{student};"
            output_lines.append(output_line)
        return output_lines

    machines = {}

    lines = input_data.split('\n')
    for line in lines:
        if line:
            parts = line.split(':')
            machine = parts[0].strip()
            students = {student.split('=')[0].strip(): int(student.split('=')[1].strip()) for student in parts[1].split(';') if student}
            machines[machine] = students

    best_solution = genetic_algorithm(population_size, generations, mutation_rate, machines)
    output_lines = convert_solution_to_output(best_solution)

    return output_lines

def detect_encoding(file_path: str) -> str:
    """
    Detects the encoding of a file.

    Parameters:
    - file_path (str): Path of the file.

    Returns:
    - str: Detected encoding.
    """
    with open(file_path, 'rb') as file:
        result = chardet.detect(file.read())
    return result['encoding']

# 'main'
input_file_path = os.path.join('entrada_50', 'entrada_1.txt')

if os.path.exists(input_file_path):
    file_encoding = detect_encoding(input_file_path)

    with open(input_file_path, 'r', encoding=file_encoding) as file:
        input_data = file.read()

    result = allocate_time_genetic(input_data)
    print(result)
else:
    print(f'The file {input_file_path} was not found.')
