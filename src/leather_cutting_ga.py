"""
Algoritmo Genético para Optimización de Cortes Circulares en Piel de Carnero
===========================================================================

Este programa utiliza algoritmos genéticos para encontrar la mejor disposición
de cortes circulares en una piel de carnero, maximizando el número de círculos
que se pueden obtener sin solapamiento.

Autor: [Su nombre]
Fecha: Septiembre 2025
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle
import pandas as pd
import random
import json
from typing import List, Tuple, Dict, Optional
import os
from datetime import datetime


class LeatherCuttingGA:
    """
    Clase principal que implementa un Algoritmo Genético para optimizar
    el corte de círculos en una piel de carnero.
    """
    
    def __init__(self, image_path: str, circle_diameter_cm: float = 20.0, 
                 pixels_per_cm: float = 5.0, population_size: int = 100,
                 generations: int = 500, mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8, elite_size: int = 10):
        """
        Inicializa el algoritmo genético.
        
        Args:
            image_path: Ruta a la imagen de la piel
            circle_diameter_cm: Diámetro de los círculos en cm
            pixels_per_cm: Conversión de píxeles a cm (5 píxeles = 1 cm)
            population_size: Tamaño de la población
            generations: Número de generaciones
            mutation_rate: Tasa de mutación
            crossover_rate: Tasa de cruza
            elite_size: Número de individuos élite a conservar
        """
        self.image_path = image_path
        self.circle_diameter_cm = circle_diameter_cm
        self.circle_radius_cm = circle_diameter_cm / 2.0
        self.pixels_per_cm = pixels_per_cm
        self.circle_radius_pixels = int(self.circle_radius_cm * pixels_per_cm)
        
        # Parámetros del GA
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        
        # Cargar y procesar imagen
        self.original_image = None
        self.binary_image = None
        self.valid_positions = []
        self.image_height = 0
        self.image_width = 0
        
        # Resultados
        self.best_solution = None
        self.fitness_history = []
        self.generation_stats = []
        
        self.load_and_process_image()
    
    def load_and_process_image(self):
        """Carga la imagen original y genera la versión binarizada."""
        print("Cargando y procesando imagen...")
        
        # Cargar imagen original
        self.original_image = cv2.imread(self.image_path)
        if self.original_image is None:
            raise ValueError(f"No se pudo cargar la imagen: {self.image_path}")
        
        # Convertir a RGB para matplotlib
        self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        self.image_height, self.image_width = self.original_image.shape[:2]
        
        # Crear imagen binarizada por umbralización
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)
        
        # Usar umbral de Otsu para separar piel (oscuro) del fondo (claro)
        threshold_value, self.binary_image = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        
        print(f"Umbral calculado: {threshold_value}")
        print(f"Dimensiones de imagen: {self.image_width}x{self.image_height} píxeles")
        print(f"Dimensiones reales: {self.image_width/self.pixels_per_cm:.1f}x{self.image_height/self.pixels_per_cm:.1f} cm")
        
        # Guardar imagen binarizada
        self._save_binary_outputs()
        
        # Encontrar posiciones válidas para los centros de los círculos
        self._find_valid_positions()
    
    def _save_binary_outputs(self):
        """Guarda la imagen binarizada y el archivo CSV."""
        # Directorio de salida
        output_dir = os.path.dirname(self.image_path)
        
        # Guardar imagen binarizada PNG
        binary_path = os.path.join(output_dir, "piel_carnero_bin.png")
        cv2.imwrite(binary_path, self.binary_image)
        print(f"Imagen binarizada guardada: {binary_path}")
        
        # Guardar como CSV
        csv_path = os.path.join(output_dir, "piel_carnero_bin.csv")
        np.savetxt(csv_path, self.binary_image, delimiter=',', fmt='%d')
        print(f"Datos binarios guardados: {csv_path}")
    
    def _find_valid_positions(self):
        """
        Encuentra todas las posiciones válidas donde se puede colocar
        completamente un círculo dentro del área de piel.
        OPTIMIZADO: Usa muestreo espacial para reducir posiciones candidatas.
        """
        print("Calculando posiciones válidas para círculos...")
        
        self.valid_positions = []
        margin = self.circle_radius_pixels
        
        # Crear kernel circular para la operación morfológica
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                         (2*self.circle_radius_pixels, 2*self.circle_radius_pixels))
        
        # Erosión: reduce el área donde un círculo puede colocarse completamente
        eroded = cv2.erode(self.binary_image, kernel, iterations=1)
        
        # OPTIMIZACIÓN: Muestreo espacial para reducir candidatos
        # En lugar de usar todos los píxeles, usar una grilla con espaciado
        spacing = max(5, self.circle_radius_pixels // 4)  # Espaciado adaptativo
        
        valid_positions_all = []
        for y in range(0, self.image_height, spacing):
            for x in range(0, self.image_width, spacing):
                if y < eroded.shape[0] and x < eroded.shape[1] and eroded[y, x] == 255:
                    valid_positions_all.append((x, y))
        
        # Limitar número máximo de posiciones para mejor rendimiento
        max_positions = min(10000, len(valid_positions_all))  # Máximo 10k posiciones
        if len(valid_positions_all) > max_positions:
            # Muestreo aleatorio uniforme
            indices = np.random.choice(len(valid_positions_all), max_positions, replace=False)
            self.valid_positions = [valid_positions_all[i] for i in indices]
        else:
            self.valid_positions = valid_positions_all
        
        print(f"Posiciones válidas encontradas: {len(self.valid_positions)} (optimizado)")
        
        if len(self.valid_positions) == 0:
            print("ADVERTENCIA: No se encontraron posiciones válidas para círculos de este tamaño.")
    
    def create_individual(self) -> List[Tuple[int, int]]:
        """
        Crea un individuo (solución) aleatoria.
        Un individuo es una lista de posiciones (x, y) para los centros de los círculos.
        OPTIMIZADO: Límites más conservadores y algoritmo greedy mejorado.
        """
        if len(self.valid_positions) == 0:
            return []
        
        # OPTIMIZACIÓN: Límites más conservadores para evitar bucles largos
        max_circles = min(30, len(self.valid_positions) // 50)  # Más conservador
        if max_circles < 1:
            max_circles = 1
        
        # Usar menos círculos iniciales para convergencia más rápida
        num_circles = random.randint(1, max(1, max_circles // 2))
        
        circles = []
        attempts = 0
        max_attempts = min(100, num_circles * 5)  # Menos intentos por círculo
        
        # OPTIMIZACIÓN: Usar enfoque greedy con selección espacial inteligente
        available_positions = self.valid_positions.copy()
        
        while len(circles) < num_circles and attempts < max_attempts and available_positions:
            # Seleccionar posición aleatoria de las disponibles
            pos_idx = random.randint(0, len(available_positions) - 1)
            pos = available_positions[pos_idx]
            
            # Verificar que no se solape con círculos existentes
            if self._is_valid_circle_position(pos, circles):
                circles.append(pos)
                # OPTIMIZACIÓN: Remover posiciones cercanas para evitar futuros conflictos
                available_positions = self._remove_conflicting_positions(pos, available_positions)
            else:
                # Remover posición que causó conflicto
                available_positions.pop(pos_idx)
            
            attempts += 1
        
        return circles
    
    def _remove_conflicting_positions(self, placed_pos: Tuple[int, int], 
                                    available_positions: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        OPTIMIZACIÓN: Remueve posiciones que estarían muy cerca del círculo colocado.
        Esto reduce significativamente el número de verificaciones futuras.
        """
        x_placed, y_placed = placed_pos
        min_distance_sq = (2 * self.circle_radius_pixels) ** 2  # Distancia al cuadrado para evitar sqrt
        
        filtered_positions = []
        for x, y in available_positions:
            distance_sq = (x - x_placed)**2 + (y - y_placed)**2
            if distance_sq >= min_distance_sq:
                filtered_positions.append((x, y))
        
        return filtered_positions
    
    def _is_valid_circle_position(self, pos: Tuple[int, int], 
                                existing_circles: List[Tuple[int, int]]) -> bool:
        """
        Verifica si una posición es válida para colocar un círculo
        (no se solapa con círculos existentes).
        OPTIMIZADO: Usa distancia al cuadrado para evitar sqrt.
        """
        x, y = pos
        min_distance_sq = (2 * self.circle_radius_pixels) ** 2  # Evitar sqrt
        
        for ex, ey in existing_circles:
            distance_sq = (x - ex)**2 + (y - ey)**2
            if distance_sq < min_distance_sq:
                return False
        
        return True
    
    def fitness(self, individual: List[Tuple[int, int]]) -> float:
        """
        Calcula la aptitud de un individuo.
        Fitness = número de círculos válidos (sin solapamiento) que caben en la piel.
        OPTIMIZADO: Evita verificaciones costosas innecesarias.
        """
        if not individual:
            return 0.0
        
        valid_circles = []
        # OPTIMIZACIÓN: Usar set para verificación O(1) en lugar de lista O(n)
        valid_positions_set = set(self.valid_positions) if len(self.valid_positions) > 1000 else self.valid_positions
        
        for pos in individual:
            # OPTIMIZACIÓN: Verificar primero si está en posiciones válidas (más rápido)
            if isinstance(valid_positions_set, set):
                if pos not in valid_positions_set:
                    continue
            else:
                if pos not in valid_positions_set:
                    continue
            
            # Luego verificar solapamiento
            if self._is_valid_circle_position(pos, valid_circles):
                valid_circles.append(pos)
        
        # Fitness principal: número de círculos válidos
        fitness_score = len(valid_circles)
        
        # OPTIMIZACIÓN: Bonus simplificado para mejor rendimiento
        if len(valid_circles) > 1:
            # Calcular solo algunas distancias representativas en lugar de todas
            sample_size = min(5, len(valid_circles))
            if len(valid_circles) > sample_size:
                # Muestrear círculos para calcular diversidad
                sampled_circles = random.sample(valid_circles, sample_size)
            else:
                sampled_circles = valid_circles
            
            total_distance = 0
            count = 0
            for i in range(len(sampled_circles)):
                for j in range(i+1, len(sampled_circles)):
                    x1, y1 = sampled_circles[i]
                    x2, y2 = sampled_circles[j]
                    # Usar distancia Manhattan aproximada (más rápida)
                    dist = abs(x1-x2) + abs(y1-y2)
                    total_distance += dist
                    count += 1
            
            if count > 0:
                avg_distance = total_distance / count
                # Bonus reducido para enfocarse en cantidad
                distance_bonus = min(0.2, avg_distance / (self.circle_radius_pixels * 8))
                fitness_score += distance_bonus
        
        return fitness_score
    
    def selection(self, population: List[List[Tuple[int, int]]], 
                 fitness_scores: List[float]) -> List[Tuple[int, int]]:
        """Selección por torneo."""
        tournament_size = 3
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_idx].copy()
    
    def crossover(self, parent1: List[Tuple[int, int]], 
                 parent2: List[Tuple[int, int]]) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Cruza uniforme: combina círculos de ambos padres evitando solapamientos.
        """
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        child1, child2 = [], []
        
        # Crear pools de círculos de ambos padres
        all_circles_p1 = parent1.copy()
        all_circles_p2 = parent2.copy()
        
        # Para cada hijo, seleccionar círculos alternativamente
        for circles, other_circles in [(all_circles_p1, all_circles_p2), (all_circles_p2, all_circles_p1)]:
            child = child1 if circles == all_circles_p1 else child2
            
            # Mezclar círculos de ambos padres
            combined = circles + other_circles
            random.shuffle(combined)
            
            for pos in combined:
                if self._is_valid_circle_position(pos, child) and len(child) < 50:
                    child.append(pos)
        
        return child1, child2
    
    def mutate(self, individual: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Mutación: puede agregar, eliminar o mover círculos.
        """
        if random.random() > self.mutation_rate:
            return individual
        
        mutated = individual.copy()
        mutation_type = random.choice(['add', 'remove', 'move'])
        
        if mutation_type == 'add' and len(self.valid_positions) > 0:
            # Agregar un círculo en posición válida
            for _ in range(10):  # Intentos limitados
                pos = random.choice(self.valid_positions)
                if self._is_valid_circle_position(pos, mutated):
                    mutated.append(pos)
                    break
        
        elif mutation_type == 'remove' and len(mutated) > 0:
            # Eliminar un círculo aleatorio
            mutated.pop(random.randint(0, len(mutated) - 1))
        
        elif mutation_type == 'move' and len(mutated) > 0 and len(self.valid_positions) > 0:
            # Mover un círculo a nueva posición
            idx = random.randint(0, len(mutated) - 1)
            old_pos = mutated[idx]
            
            for _ in range(10):  # Intentos limitados
                new_pos = random.choice(self.valid_positions)
                temp_circles = mutated[:idx] + mutated[idx+1:]
                if self._is_valid_circle_position(new_pos, temp_circles):
                    mutated[idx] = new_pos
                    break
        
        return mutated
    
    def run_ga(self) -> Dict:
        """
        Ejecuta el algoritmo genético completo.
        
        Returns:
            Diccionario con resultados del GA
        """
        print(f"\nIniciando Algoritmo Genético...")
        print(f"Parámetros:")
        print(f"  - Población: {self.population_size}")
        print(f"  - Generaciones: {self.generations}")
        print(f"  - Diámetro círculo: {self.circle_diameter_cm} cm ({self.circle_radius_pixels*2} píxeles)")
        print(f"  - Tasa mutación: {self.mutation_rate}")
        print(f"  - Tasa cruza: {self.crossover_rate}")
        
        # Inicializar población
        print("Creando población inicial...")
        population = []
        for i in range(self.population_size):
            individual = self.create_individual()
            population.append(individual)
            if (i + 1) % 10 == 0:
                print(f"  Individuos creados: {i + 1}/{self.population_size}")
        print("✅ Población inicial creada")
        
        best_fitness_ever = 0
        best_individual_ever = []
        
        for generation in range(self.generations):
            # Evaluar fitness
            fitness_scores = [self.fitness(ind) for ind in population]
            
            # Estadísticas de generación
            max_fitness = max(fitness_scores)
            avg_fitness = np.mean(fitness_scores)
            min_fitness = min(fitness_scores)
            
            # Actualizar mejor solución global
            if max_fitness > best_fitness_ever:
                best_fitness_ever = max_fitness
                best_idx = fitness_scores.index(max_fitness)
                best_individual_ever = population[best_idx].copy()
            
            # Guardar estadísticas
            self.fitness_history.append(max_fitness)
            self.generation_stats.append({
                'generation': generation,
                'max_fitness': max_fitness,
                'avg_fitness': avg_fitness,
                'min_fitness': min_fitness,
                'population_diversity': len(set(str(ind) for ind in population))
            })
            
            # Mostrar progreso más frecuentemente para troubleshooting
            if generation % 10 == 0 or generation == self.generations - 1:
                print(f"Gen {generation:3d}: Max={max_fitness:.2f}, Avg={avg_fitness:.2f}, "
                      f"Mejor global={best_fitness_ever:.2f}")
                # Mostrar información adicional ocasionalmente
                if generation % 25 == 0:
                    print(f"    Círculos en mejor solución: {len(best_individual_ever)}")
            
            # Crear nueva población
            new_population = []
            
            # Elitismo: conservar mejores individuos
            sorted_indices = sorted(range(len(fitness_scores)), 
                                  key=lambda i: fitness_scores[i], reverse=True)
            for i in range(self.elite_size):
                new_population.append(population[sorted_indices[i]].copy())
            
            # Generar resto de población mediante cruza y mutación
            while len(new_population) < self.population_size:
                parent1 = self.selection(population, fitness_scores)
                parent2 = self.selection(population, fitness_scores)
                
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.extend([child1, child2])
            
            # Truncar población al tamaño correcto
            population = new_population[:self.population_size]
        
        # Guardar mejor solución
        self.best_solution = best_individual_ever
        
        return {
            'best_solution': best_individual_ever,
            'best_fitness': best_fitness_ever,
            'num_circles': int(best_fitness_ever),
            'circle_positions': best_individual_ever,
            'generations_run': self.generations,
            'final_stats': self.generation_stats[-1] if self.generation_stats else {}
        }
    
    def visualize_results(self, save_path: Optional[str] = None, show_plot: bool = True):
        """
        Visualiza los resultados del algoritmo genético.
        """
        if self.best_solution is None:
            print("No hay solución para visualizar. Ejecute el GA primero.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Optimización de Cortes Circulares - Diámetro: {self.circle_diameter_cm} cm', 
                     fontsize=16, fontweight='bold')
        
        # 1. Imagen original
        axes[0, 0].imshow(self.original_image)
        axes[0, 0].set_title('Imagen Original')
        axes[0, 0].axis('off')
        
        # 2. Imagen binarizada
        axes[0, 1].imshow(self.binary_image, cmap='gray')
        axes[0, 1].set_title('Imagen Binarizada\n(Negro = Piel útil)')
        axes[0, 1].axis('off')
        
        # 3. Resultado con círculos
        axes[1, 0].imshow(self.original_image)
        for i, (x, y) in enumerate(self.best_solution):
            circle = Circle((x, y), self.circle_radius_pixels, 
                          fill=False, color='red', linewidth=2)
            axes[1, 0].add_patch(circle)
            # Agregar número del círculo
            axes[1, 0].text(x, y, str(i+1), ha='center', va='center', 
                           color='white', fontweight='bold', fontsize=8,
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='red', alpha=0.7))
        
        axes[1, 0].set_title(f'Solución Óptima\n{len(self.best_solution)} círculos de {self.circle_diameter_cm} cm')
        axes[1, 0].axis('off')
        
        # 4. Evolución del fitness
        axes[1, 1].plot(self.fitness_history, 'b-', linewidth=2)
        axes[1, 1].set_title('Evolución del Fitness')
        axes[1, 1].set_xlabel('Generación')
        axes[1, 1].set_ylabel('Mejor Fitness')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualización guardada en: {save_path}")
        
        if show_plot:
            plt.show()
        
        return fig
    
    def generate_report(self) -> Dict:
        """
        Genera un reporte completo de los resultados.
        """
        if self.best_solution is None:
            return {"error": "No hay resultados para reportar. Ejecute el GA primero."}
        
        # Calcular estadísticas
        num_circles = len(self.best_solution)
        area_per_circle = np.pi * (self.circle_radius_cm ** 2)
        total_area_used = num_circles * area_per_circle
        
        # Calcular área total de piel disponible
        piel_pixels = np.sum(self.binary_image == 255)
        piel_area_cm2 = piel_pixels / (self.pixels_per_cm ** 2)
        
        efficiency = (total_area_used / piel_area_cm2) * 100 if piel_area_cm2 > 0 else 0
        
        # Crear tabla de posiciones
        positions_table = []
        for i, (x_px, y_px) in enumerate(self.best_solution):
            x_cm = x_px / self.pixels_per_cm
            y_cm = y_px / self.pixels_per_cm
            positions_table.append({
                'Círculo': i + 1,
                'Centro_X_px': x_px,
                'Centro_Y_px': y_px,
                'Centro_X_cm': round(x_cm, 2),
                'Centro_Y_cm': round(y_cm, 2)
            })
        
        report = {
            'parametros': {
                'diametro_circulo_cm': self.circle_diameter_cm,
                'radio_circulo_cm': self.circle_radius_cm,
                'pixels_por_cm': self.pixels_per_cm,
                'generaciones': self.generations,
                'tamano_poblacion': self.population_size
            },
            'resultados': {
                'num_circulos_obtenidos': num_circles,
                'area_por_circulo_cm2': round(area_per_circle, 2),
                'area_total_utilizada_cm2': round(total_area_used, 2),
                'area_piel_disponible_cm2': round(piel_area_cm2, 2),
                'eficiencia_uso_porcentaje': round(efficiency, 2),
                'fitness_final': len(self.best_solution)
            },
            'posiciones_circulos': positions_table,
            'estadisticas_ga': {
                'mejor_fitness_final': self.fitness_history[-1] if self.fitness_history else 0,
                'convergencia': len([f for f in self.fitness_history[-50:] if f == max(self.fitness_history[-50:])]) if len(self.fitness_history) >= 50 else 0
            }
        }
        
        return report
    
    def save_results(self, output_dir: str):
        """
        Guarda todos los resultados en archivos.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Guardar reporte en JSON
        report = self.generate_report()
        report_path = os.path.join(output_dir, f'reporte_circulos_{self.circle_diameter_cm}cm.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Guardar tabla de posiciones como CSV
        if 'posiciones_circulos' in report:
            df = pd.DataFrame(report['posiciones_circulos'])
            csv_path = os.path.join(output_dir, f'posiciones_circulos_{self.circle_diameter_cm}cm.csv')
            df.to_csv(csv_path, index=False, encoding='utf-8')
        
        # Guardar visualización
        viz_path = os.path.join(output_dir, f'resultado_visual_{self.circle_diameter_cm}cm.png')
        self.visualize_results(save_path=viz_path, show_plot=False)
        
        print(f"\nResultados guardados en: {output_dir}")
        print(f"  - Reporte: {report_path}")
        print(f"  - Posiciones CSV: {csv_path}")
        print(f"  - Visualización: {viz_path}")
        
        return output_dir


def main():
    """
    Función principal que ejecuta el programa completo.
    """
    print("=" * 70)
    print("OPTIMIZACIÓN DE CORTES CIRCULARES EN PIEL DE CARNERO")
    print("Algoritmo Genético - Versión 1.0")
    print("=" * 70)
    
    # Configuración
    image_path = r"c:\Users\andre\Core\Eye_ofthe_Universe\Universs\Inteligencia_Artificial\Proyectos\Algoritmos_Geneticos\GA_VImageMan\resources\images\piel_carnero.png"
    output_base_dir = r"c:\Users\andre\Core\Eye_ofthe_Universe\Universs\Inteligencia_Artificial\Proyectos\Algoritmos_Geneticos\GA_VImageMan\results"
    
    # Casos a probar
    diameters = [20.0, 25.0]  # cm
    
    all_results = {}
    
    for diameter in diameters:
        print(f"\n{'='*50}")
        print(f"PROCESANDO CÍRCULOS DE {diameter} CM DE DIÁMETRO")
        print(f"{'='*50}")
        
        try:
            # Crear instancia del GA con parámetros optimizados
            ga = LeatherCuttingGA(
                image_path=image_path,
                circle_diameter_cm=diameter,
                pixels_per_cm=5.0,
                population_size=50,     # Reducido para mejor rendimiento
                generations=100,        # Reducido para mejor rendimiento
                mutation_rate=0.2,      # Aumentado para más exploración
                crossover_rate=0.8,     # Ligeramente reducido
                elite_size=5            # Reducido proporcionalmente
            )
            
            # Ejecutar algoritmo genético
            results = ga.run_ga()
            
            # Generar reporte
            report = ga.generate_report()
            
            # Guardar resultados
            output_dir = os.path.join(output_base_dir, f"diameter_{diameter}cm")
            ga.save_results(output_dir)
            
            # Mostrar resumen
            print(f"\n🎯 RESULTADOS PARA CÍRCULOS DE {diameter} CM:")
            print(f"   ✅ Número de círculos obtenidos: {results['num_circles']}")
            print(f"   📏 Área por círculo: {report['resultados']['area_por_circulo_cm2']} cm²")
            print(f"   📐 Área total utilizada: {report['resultados']['area_total_utilizada_cm2']} cm²")
            print(f"   ⚡ Eficiencia de uso: {report['resultados']['eficiencia_uso_porcentaje']:.1f}%")
            
            all_results[diameter] = {
                'ga_instance': ga,
                'results': results,
                'report': report
            }
            
        except Exception as e:
            print(f"❌ Error procesando diámetro {diameter} cm: {str(e)}")
            continue
    
    # Mostrar comparación final
    print(f"\n{'='*70}")
    print("RESUMEN COMPARATIVO")
    print(f"{'='*70}")
    
    comparison_data = []
    for diameter, data in all_results.items():
        report = data['report']
        comparison_data.append({
            'Diámetro (cm)': diameter,
            'Círculos obtenidos': report['resultados']['num_circulos_obtenidos'],
            'Área utilizada (cm²)': report['resultados']['area_total_utilizada_cm2'],
            'Eficiencia (%)': report['resultados']['eficiencia_uso_porcentaje']
        })
    
    if comparison_data:
        df_comparison = pd.DataFrame(comparison_data)
        print(df_comparison.to_string(index=False))
        
        # Guardar comparación
        comparison_path = os.path.join(output_base_dir, "comparacion_diametros.csv")
        os.makedirs(output_base_dir, exist_ok=True)
        df_comparison.to_csv(comparison_path, index=False, encoding='utf-8')
        print(f"\n📊 Tabla comparativa guardada en: {comparison_path}")
    
    print(f"\n🎉 Proceso completado exitosamente!")
    print(f"📁 Todos los resultados disponibles en: {output_base_dir}")


if __name__ == "__main__":
    main()