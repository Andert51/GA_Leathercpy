#!/usr/bin/env python3
"""
Script de ejecución rápida para el Algoritmo Genético de Cortes Circulares
========================================================================

Este script ejecuta el programa principal con configuraciones preestablecidas.
Se puede modificar fácilmente para probar diferentes parámetros.
"""

import os
import sys
from leather_cutting_ga import LeatherCuttingGA, main

def run_quick_test():
    """
    Ejecuta una prueba rápida con parámetros reducidos para verificación.
    """
    print(" EJECUTANDO PRUEBA RÁPIDA")
    print("=" * 50)
    
    # Configuración para prueba rápida
    image_path = r"../resources/images/piel_carnero.png"
    
    if not os.path.exists(image_path):
        print(f" Error: No se encuentra la imagen en {image_path}")
        print("   Verifique que la imagen esté en la carpeta correcta.")
        return False
    
    try:
        # Crear instancia con parámetros MUY reducidos para prueba rápida
        ga = LeatherCuttingGA(
            image_path=image_path,
            circle_diameter_cm=20.0,
            pixels_per_cm=5.0,
            population_size=20,      # MUY reducido para rendimiento
            generations=30,          # MUY reducido para prueba y rendimiento
            mutation_rate=0.2,       # Aumentado para más exploracion
            crossover_rate=0.8,
            elite_size=3
        )
        
        print(" Imagen cargada correctamente")
        print(f"   Dimensiones: {ga.image_width}x{ga.image_height} píxeles")
        print(f"   Posiciones válidas: {len(ga.valid_positions)}")
        
        if len(ga.valid_positions) == 0:
            print(" No se encontraron posiciones válidas para círculos")
            return False
        
        # Ejecutar algoritmo genético
        print("\n Ejecutando Algoritmo Genético...")
        results = ga.run_ga()
        
        # Mostrar resultados
        print(f"\n RESULTADO DE PRUEBA RÁPIDA:")
        print(f"    Círculos encontrados: {results['num_circles']}")
        print(f"    Fitness final: {results['best_fitness']:.2f}")
        
        # Generar reporte
        report = ga.generate_report()
        print(f"    Área utilizada: {report['resultados']['area_total_utilizada_cm2']:.1f} cm²")
        print(f"    Eficiencia: {report['resultados']['eficiencia_uso_porcentaje']:.1f}%")
        
        # Crear directorio de resultados de prueba
        test_results_dir = "../results/test_run"
        ga.save_results(test_results_dir)
        
        print(f"\n Prueba completada exitosamente!")
        print(f" Resultados guardados en: {test_results_dir}")
        
        return True
        
    except Exception as e:
        print(f" Error durante la prueba: {str(e)}")
        return False

def run_full_optimization():
    """
    Ejecuta la optimización completa con todos los parámetros.
    """
    print(" EJECUTANDO OPTIMIZACIÓN COMPLETA")
    print("=" * 50)
    print("  Esto puede tomar varios minutos...")
    
    try:
        main()
        return True
    except Exception as e:
        print(f" Error durante la optimización completa: {str(e)}")
        return False

def interactive_menu():
    """
    Menú interactivo para seleccionar tipo de ejecución.
    """
    print("=" * 70)
    print("OPTIMIZACIÓN DE CORTES CIRCULARES - ALGORITMO GENÉTICO")
    print("=" * 70)
    print()
    print("Seleccione una opción:")
    print("1.  Prueba rápida (50 individuos, 100 generaciones)")
    print("2.  Optimización completa (150 individuos, 300 generaciones)")
    print("3.  Salir")
    print()
    
    while True:
        try:
            choice = input("Ingrese su opción (1-3): ").strip()
            
            if choice == '1':
                return run_quick_test()
            elif choice == '2':
                return run_full_optimization()
            elif choice == '3':
                print(" ¡Hasta luego!")
                return True
            else:
                print(" Opción inválida. Por favor ingrese 1, 2 o 3.")
                
        except KeyboardInterrupt:
            print("\n\n Programa interrumpido por el usuario.")
            return False
        except Exception as e:
            print(f" Error: {str(e)}")
            return False

if __name__ == "__main__":
    # Cambiar al directorio src si no ese esta ahi
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)
    
    success = interactive_menu()
    
    if success:
        print("\n Ejecución completada.")
    else:
        print("\n Ejecución terminada con errores.")
        sys.exit(1)
