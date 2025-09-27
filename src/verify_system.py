#!/usr/bin/env python3
"""
Script de verificación del sistema
==================================

Este script verifica que todas las dependencias estén instaladas correctamente
y que el sistema esté listo para ejecutar el algoritmo genético.
"""

import sys
import os
import importlib
from pathlib import Path

def check_python_version():
    """Verifica la versión de Python."""
    print("Verificando versión de Python...")
    version = sys.version_info
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f" Python {version.major}.{version.minor} detectado")
        print("   Se requiere Python 3.8 o superior")
        return False
    
    print(f" Python {version.major}.{version.minor}.{version.micro} - OK")
    return True

def check_dependencies():
    """Verifica que todas las dependencias estén instaladas."""
    print("\n Verificando dependencias...")
    
    required_packages = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'matplotlib': 'matplotlib',
        'PIL': 'Pillow',
        'scipy': 'scipy',
        'tqdm': 'tqdm',
        'skimage': 'scikit-image'
    }
    
    missing_packages = []
    
    for module_name, package_name in required_packages.items():
        try:
            importlib.import_module(module_name)
            print(f" {package_name} - OK")
        except ImportError:
            print(f" {package_name} - FALTANTE")
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\n Dependencias faltantes: {', '.join(missing_packages)}")
        print("   Ejecute: pip install -r requirements.txt")
        return False
    
    return True

def check_file_structure():
    """Verifica la estructura de archivos del proyecto."""
    print("\n Verificando estructura de archivos...")
    
    required_files = [
        'requirements.txt',
        'config.json',
        'src/leather_cutting_ga.py',
        'src/run_ga.py',
        'resources/images/piel_carnero.png'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f" {file_path} - OK")
        else:
            print(f" {file_path} - FALTANTE")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n Archivos faltantes: {', '.join(missing_files)}")
        return False
    
    return True

def check_image_file():
    """Verifica que la imagen de entrada sea válida."""
    print("\n  Verificando imagen de entrada...")
    
    image_path = 'resources/images/piel_carnero.png'
    
    if not os.path.exists(image_path):
        print(f" Imagen no encontrada: {image_path}")
        return False
    
    try:
        import cv2
        image = cv2.imread(image_path)
        
        if image is None:
            print(f" No se pudo cargar la imagen: {image_path}")
            return False
        
        height, width = image.shape[:2]
        print(f" Imagen cargada: {width}x{height} píxeles")
        
        # Verificar que no esté completamente blanca o negra
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        unique_values = len(set(gray.flatten()))
        
        if unique_values < 10:
            print("  Advertencia: La imagen parece tener muy pocos valores únicos")
            print("   Esto podría afectar la umbralización")
        
        return True
        
    except Exception as e:
        print(f" Error al verificar imagen: {str(e)}")
        return False

def test_basic_functionality():
    """Prueba funcionalidad básica del algoritmo."""
    print("\n Probando funcionalidad básica...")
    
    try:
        sys.path.append('src')
        from leather_cutting_ga import LeatherCuttingGA
        
        # Crear instancia mínima
        ga = LeatherCuttingGA(
            image_path='resources/images/piel_carnero.png',
            circle_diameter_cm=20.0,
            pixels_per_cm=5.0,
            population_size=10,
            generations=5
        )
        
        print(f" Imagen procesada: {ga.image_width}x{ga.image_height}")
        print(f" Posiciones válidas: {len(ga.valid_positions)}")
        
        # Crear un individuo de prueba
        individual = ga.create_individual()
        print(f" Individuo creado: {len(individual)} círculos")
        
        # Calcular fitness
        fitness = ga.fitness(individual)
        print(f" Fitness calculado: {fitness:.2f}")
        
        return True
        
    except Exception as e:
        print(f" Error en prueba funcional: {str(e)}")
        return False

def create_test_directories():
    """Crea directorios necesarios para las pruebas."""
    print("\n Creando directorios de prueba...")
    
    directories = ['results', 'results/test']
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f" Directorio creado/verificado: {directory}")
    
    return True

def main():
    """Función principal de verificación."""
    print("=" * 70)
    print("VERIFICACIÓN DEL SISTEMA - ALGORITMO GENÉTICO")
    print("=" * 70)
    
    checks = [
        ("Versión de Python", check_python_version),
        ("Dependencias", check_dependencies),
        ("Estructura de archivos", check_file_structure),
        ("Imagen de entrada", check_image_file),
        ("Directorios", create_test_directories),
        ("Funcionalidad básica", test_basic_functionality)
    ]
    
    all_passed = True
    
    for check_name, check_function in checks:
        try:
            if not check_function():
                all_passed = False
        except Exception as e:
            print(f" Error en {check_name}: {str(e)}")
            all_passed = False
    
    print("\n" + "=" * 70)
    
    if all_passed:
        print(" TODAS LAS VERIFICACIONES PASARON EXITOSAMENTE")
        print(" El sistema está listo para ejecutar el algoritmo genético")
        print("\nPara ejecutar:")
        print("  - Windows: Doble clic en 'run_genetic_algorithm.bat'")
        print("  - Python directo: 'python src/run_ga.py'")
    else:
        print(" ALGUNAS VERIFICACIONES FALLARON")
        print("   Por favor corrija los errores antes de continuar")
        return False
    
    print("=" * 70)
    return True

if __name__ == "__main__":
    success = main()
    
    if not success:
        input("\nPresione Enter para salir...")
        sys.exit(1)
