"""
Gestor de patrones para detección de consultas.
Carga patrones desde archivo YAML para fácil mantenimiento.
"""

import yaml
import os
from typing import Dict, List, Optional
from pathlib import Path

class PatternManager:
    """Gestiona patrones de detección desde archivo YAML."""
    
    def __init__(self):
        self.patterns = {}
        self.config_path = Path(__file__).parent.parent / "config" / "patterns.yaml"
        self._load_patterns()
    
    def _load_patterns(self):
        """Carga patrones desde archivo YAML."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as file:
                    self.patterns = yaml.safe_load(file)
                print(f"Patrones cargados desde: {self.config_path}")
            else:
                print(f"Archivo de patrones no encontrado: {self.config_path}")
                self._create_default_patterns()
        except Exception as e:
            print(f"Error cargando patrones: {e}")
            self._create_default_patterns()
    
    def _create_default_patterns(self):
        """Crea patrones por defecto como fallback."""
        self.patterns = {
            "system_patterns": [
                'qué modelo', 'que modelo', 'what model', 'which model',
                'modelo actual', 'current model', 'modelo estás usando', 
                'modelo estas usando'
            ],
            "educational_viz_patterns": [
                'qué es visualización', 'que es visualizacion',
                'tipos de gráficos', 'tipos de graficos'
            ],
            "visualization_creation_patterns": [
                'crea', 'crear', 'genera', 'generar'
            ]
        }
    
    def get_patterns(self, pattern_type: str) -> List[str]:
        """Obtiene lista de patrones por tipo."""
        return self.patterns.get(pattern_type, [])
    
    def get_dataset_patterns(self) -> Dict[str, List[str]]:
        """Obtiene patrones específicos para consultas de dataset."""
        dataset_patterns = self.patterns.get("dataset_query_patterns", {})
        return {
            "specific": dataset_patterns.get("specific", []),
            "dataset_words": dataset_patterns.get("general", {}).get("dataset_words", []),
            "action_words": dataset_patterns.get("general", {}).get("action_words", []),
            "request_words": dataset_patterns.get("general", {}).get("request_words", [])
        }
    
    def reload_patterns(self):
        """Recarga patrones desde archivo YAML."""
        self._load_patterns()
        print("Patrones recargados")
    
    def get_version(self) -> str:
        """Obtiene versión de los patrones."""
        return self.patterns.get("version", "unknown")
    
    def get_supported_languages(self) -> List[str]:
        """Obtiene idiomas soportados."""
        return self.patterns.get("supported_languages", ["es", "en"])
    
    def add_pattern_to_yaml(self, pattern_type: str, pattern: str) -> bool:
        """Agrega un patrón al archivo YAML (requiere reinicio)."""
        try:
            # Agregar al diccionario en memoria
            if pattern_type not in self.patterns:
                self.patterns[pattern_type] = []
            
            if pattern not in self.patterns[pattern_type]:
                self.patterns[pattern_type].append(pattern)
                
                # Guardar en archivo YAML
                with open(self.config_path, 'w', encoding='utf-8') as file:
                    yaml.dump(self.patterns, file, default_flow_style=False, 
                             allow_unicode=True, sort_keys=False)
                
                print(f"Patrón agregado: {pattern} -> {pattern_type}")
                return True
            else:
                print(f"Patrón ya existe: {pattern}")
                return False
                
        except Exception as e:
            print(f"Error agregando patrón: {e}")
            return False

# Instancia singleton
pattern_manager = PatternManager()
