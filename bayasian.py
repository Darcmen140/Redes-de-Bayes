# path: expert_system/expert_system.py

import tkinter as tk
from tkinter import messagebox
import sqlite3
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Clase que representa la Base de Conocimiento (red bayesiana)
class KnowledgeBase:
    def __init__(self):
        self.model = BayesianNetwork([('Inteligencia', 'Nota'),
                                      ('Dificultad', 'Nota'),
                                      ('Asistencia', 'Nota')])
        self._define_cpds()

    def _define_cpds(self):
        cpd_inteligencia = TabularCPD(variable='Inteligencia', variable_card=2, values=[[0.7], [0.3]])
        cpd_dificultad = TabularCPD(variable='Dificultad', variable_card=2, values=[[0.6], [0.4]])
        cpd_asistencia = TabularCPD(variable='Asistencia', variable_card=2, values=[[0.8], [0.2]])

        cpd_nota = TabularCPD(variable='Nota', variable_card=2,
                              values=[[0.9, 0.7, 0.8, 0.1, 0.8, 0.6, 0.7, 0.3],
                                      [0.1, 0.3, 0.2, 0.9, 0.2, 0.4, 0.3, 0.7]],
                              evidence=['Inteligencia', 'Dificultad', 'Asistencia'],
                              evidence_card=[2, 2, 2])

        self.model.add_cpds(cpd_inteligencia, cpd_dificultad, cpd_asistencia, cpd_nota)
        assert self.model.check_model()

    def get_model(self):
        return self.model

# Clase que representa el Motor de Inferencia
class InferenceEngine:
    def __init__(self, model):
        self.infer_engine = VariableElimination(model)

    def infer(self, evidence):
        result = self.infer_engine.query(variables=['Nota'], evidence=evidence)
        return result

# Clase que representa la Base de Hechos
class FactBase:
    def __init__(self):
        self.facts = {}

    def add_fact(self, key, value):
        self.facts[key] = value

    def get_facts(self):
        return self.facts

# Clase que representa el Subsistema de Justificación
class JustificationSubsystem:
    def __init__(self):
        self.justifications = []

    def add_justification(self, justification):
        self.justifications.append(justification)

    def get_justifications(self):
        return self.justifications

# Clase que representa el Sistema de Adquisición de Conocimiento
class KnowledgeAcquisitionSystem:
    def __init__(self, knowledge_base):
        self.knowledge_base = knowledge_base

    def update_knowledge(self, new_cpds):
        for cpd in new_cpds:
            self.knowledge_base.model.add_cpds(cpd)
        assert self.knowledge_base.model.check_model()

# Clase que representa la Base de Datos
class Database:
    def __init__(self, db_name='expert_system.db'):
        self.conn = sqlite3.connect(db_name)
        self.create_tables()

    def create_tables(self):
        with self.conn:
            self.conn.execute('''CREATE TABLE IF NOT EXISTS facts (
                                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                                    key TEXT NOT NULL,
                                    value INTEGER NOT NULL);''')
            self.conn.execute('''CREATE TABLE IF NOT EXISTS results (
                                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                                    result REAL NOT NULL);''')

    def insert_fact(self, key, value):
        with self.conn:
            self.conn.execute("INSERT INTO facts (key, value) VALUES (?, ?);", (key, value))

    def insert_result(self, result):
        with self.conn:
            self.conn.execute("INSERT INTO results (result) VALUES (?);", (result,))

    def get_facts(self):
        with self.conn:
            cursor = self.conn.execute("SELECT key, value FROM facts;")
            return cursor.fetchall()

    def get_results(self):
        with self.conn:
            cursor = self.conn.execute("SELECT result FROM results;")
            return cursor.fetchall()

# Clase que representa la Interfaz de Usuario
class UserInterface:
    def __init__(self, expert_system):
        self.expert_system = expert_system
        self.root = tk.Tk()
        self.root.title("Sistema Experto")

        self.inteligencia_label = tk.Label(self.root, text="¿Es el estudiante inteligente? (1: Sí, 0: No)")
        self.inteligencia_label.pack()
        self.inteligencia_entry = tk.Entry(self.root)
        self.inteligencia_entry.pack()

        self.asistencia_label = tk.Label(self.root, text="¿Asistió el estudiante a todas las clases? (1: Sí, 0: No)")
        self.asistencia_label.pack()
        self.asistencia_entry = tk.Entry(self.root)
        self.asistencia_entry.pack()

        self.submit_button = tk.Button(self.root, text="Ejecutar Inferencia", command=self.run_inference)
        self.submit_button.pack()

        self.result_label = tk.Label(self.root, text="")
        self.result_label.pack()

    def run_inference(self):
        try:
            inteligencia = int(self.inteligencia_entry.get())
            asistencia = int(self.asistencia_entry.get())
            if inteligencia not in [0, 1] or asistencia not in [0, 1]:
                raise ValueError

            self.expert_system.fact_base.add_fact('Inteligencia', inteligencia)
            self.expert_system.fact_base.add_fact('Asistencia', asistencia)

            result = self.expert_system.inference_engine.infer(self.expert_system.fact_base.get_facts())
            result_value = result.values[1]
            self.expert_system.justification_subsystem.add_justification(f"Resultado basado en hechos: {self.expert_system.fact_base.get_facts()}")
            self.expert_system.database.insert_fact('Inteligencia', inteligencia)
            self.expert_system.database.insert_fact('Asistencia', asistencia)
            self.expert_system.database.insert_result(result_value)
            
            self.result_label.config(text=f"Probabilidad de obtener una buena nota: {result_value:.2f}")
        except ValueError:
            messagebox.showerror("Entrada no válida", "Por favor, ingrese 1 para Sí o 0 para No.")

    def start(self):
        self.root.mainloop()

# Clase principal del Sistema Experto
class ExpertSystem:
    def __init__(self):
        self.knowledge_base = KnowledgeBase()
        self.inference_engine = InferenceEngine(self.knowledge_base.get_model())
        self.fact_base = FactBase()
        self.justification_subsystem = JustificationSubsystem()
        self.knowledge_acquisition_system = KnowledgeAcquisitionSystem(self.knowledge_base)
        self.database = Database()
        self.user_interface = UserInterface(self)

    def run(self):
        self.user_interface.start()

# Ejecutar el sistema experto
if __name__ == "__main__":
    expert_system = ExpertSystem()
    expert_system.run()
