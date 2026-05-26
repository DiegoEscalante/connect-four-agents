# Agente Connect-4 – Cristian Manuel Castañeda Gutiérrez

**Fundamentos de Inteligencia Artificial | Universidad de La Sabana**

---

## Código del agente y Nootebook

| Versión | Archivo | Enlace |
|---------|---------|--------|
| V1 | `policy.py` | [Ver en GitHub](https://github.com/DiegoEscalante/connect-four-agents/blob/Castaneda-agent/groups/Cristian/policy.py) |
| V2 | `policy_qValues.py` | [Ver en GitHub](https://github.com/DiegoEscalante/connect-four-agents/blob/Castaneda-agent/groups/Cristian/policy_qValues.py) |
|  | `entrega.py` | [`entrega.ipynb`](entrega.ipynb). |

---

---

## Guía de uso

```bash
# Desde la raíz del repositorio
cd connect-four-agents

# Validar V1 y V2 vs aleatorio y entre sí
python validate.py

# Validar V2 vs Grupo A
python validate_vs_groupA.py

# Abrir el notebook de análisis
jupyter notebook groups/Cristian/entrega.ipynb
```

**Requisitos:** `numpy`, `matplotlib`, `jupyter`. El archivo `q_table.pkl` se genera automáticamente al jugar partidas con V2.
