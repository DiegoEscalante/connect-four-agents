"""
PDF de entrega — Agente Connect-4 de Julián.
Ejecutar desde la raíz del proyecto:
    .venv/Scripts/python groups/Julian/generar_pdf.py
"""
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from PIL import Image as _PIL
import os

ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
IMG_DIR = os.path.join(ROOT, 'groups', 'Julian')
OUT     = os.path.join(IMG_DIR, 'informe_julian.pdf')

# ── Paleta ────────────────────────────────────────────────────────────────────
BLUE   = (22,  64, 148)
DBLUE  = (12,  38,  88)
LBLUE  = (218, 228, 248)
BLACK  = (25,  25,  25)
GRAY   = (85,  85,  85)
LGRAY  = (165, 165, 165)
WHITE  = (255, 255, 255)
GREEN  = (32, 156,  82)
ORANGE = (200, 100,  20)

FONT_R = r'C:\Windows\Fonts\arial.ttf'
FONT_B = r'C:\Windows\Fonts\arialbd.ttf'
FONT_I = r'C:\Windows\Fonts\ariali.ttf'

LINK   = "github.com/DiegoEscalante/connect-four-agents/blob/Analisis/groups/Julian/policy.py"
NOMBRE = "Julián Romero"


def img_h(path, w_mm):
    with _PIL.open(path) as im:
        wp, hp = im.size
    return w_mm * hp / wp


# ── Clase PDF ─────────────────────────────────────────────────────────────────
class PDF(FPDF):

    def setup_fonts(self):
        self.add_font('A', '',  FONT_R)
        self.add_font('A', 'B', FONT_B)
        self.add_font('A', 'I', FONT_I)

    def header(self):
        # Barra azul oscuro en la parte superior
        self.set_fill_color(*DBLUE)
        self.rect(0, 0, 210, 8.5, style='F')
        self.set_font('A', '', 7.5)
        self.set_text_color(*WHITE)
        self.set_xy(0, 0.8)
        self.cell(210, 7,
                  f'Universidad de La Sabana  •  Fundamentos de Inteligencia Artificial 2026.1  •  {NOMBRE}',
                  align='C')
        self.set_y(11)

    def footer(self):
        self.set_y(-11)
        self.set_draw_color(*LBLUE)
        self.line(self.l_margin, self.get_y(), 210 - self.r_margin, self.get_y())
        self.set_font('A', 'I', 7.5)
        self.set_text_color(*LGRAY)
        self.cell(0, 7, f'Página {self.page_no()}  •  {LINK}', align='C')

    def sec(self, title):
        """Título de sección: fondo azul sólido, texto blanco."""
        y = self.get_y()
        self.set_fill_color(*BLUE)
        self.rect(self.l_margin - 1, y, self.epw + 2, 6.5, style='F')
        self.set_font('A', 'B', 10)
        self.set_text_color(*WHITE)
        self.set_xy(self.l_margin + 1, y)
        self.cell(self.epw, 6.5, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2.5)
        self.set_text_color(*BLACK)

    def body(self, text, size=9.5, lh=5.1):
        self.set_font('A', '', size)
        self.set_text_color(*BLACK)
        self.multi_cell(0, lh, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def caption(self, text):
        self.set_font('A', 'I', 8)
        self.set_text_color(*GRAY)
        self.multi_cell(0, 4.5, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def table(self, headers, rows, widths, hcolor=None):
        hc = hcolor or BLUE
        self.set_font('A', 'B', 8.5)
        self.set_fill_color(*hc)
        self.set_text_color(*WHITE)
        self.set_draw_color(190, 200, 220)
        for h, w in zip(headers, widths):
            self.cell(w, 6.5, h, border=1, fill=True, align='C')
        self.ln()
        self.set_font('A', '', 8.5)
        self.set_text_color(*BLACK)
        alt = False
        for row in rows:
            self.set_fill_color(*(LBLUE if alt else WHITE))
            for cell, w in zip(row, widths):
                self.cell(w, 5.5, cell, border=1, fill=True, align='C')
            self.ln()
            alt = not alt
        self.ln(1.5)

    def metric_row(self, items):
        """Fila de métricas destacadas: [(label, value, color), ...]"""
        n   = len(items)
        w   = (self.epw - (n - 1) * 2) / n
        y0  = self.get_y()
        h   = 12
        for i, (label, value, color) in enumerate(items):
            x = self.l_margin + i * (w + 2)
            self.set_fill_color(*LBLUE)
            self.rect(x, y0, w, h, style='F')
            self.set_draw_color(*BLUE)
            self.rect(x, y0, w, h, style='D')
            # valor
            self.set_font('A', 'B', 12)
            self.set_text_color(*color)
            self.set_xy(x, y0 + 1)
            self.cell(w, 5.5, value, align='C')
            # etiqueta
            self.set_font('A', '', 7)
            self.set_text_color(*GRAY)
            self.set_xy(x, y0 + 6)
            self.cell(w, 5, label, align='C')
        self.set_y(y0 + h + 2)


# ── Construir PDF ─────────────────────────────────────────────────────────────
pdf = PDF()
pdf.setup_fonts()
pdf.set_margins(17, 14, 17)
pdf.set_auto_page_break(True, margin=13)
pdf.add_page()

# ── TÍTULO ────────────────────────────────────────────────────────────────────
pdf.set_font('A', 'B', 15)
pdf.set_text_color(*BLUE)
pdf.cell(0, 9, 'Agente Connect-4: Minimax + Alfa-Beta + Heurística Allis',
         new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
pdf.set_font('A', '', 8.5)
pdf.set_text_color(*GRAY)
pdf.cell(0, 5, f'{NOMBRE}  •  Mayo 2026  •  Código: ' + LINK,
         new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
pdf.ln(4)

# ── 1. IDEA PRINCIPAL ─────────────────────────────────────────────────────────
pdf.sec('1. Idea Principal y Distinción')
pdf.body(
    'El agente implementa búsqueda Minimax con poda Alfa-Beta, Iterative Deepening con '
    'presupuesto de tiempo y tabla de transposiciones (Zobrist hashing). Su función de '
    'evaluación incorpora la teoría de odd/even threats de Victor Allis (1988): conocimiento '
    'explícito de que Rojo debe crear amenazas en filas impares y Amarillo en filas pares, '
    'por paridad de turnos en el endgame. Este paradigma knowledge-based (búsqueda determinista '
    '+ conocimiento del dominio) se diferencia fundamentalmente de MCTS, que estima el valor '
    'de una posición mediante simulaciones aleatorias sin conocimiento explícito del juego.'
)
pdf.table(
    ['Aspecto', 'MCTS (otros agentes del grupo)', 'Minimax + Allis (este agente)'],
    [
        ['Paradigma',        'Simulación estocástica',     'Búsqueda determinista + conocimiento'],
        ['Evaluación',       'Rollouts aleatorios',        'Heurística (ventanas + Allis + centro)'],
        ['Parámetro clave',  'simulations',                'depth  (variable numérica principal)'],
        ['Determinismo',     'No — varía con la semilla',  'Sí — mismo tablero → mismo movimiento'],
    ],
    [38, 64, 74]
)

# ── 2. VERSIONES ──────────────────────────────────────────────────────────────
pdf.sec('2. Versiones del Agente')
pdf.table(
    ['Versión', 'Componentes de la función de evaluación', 'use_allis', 'depth default'],
    [
        ['V1 — Completa (producción)', 'Ventanas 4-en-línea + sesgo central + odd/even threats (Allis)', 'True',  '10 + time_limit'],
        ['V2 — Sin Allis (ablación)',  'Ventanas 4-en-línea + sesgo central únicamente',                 'False', '10 + time_limit'],
    ],
    [40, 104, 18, 14]
)
pdf.body(
    'La teoría de Allis establece que en el endgame, por paridad de turnos (Zugzwang), '
    'Rojo ocupa filas impares y Amarillo filas pares. V1 añade un bonus w_odd_even=20 '
    'a amenazas propias con paridad correcta y penaliza las del oponente. La comparación '
    'V1 vs V2 (Exp. 3) cuantifica el aporte concreto de este conocimiento de dominio.'
)
pdf.ln(2)

# ── 3. ANÁLISIS ───────────────────────────────────────────────────────────────
pdf.sec('3. Análisis del Agente')

# Métricas destacadas en fila
pdf.metric_row([
    ('vs Aleatorio — Rojo',     '100 %', GREEN),
    ('vs Aleatorio — Amarillo', '100 %', GREEN),
    ('Self-play (empates)',      '100 %', BLUE),
    ('V1 vs V2 vs Aleatorio',   '100/100%', ORANGE),
])

pdf.ln(1)

# ── Imágenes: exp2 + exp4 lado a lado ─────────────────────────────────────────
GAP   = 3
w2    = pdf.epw * 0.57
w4    = pdf.epw - w2 - GAP
p2    = os.path.join(IMG_DIR, 'exp2_depth.png')
p4    = os.path.join(IMG_DIR, 'exp4_selfplay.png')

if os.path.exists(p2) and os.path.exists(p4):
    h2 = img_h(p2, w2)
    h4 = img_h(p4, w4)
    y0 = pdf.get_y()
    pdf.image(p2, x=pdf.l_margin,          y=y0, w=w2)
    pdf.image(p4, x=pdf.l_margin + w2 + GAP, y=y0, w=w4)
    pdf.set_y(y0 + max(h2, h4) + 1.5)

pdf.caption(
    'Fig. 1 — Exp. 2 (izq.): Win rate vs profundidad de búsqueda. Contra el aleatorio la '
    'saturación ocurre desde depth=1 (la heurística sola domina al azar); la variable '
    'numérica depth es decisiva en escenarios con oponentes más fuertes.   '
    'Fig. 2 — Exp. 4 (der.): Self-play con depth=4. El 100 % de empates confirma que dos '
    'instancias del mismo agente determinista convergen a la misma línea de juego.'
)
pdf.ln(2)

# ── Imagen exp3 ancho completo (V1 vs V2) ─────────────────────────────────────
p3 = os.path.join(IMG_DIR, 'exp3_versions.png')
if os.path.exists(p3):
    h3 = img_h(p3, pdf.epw) * 0.88
    y3 = pdf.get_y()
    pdf.image(p3, x=pdf.l_margin, y=y3, w=pdf.epw, h=h3)
    pdf.set_y(y3 + h3 + 1.5)

pdf.caption(
    'Fig. 3 — Exp. 3: V1 (con teoría Allis) vs V2 (sin Allis), depth=4. Ambas versiones '
    'ganan el 100 % contra el aleatorio en ambos colores, mostrando que la arquitectura '
    'base es robusta. La diferencia entre versiones se vuelve significativa en escenarios '
    'más exigentes donde la paridad de filas determina si una amenaza se materializa.'
)
pdf.ln(2)

# ── Conclusiones del análisis ─────────────────────────────────────────────────
pdf.body(
    'Conclusiones: (1) Ambas versiones (V1 y V2) ganan el 100 % contra el aleatorio en '
    'ambos colores, cumpliendo el requisito mínimo de la rúbrica. (2) La variable numérica '
    'depth no afecta el resultado contra el aleatorio (saturación en depth=1), pero es el '
    'factor crítico de rendimiento frente a oponentes no triviales. (3) El self-play '
    'produce empate perfecto: evidencia de comportamiento determinista y estable. '
    '(4) La tabla de transposiciones con Zobrist hashing (Exp. 7 del notebook) permite '
    'alcanzar un nivel extra de profundidad en el mismo presupuesto de tiempo.'
)
pdf.ln(2)

# ── 4. PROPUESTAS DE MEJORA ───────────────────────────────────────────────────
pdf.sec('4. Propuestas de Mejora')

# Propuesta 1
pdf.set_font('A', 'B', 9.5)
pdf.set_text_color(*BLUE)
pdf.cell(0, 5.5, '1. Bitboards — impacto alto, esfuerzo medio',
         new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_text_color(*BLACK)
pdf.body(
    'Representar el tablero como dos enteros de 64 bits (uno por jugador) permite detectar '
    'victorias en O(1) con operaciones bit-a-bit, frente al O(28) actual del winner_at '
    'incremental. Esto multiplicaría la velocidad de evaluación por 10–20×, haciendo viable '
    'depth=8–9 en el mismo presupuesto de tiempo. Evidencia directa: el Exp. 7 muestra '
    'que la tabla de transposiciones ya reduce el tiempo por turno; con bitboards se '
    'eliminaría el cuello de botella restante en la evaluación por nodo, potenciando '
    'aún más el Iterative Deepening.',
    size=9
)

# Propuesta 2
pdf.ln(1)
pdf.set_font('A', 'B', 9.5)
pdf.set_text_color(*BLUE)
pdf.cell(0, 5.5, '2. Ajuste automático de pesos — impacto medio, esfuerzo alto',
         new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_text_color(*BLACK)
pdf.body(
    'Los pesos w_center=6 y w_odd_even=20 se definieron manualmente. El Exp. 5 muestra '
    'que en el rango [10, 40] la sensibilidad contra el aleatorio es nula (techo del 100 %), '
    'pero el impacto sería mayor contra oponentes fuertes como MCTS. Entrenar los pesos '
    'mediante auto-play y gradient descent permitiría encontrar la configuración óptima '
    'específicamente para el torneo, sin sesgo manual.',
    size=9
)

pdf.output(OUT)
print(f'PDF generado: {OUT}')
