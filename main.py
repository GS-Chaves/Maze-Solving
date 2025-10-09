import argparse
import math
import os
import sys
from collections import deque

try:
    import cv2
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "Este programa requer as bibliotecas 'opencv-python' (cv2) e 'numpy'. "
        "Instale-as com 'pip install opencv-python numpy'."
    ) from exc

from maze import MazeMap, MazeSize


class Environment:
    """Simula o ambiente do labirinto a partir de um arquivo de texto."""

    DIRECTIONS = {
        'N': (0, -1),
        'S': (0, 1),
        'L': (1, 0),
        'O': (-1, 0),
    }

    PERIPHERAL_OFFSETS = {
        'N': {
            'front': (0, -1),
            'front_left': (-1, -1),
            'front_right': (1, -1),
            'left': (-1, 0),
            'right': (1, 0),
            'back': (0, 1),
            'back_left': (-1, 1),
            'back_right': (1, 1),
        },
        'S': {
            'front': (0, 1),
            'front_left': (1, 1),
            'front_right': (-1, 1),
            'left': (1, 0),
            'right': (-1, 0),
            'back': (0, -1),
            'back_left': (1, -1),
            'back_right': (-1, -1),
        },
        'L': {
            'front': (1, 0),
            'front_left': (1, -1),
            'front_right': (1, 1),
            'left': (0, -1),
            'right': (0, 1),
            'back': (-1, 0),
            'back_left': (-1, -1),
            'back_right': (-1, 1),
        },
        'O': {
            'front': (-1, 0),
            'front_left': (-1, 1),
            'front_right': (-1, -1),
            'left': (0, 1),
            'right': (0, -1),
            'back': (1, 0),
            'back_left': (1, 1),
            'back_right': (1, -1),
        },
    }
    FOOD_SECTOR_ORDER = ['L', 'NE', 'N', 'NO', 'O', 'SO', 'S', 'SE']

    def __init__(self, filepath):
        self.filepath = filepath
        self.grid = self._load_map(filepath)
        self.height = len(self.grid)
        self.width = len(self.grid[0]) if self.grid else 0
        self.direction = 'N'
        self.position = self._find_symbol('E')
        if self.position is None:
            raise ValueError("O labirinto precisa ter uma entrada 'E'")
        self.exit_position = self._find_symbol('S')
        self.food_positions = self._collect_positions('o')
        self.total_food = len(self.food_positions)
        self.collected_food = 0

    def _load_map(self, filepath):
        with open(filepath, "r", encoding="ascii") as maze_file:
            raw_lines = [line.rstrip('\n') for line in maze_file if line.strip('\n')]

        if not raw_lines:
            raise ValueError("O arquivo do labirinto está vazio")

        width = len(raw_lines[0])
        grid = []
        for line in raw_lines:
            if len(line) != width:
                raise ValueError("Todas as linhas do labirinto devem ter o mesmo tamanho")
            grid.append(list(line))
        return grid

    def _find_symbol(self, symbol):
        for y, row in enumerate(self.grid):
            for x, cell in enumerate(row):
                if cell == symbol:
                    return (x, y)
        return None

    def _collect_positions(self, symbol):
        positions = set()
        for y, row in enumerate(self.grid):
            for x, cell in enumerate(row):
                if cell == symbol:
                    positions.add((x, y))
        return positions

    def in_bounds(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    def get_tile(self, x, y):
        if not self.in_bounds(x, y):
            return 'X'
        return self.grid[y][x]

    def get_position(self):
        return self.position

    def get_direction(self):
        return self.direction

    def current_tile(self):
        x, y = self.position
        return self.grid[y][x]

    def food_remaining(self):
        return len(self.food_positions)

    def food_total(self):
        return self.total_food

    def food_collected(self):
        return self.collected_food

    def setDirection(self, direction):
        if direction not in self.DIRECTIONS:
            raise ValueError("Direção inválida: use 'N', 'S', 'L' ou 'O'")
        self.direction = direction

    def move(self):
        dx, dy = self.DIRECTIONS[self.direction]
        next_x = self.position[0] + dx
        next_y = self.position[1] + dy

        if not self.in_bounds(next_x, next_y):
            return False, 'X', 0

        target_cell = self.grid[next_y][next_x]
        if target_cell == 'X':
            return False, target_cell, 0
        if target_cell == 'S' and self.food_remaining() > 0:
            return False, target_cell, 0

        reward = -1
        if target_cell == 'o':
            reward += 10
            self.food_positions.discard((next_x, next_y))
            self.grid[next_y][next_x] = '_'
            target_cell = '_'
            self.collected_food += 1

        self.position = (next_x, next_y)
        return True, target_cell, reward

    def getSensor(self):
        sensor = []
        cx, cy = self.position
        for dy in (-1, 0, 1):
            row = []
            for dx in (-1, 0, 1):
                sample_x = cx + dx
                sample_y = cy + dy
                if dx == 0 and dy == 0:
                    row.append(self.direction)
                elif self.in_bounds(sample_x, sample_y):
                    row.append(self.grid[sample_y][sample_x])
                else:
                    row.append('X')
            sensor.append(row)
        return sensor

    def get_peripheral_view(self):
        offsets = self.PERIPHERAL_OFFSETS[self.direction]
        cx, cy = self.position
        
        view = {}
        for label, (dx, dy) in offsets.items():
            x = cx + dx
            y = cy + dy
            view[label] = {
                'coord': (x, y),
                'tile': self.get_tile(x, y),
            }
        return view

    def get_food_direction_sensor(self):
        """Retorna contagens de comida remanescente em cada direção cardeal/diagonal."""
        counts = {label: 0 for label in self.FOOD_SECTOR_ORDER}
        nearest = {label: math.inf for label in self.FOOD_SECTOR_ORDER}
        cx, cy = self.position

        for food_x, food_y in self.food_positions:
            dx = food_x - cx
            dy = food_y - cy
            if dx == 0 and dy == 0:
                continue
            angle = math.degrees(math.atan2(-dy, dx))
            if angle < 0:
                angle += 360
            sector_index = int((angle + 22.5) // 45) % len(self.FOOD_SECTOR_ORDER)
            sector = self.FOOD_SECTOR_ORDER[sector_index]
            counts[sector] += 1
            distance = abs(dx) + abs(dy)
            if distance < nearest[sector]:
                nearest[sector] = distance

        center_value = 1 if self.current_tile() == 'o' else 0
        matrix = [
            [counts['NO'], counts['N'], counts['NE']],
            [counts['O'], center_value, counts['L']],
            [counts['SO'], counts['S'], counts['SE']],
        ]
        return {
            'counts': counts,
            'matrix': matrix,
            'total_food': sum(counts.values()) + center_value,
            'nearest_distance': {
                label: (None if math.isinf(nearest[label]) else nearest[label])
                for label in self.FOOD_SECTOR_ORDER
            },
        }


class Visualizer:
    """Gera frames do labirinto e salva um vídeo com o percurso do agente."""

    def __init__(
        self,
        env,
        *,
        cell_size=32,
        fps=8,
        output_path="maze_run.mp4",
        codec="mp4v",
        dump_debug_frames=False,
    ):
        self.cell_size = cell_size
        self.fps = fps
        self.output_path = output_path
        self.codec = codec
        self.width_px = env.width * cell_size
        self.height_px = env.height * cell_size
        self.frames = []
        self.last_saved_path = None
        self.color_map = {
            'X': (42, 44, 56),
            '_': (245, 246, 248),
            ' ': (245, 246, 248),
            'E': (80, 190, 120),
            'S': (200, 170, 90),
            'o': (80, 200, 255),
        }
        self.background_color = (18, 20, 26)
        self.tile_border_color = (68, 72, 80)
        self.exit_emphasis_color = (180, 200, 255)
        self.entry_emphasis_color = (140, 220, 170)
        self.food_color = (60, 210, 255)
        self.agent_body_color = (255, 170, 90)
        self.agent_core_color = (35, 100, 255)
        self.agent_outline_color = (12, 20, 35)
        self.dump_debug_frames = dump_debug_frames
        self.debug_dir = "frames_debug"

    def add_frame(self, env):
        frame = self._render_frame(env)
        self._validate_frame(frame)
        self.frames.append(frame.copy())
        if self.dump_debug_frames:
            os.makedirs(self.debug_dir, exist_ok=True)
            index = len(self.frames) - 1
            cv2.imwrite(os.path.join(self.debug_dir, f"frame_{index:04d}.png"), frame)

    def save(self, output_path=None):
        if not self.frames:
            raise RuntimeError("Nenhum frame capturado para gerar o vídeo.")

        target_path = output_path or self.output_path
        frame_size = (self.width_px, self.height_px)

        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        writer = cv2.VideoWriter(target_path, fourcc, self.fps, frame_size)

        if not writer.isOpened():
            writer.release()
            raise RuntimeError(
                f"Não foi possível inicializar o gravador de vídeo com OpenCV (codec {self.codec})."
            )

        for frame in self.frames:
            writer.write(self._ensure_frame_format(frame))
        writer.release()
        file_size = os.path.getsize(target_path) if os.path.exists(target_path) else 0
        if file_size == 0:
            raise RuntimeError("Vídeo gerado com tamanho zero; verifique codecs instalados.")
        self.last_saved_path = target_path
        return target_path

    def _render_frame(self, env):
        frame = np.full((self.height_px, self.width_px, 3), self.background_color, dtype=np.uint8)
        for y in range(env.height):
            for x in range(env.width):
                char = env.grid[y][x]
                color = self.color_map.get(char, (200, 200, 200))
                x0 = x * self.cell_size
                y0 = y * self.cell_size
                top_left = (x0, y0)
                bottom_right = (x0 + self.cell_size - 1, y0 + self.cell_size - 1)
                cv2.rectangle(
                    frame,
                    top_left,
                    bottom_right,
                    color,
                    -1,
                    lineType=cv2.LINE_AA,
                )
                if char == 'S':
                    cv2.rectangle(
                        frame,
                        top_left,
                        bottom_right,
                        self.exit_emphasis_color,
                        2,
                        lineType=cv2.LINE_AA,
                    )
                elif char == 'E':
                    cv2.rectangle(
                        frame,
                        top_left,
                        bottom_right,
                        self.entry_emphasis_color,
                        2,
                        lineType=cv2.LINE_AA,
                    )
                if char == 'o':
                    self._draw_food(frame, top_left, bottom_right)
                cv2.rectangle(
                    frame,
                    top_left,
                    bottom_right,
                    self.tile_border_color,
                    1,
                    lineType=cv2.LINE_AA,
                )

        agent_pos = env.get_position()
        if agent_pos is not None:
            self._draw_agent(frame, agent_pos, env.get_direction())

        return frame

    def _draw_agent(self, frame, position, direction):
        x, y = position
        x0 = x * self.cell_size
        y0 = y * self.cell_size
        center = (x0 + self.cell_size // 2, y0 + self.cell_size // 2)
        outer_radius = max(4, self.cell_size // 2 - 2)
        radius = max(4, self.cell_size // 3)
        cv2.circle(frame, center, outer_radius, self.agent_outline_color, -1, lineType=cv2.LINE_AA)
        cv2.circle(frame, center, radius, self.agent_body_color, -1, lineType=cv2.LINE_AA)
        inner_radius = max(2, radius // 2)
        cv2.circle(frame, center, inner_radius, self.agent_core_color, -1, lineType=cv2.LINE_AA)

        tip_offset = max(3, self.cell_size // 2)
        half = max(2, self.cell_size // 4)
        if direction == 'N':
            points = np.array([
                (center[0], center[1] - tip_offset),
                (center[0] - half, center[1] + half),
                (center[0] + half, center[1] + half),
            ], dtype=np.int32)
        elif direction == 'S':
            points = np.array([
                (center[0], center[1] + tip_offset),
                (center[0] - half, center[1] - half),
                (center[0] + half, center[1] - half),
            ], dtype=np.int32)
        elif direction == 'L':
            points = np.array([
                (center[0] + tip_offset, center[1]),
                (center[0] - half, center[1] - half),
                (center[0] - half, center[1] + half),
            ], dtype=np.int32)
        else:  # 'O'
            points = np.array([
                (center[0] - tip_offset, center[1]),
                (center[0] + half, center[1] - half),
                (center[0] + half, center[1] + half),
            ], dtype=np.int32)

        cv2.fillPoly(frame, [points.reshape(-1, 1, 2)], (255, 255, 255), lineType=cv2.LINE_AA)

    def _draw_food(self, frame, top_left, bottom_right):
        center = (
            (top_left[0] + bottom_right[0]) // 2,
            (top_left[1] + bottom_right[1]) // 2,
        )
        radius = max(3, self.cell_size // 5)
        cv2.circle(frame, center, radius + 1, (255, 255, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(frame, center, radius, self.food_color, -1, lineType=cv2.LINE_AA)

    def _ensure_frame_format(self, frame):
        if frame.shape[:2] != (self.height_px, self.width_px):
            raise RuntimeError("Frame com dimensões inesperadas ao salvar o vídeo.")
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        return np.ascontiguousarray(frame)

    def _validate_frame(self, frame):
        if frame.shape[:2] != (self.height_px, self.width_px):
            raise RuntimeError(
                f"Frame inválido: esperado {(self.height_px, self.width_px)}, obtido {frame.shape[:2]}"
            )
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise RuntimeError("Frame inválido: esperado imagem colorida com 3 canais.")


def resolve_maze_size(size_name=None):
    if size_name:
        if size_name in MazeSize.__members__:
            return MazeSize[size_name]
        raise ValueError(
            f"Tamanho inválido '{size_name}'. Escolha entre: "
            + ', '.join(MazeSize.__members__.keys())
        )
    env_choice = os.environ.get("MAZE_SIZE", "").upper()
    if env_choice and env_choice in MazeSize.__members__:
        return MazeSize[env_choice]
    return MazeSize.MEDIUM


def _parse_size_name(value):
    normalized = value.strip().replace('-', '_').replace(' ', '_').upper()
    candidates = {normalized, normalized.replace('_', '')}
    for candidate in candidates:
        if candidate in MazeSize.__members__:
            return candidate
    raise argparse.ArgumentTypeError(
        f"Tamanho inválido '{value}'. Use: " + ', '.join(name.lower() for name in MazeSize.__members__)
    )


def parse_arguments():
    parser = argparse.ArgumentParser(description="Maze solver agent controller")
    parser.add_argument(
        "--maze-size",
        dest="maze_size",
        type=_parse_size_name,
        help="Define o tamanho do labirinto (small, medium, large, extralarge)",
    )
    parser.add_argument(
        "--maze-file",
        dest="maze_file",
        help="Caminho personalizado para salvar/carregar o labirinto",
    )
    parser.add_argument(
        "--video-file",
        dest="video_file",
        help="Caminho do arquivo de vídeo a ser gerado",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Regenera o labirinto mesmo se o arquivo já existir",
    )
    parser.add_argument(
        "--dump-frames",
        action="store_true",
        help="Exporta cada frame em PNG para a pasta frames_debug",
    )
    parser.add_argument(
        "--food-sensor",
        dest="food_sensor",
        action="store_true",
        help="Ativa o sensor global de comida para priorizar direções com alimento",
    )
    parser.add_argument(
        "--no-food-sensor",
        dest="food_sensor",
        action="store_false",
        help="Desativa o sensor global de comida (padrão)",
    )
    parser.add_argument(
        "--show-log",
        dest="show_log",
        action="store_true",
        default=True,
        help="Exibe o log completo de movimentos (padrão)",
    )
    parser.add_argument(
        "--no-log",
        dest="show_log",
        action="store_false",
        help="Oculta o log de movimentos na saída",
    )
    parser.set_defaults(food_sensor=None)
    return parser.parse_args()


class Agent:
    """Controla o agente inteligente que explora e resolve o labirinto."""

    MOVE_ORDER = ['N', 'L', 'S', 'O']

    def __init__(self, environment, *, visualizer=None, show_log=True, use_food_sensor=False):
        self.env = environment
        self.internal_map = {}
        self.visited = set()
        self.action_log = [f"start at {self.env.get_position()}"]
        self.path_history = [self.env.get_position()]
        self.score = 0
        self.exit_position = self.env.exit_position
        self.visualizer = visualizer
        self.video_path = None
        self.total_food = self.env.food_total()
        self.food_collected = self.env.food_collected()
        self.steps_taken = 0
        self.collected_all_food = self.total_food == self.food_collected
        self.action_log.append(f"total food to collect: {self.total_food}")
        self.show_log = show_log
        self.use_food_sensor = use_food_sensor
        if self.use_food_sensor:
            self.action_log.append("food direction sensor enabled")
        self.food_direction_counts = {}
        self.food_sensor_matrix = None
        self.food_sensor_total = 0
        self.food_sensor_distances = {}
        self.virtual_visited = set()
        self.visit_counts = {}
        self.visible_counts = {}
        self.edge_counts = {}
        self.orientation_counts = {}
        self._increment_visit(self.env.get_position())

    def run(self):
        self._dfs_visit()
        if self.food_collected < self.total_food:
            missing = self.total_food - self.food_collected
            self.action_log.append(f"warning: {missing} comida(s) permaneceram inacessíveis")

        path = self._plan_path_to_exit()
        if path is None:
            self.action_log.append("exit unreachable")
            self._capture_frame()
            return

        self._follow_path(path)
        if self.env.get_position() == self.exit_position:
            self.action_log.append("exit reached")

    def _dfs_visit(self):
        current = self.env.get_position()
        if current in self.visited:
            return True

        self.visited.add(current)
        self._sense_and_update()
        self._capture_frame()

        if self._ready_to_exit():
            return False

        for direction, next_pos in self._prioritized_neighbors(current):
            if self._ready_to_exit():
                return False
            if self._move_to(direction):
                should_continue = self._dfs_visit()
                if not should_continue:
                    return False
                self._move_to(self._opposite(direction))
        return True

    def _plan_path_to_exit(self):
        if self.exit_position is None:
            return None

        start = self.env.get_position()
        goal = self.exit_position
        queue = deque([(start, [start])])
        visited = {start}

        while queue:
            position, path = queue.popleft()
            if position == goal:
                return path
            for direction in self.MOVE_ORDER:
                neighbor = self._neighbor_position(position, direction)
                if neighbor in visited:
                    continue
                tile = self.internal_map.get(neighbor, 'X')
                if tile == 'X':
                    continue
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
        return None

    def _has_known_path_to_exit(self):
        if self.exit_position is None:
            return False
        start = self.env.get_position()
        queue = deque([start])
        visited = {start}

        while queue:
            position = queue.popleft()
            if position == self.exit_position:
                return True
            for direction in self.MOVE_ORDER:
                neighbor = self._neighbor_position(position, direction)
                if neighbor in visited:
                    continue
                tile = self.internal_map.get(neighbor, 'X')
                if tile == 'X':
                    continue
                visited.add(neighbor)
                queue.append(neighbor)
        return False

    def _follow_path(self, path):
        if len(path) <= 1:
            self._capture_frame()
            return
        for target in path[1:]:
            current = self.env.get_position()
            direction = self._direction_from_to(current, target)
            if direction is None:
                raise RuntimeError("Caminho inválido gerado pelo BFS")
            moved = self._move_to(direction)
            if not moved:
                raise RuntimeError("Movimento inesperadamente bloqueado durante o BFS")
        self._capture_frame()

    def _sense_and_update(self):
        if self.use_food_sensor:
            food_sensor = self.env.get_food_direction_sensor()
            self.food_direction_counts = food_sensor['counts']
            self.food_sensor_matrix = food_sensor['matrix']
            self.food_sensor_total = food_sensor['total_food']
            self.food_sensor_distances = food_sensor['nearest_distance']
        sensor = self.env.getSensor()
        cx, cy = self.env.get_position()
        center_tile = self.env.current_tile()
        self.internal_map[(cx, cy)] = center_tile

        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                global_x = cx + dx
                global_y = cy + dy
                char = sensor[dy + 1][dx + 1]
                if dx == 0 and dy == 0:
                    char = center_tile
                if not self.env.in_bounds(global_x, global_y):
                    continue
                previous = self.internal_map.get((global_x, global_y))
                if char == 'S':
                    self.exit_position = (global_x, global_y)
                if previous == 'S':
                    continue
                self.internal_map[(global_x, global_y)] = char
                if char in {'_', 'E'}:
                    self._register_visibility((global_x, global_y), char)

        periphery = self.env.get_peripheral_view()
        self._apply_peripheral_knowledge(periphery)

    def _capture_frame(self):
        if self.visualizer is not None:
            self.visualizer.add_frame(self.env)

    def _prioritized_neighbors(self, current):
        periphery = self.env.get_peripheral_view()
        orientation = self.env.get_direction()
        left_dir = self._left_of(orientation)
        right_dir = self._right_of(orientation)
        back_dir = self._opposite(orientation)

        candidates = []
        for direction in self.MOVE_ORDER:
            next_pos = self._neighbor_position(current, direction)
            tile = self.internal_map.get(next_pos, '?')
            if tile == 'X' or next_pos in self.visited:
                continue
            priority = self._tile_priority(tile)
            priority += self._orientation_bonus(direction, orientation, left_dir, right_dir)
            priority += self._peripheral_bonus(direction, periphery, orientation, left_dir, right_dir, back_dir)
            priority += self._food_direction_bonus(direction)
            priority -= self._revisit_penalty(next_pos)
            priority -= self._edge_penalty(current, next_pos)
            priority -= self._orientation_visit_penalty(next_pos, direction)
            candidates.append((priority, direction, next_pos))

        candidates.sort(key=lambda item: (-item[0], self.MOVE_ORDER.index(item[1])))
        return [(direction, pos) for _, direction, pos in candidates]

    def _tile_priority(self, tile):
        if tile == 'o':
            return 120
        if tile == '?':
            return 60
        if tile == '_':
            return 30
        if tile == 'S':
            return 50 if self.collected_all_food else -20
        return 0

    def _orientation_bonus(self, direction, front, left_dir, right_dir):
        bonus = 0
        if direction == front:
            bonus += 8
        elif direction in {left_dir, right_dir}:
            bonus += 4
        elif direction == self._opposite(front):
            bonus -= 2
        return bonus

    def _food_direction_bonus(self, direction):
        if not self.use_food_sensor or not self.food_direction_counts:
            return 0
        primary = self.food_direction_counts.get(direction, 0)
        diagonal_map = {
            'N': ('NE', 'NO'),
            'S': ('SE', 'SO'),
            'L': ('NE', 'SE'),
            'O': ('NO', 'SO'),
        }
        diag_keys = diagonal_map.get(direction, ())
        diagonal_total = sum(self.food_direction_counts.get(key, 0) for key in diag_keys)
        if primary == 0 and diagonal_total == 0:
            return 0

        distance = self.food_sensor_distances.get(direction)
        distance_bonus = 0
        if distance is not None:
            distance_bonus += max(0, 18 - distance * 2)
        diag_distances = [
            dist for key in diag_keys
            if (dist := self.food_sensor_distances.get(key)) is not None
        ]
        if diag_distances:
            distance_bonus += max(0, 12 - min(diag_distances) * 2)

        return primary * 14 + diagonal_total * 6 + distance_bonus

    def _peripheral_bonus(self, direction, periphery, front, left_dir, right_dir, back_dir):
        labels = self._peripheral_labels_for(direction, front, left_dir, right_dir, back_dir)
        bonus = 0
        for label in labels:
            info = periphery.get(label)
            if not info:
                continue
            tile = info['tile']
            coord = info['coord']
            visits = self.visit_counts.get(coord, 0)
            if tile == 'o':
                bonus += max(25, 45 - 4 * visits)
            elif tile in {'_', 'E'}:
                if visits == 0:
                    bonus += 12
                else:
                    bonus += max(2, 10 - visits * 2)
            elif tile == 'S':
                bonus += 30 if self.collected_all_food else -10
            elif tile == 'X':
                bonus -= 15
            if coord in self.virtual_visited:
                bonus -= 6
        return bonus

    def _peripheral_labels_for(self, direction, front, left_dir, right_dir, back_dir):
        if direction == front:
            return ['front', 'front_left', 'front_right']
        if direction == left_dir:
            return ['left', 'front_left', 'back_left']
        if direction == right_dir:
            return ['right', 'front_right', 'back_right']
        if direction == back_dir:
            return ['back', 'back_left', 'back_right']
        return []

    def _revisit_penalty(self, coord):
        visits = self.visit_counts.get(coord, 0)
        penalty = 0
        if visits > 0:
            penalty += 12 + (visits - 1) * 6
        if coord in self.virtual_visited:
            penalty += 18 + max(0, visits - 1) * 4
        return penalty

    def _edge_penalty(self, current, neighbor):
        forward = self.edge_counts.get((current, neighbor), 0)
        backward = self.edge_counts.get((neighbor, current), 0)
        penalty = 0
        if forward > 0:
            penalty += 10 + (forward - 1) * 6
        if backward > 1:
            penalty += (backward - 1) * 4
        return penalty

    def _orientation_visit_penalty(self, coord, direction):
        count = self.orientation_counts.get((coord, direction), 0)
        if count <= 0:
            return 0
        return 8 + (count - 1) * 5

    def _apply_peripheral_knowledge(self, periphery):
        for label, info in periphery.items():
            x, y = info['coord']
            tile = info['tile']
            if not self.env.in_bounds(x, y):
                continue

            known = self.internal_map.get((x, y))
            if known == 'S':
                continue

            if tile == 'S':
                self.internal_map[(x, y)] = 'S'
                self.exit_position = (x, y)
                continue

            if tile == 'o':
                self.internal_map[(x, y)] = 'o'
                continue

            if tile == 'X':
                self.internal_map[(x, y)] = 'X'
                self._mark_virtual_visit((x, y))
                continue

            if tile in {'_', 'E'}:
                self.internal_map[(x, y)] = tile
                self._register_visibility((x, y), tile)
                should_mark = False
                seen = self.visible_counts.get((x, y), 0)
                if label in {'left', 'right', 'back_left', 'back_right', 'front_left', 'front_right'} and seen >= 2:
                    should_mark = True
                elif seen >= 3:
                    should_mark = True
                if should_mark and self._should_mark_virtual((x, y)):
                    self._mark_virtual_visit((x, y))

    def _increment_visit(self, coord):
        if coord is None:
            return
        self.visit_counts[coord] = self.visit_counts.get(coord, 0) + 1
        current_dir = self.env.get_direction()
        key = (coord, current_dir)
        if key not in self.orientation_counts:
            self.orientation_counts[key] = 0

    def _increment_edge(self, start, end):
        if start is None or end is None:
            return
        key = (start, end)
        self.edge_counts[key] = self.edge_counts.get(key, 0) + 1

    def _increment_orientation_visit(self, coord, direction):
        if coord is None:
            return
        key = (coord, direction)
        self.orientation_counts[key] = self.orientation_counts.get(key, 0) + 1

    def _register_visibility(self, coord, tile):
        if tile not in {'_', 'E'}:
            return
        if coord == self.env.get_position():
            return
        self.visible_counts[coord] = self.visible_counts.get(coord, 0) + 1

    def _should_mark_virtual(self, coord):
        x, y = coord
        tile_here = self.internal_map.get(coord)
        if tile_here not in {'_', 'E'}:
            return False
        for dx, dy in Environment.DIRECTIONS.values():
            neighbor = (x + dx, y + dy)
            tile = self.internal_map.get(neighbor)
            if tile == 'o':
                return False
            if tile in {'_', 'E'}:
                if self.visible_counts.get(neighbor, 0) < 2 and self.visit_counts.get(neighbor, 0) == 0:
                    return False
        return True

    def _mark_virtual_visit(self, coord):
        if coord == self.env.get_position():
            return
        if coord in self.virtual_visited:
            return
        self.virtual_visited.add(coord)

    def _ready_to_exit(self):
        return self.collected_all_food and self._has_known_path_to_exit()

    def _move_to(self, direction):
        self._turn_to(direction)
        previous_remaining = self.env.food_remaining()
        previous_position = self.env.get_position()
        success, cell, reward = self.env.move()
        if not success:
            self.action_log.append(f"move blocked {direction}")
            return False
        self.steps_taken += 1
        self.score += reward
        new_pos = self.env.get_position()
        self.action_log.append(f"move {direction} -> {new_pos} ({cell})")
        self.path_history.append(new_pos)
        self._increment_visit(new_pos)
        self._increment_edge(previous_position, new_pos)
        self._increment_orientation_visit(new_pos, direction)
        self._sense_and_update()
        current_remaining = self.env.food_remaining()
        if current_remaining < previous_remaining:
            collected_now = previous_remaining - current_remaining
            self.food_collected += collected_now
            if self.food_collected >= self.total_food and not self.collected_all_food:
                self.collected_all_food = True
                self.action_log.append("all food collected")
        self._capture_frame()
        return True

    def _turn_to(self, direction):
        if self.env.get_direction() == direction:
            return
        self.env.setDirection(direction)
        self.action_log.append(f"turn {direction}")

    def _neighbor_position(self, position, direction):
        dx, dy = Environment.DIRECTIONS[direction]
        return (position[0] + dx, position[1] + dy)

    def _direction_from_to(self, start, end):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        for direction, (offset_x, offset_y) in Environment.DIRECTIONS.items():
            if (dx, dy) == (offset_x, offset_y):
                return direction
        return None

    def _opposite(self, direction):
        opposites = {'N': 'S', 'S': 'N', 'L': 'O', 'O': 'L'}
        return opposites[direction]

    def _left_of(self, direction):
        left_map = {'N': 'O', 'O': 'S', 'S': 'L', 'L': 'N'}
        return left_map[direction]

    def _right_of(self, direction):
        right_map = {'N': 'L', 'L': 'S', 'S': 'O', 'O': 'N'}
        return right_map[direction]

    def render_internal_map(self):
        if not self.internal_map:
            return ""

        xs = [coord[0] for coord in self.internal_map]
        ys = [coord[1] for coord in self.internal_map]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        current = self.env.get_position()

        rows = []
        for y in range(min_y, max_y + 1):
            row_chars = []
            for x in range(min_x, max_x + 1):
                if (x, y) == current:
                    row_chars.append('@')
                else:
                    row_chars.append(self.internal_map.get((x, y), '?'))
            rows.append(''.join(row_chars))
        return '\n'.join(rows)

    def save_visualization(self):
        if self.visualizer is None:
            return None
        try:
            path = self.visualizer.save()
        except RuntimeError as error:
            self.action_log.append(f"video generation failed: {error}")
            return None
        self.video_path = path
        return path

    def print_report(self):
        total_food_points = self.food_collected * 10
        total_step_cost = self.steps_taken
        print(f"Pontuação final: {self.score}")
        print(
            f"Recompensas: {self.food_collected} comida(s) x 10 = {total_food_points} | "
            f"Custo de passos: {self.steps_taken} x -1 = -{total_step_cost}"
        )
        print("\nMapa aprendido pelo agente:")
        print(self.render_internal_map())
        if self.show_log:
            print("\nSequência de passos:")
            for action in self.action_log:
                print(action)
        if self.video_path:
            print(f"\nVídeo salvo em: {self.video_path}")


def main():
    args = parse_arguments()
    has_cli_args = len(sys.argv) > 1
    maze_size = resolve_maze_size(args.maze_size)
    default_maze_file = f"maze.txt"
    maze_file = args.maze_file or default_maze_file

    use_existing_default = (
        not has_cli_args
        and args.maze_file is None
        and os.path.exists(maze_file)
    )
    if use_existing_default:
        regenerate = False
    else:
        regenerate = args.regenerate or args.maze_size is not None or not os.path.exists(maze_file)

    if regenerate:
        generated = MazeMap(size=maze_size, auto_generate=True, filepath=maze_file)
        print(
            f"Labirinto gerado em {generated.filepath} "
            f"({maze_size.name.title()} {generated.width}x{generated.height})"
        )
    else:
        print(f"Usando labirinto existente: {maze_file}")

    env = Environment(maze_file)

    dump_frames = args.dump_frames
    if not dump_frames:
        dump_frames = os.environ.get("MAZE_DUMP_FRAMES", "").lower() in {"1", "true", "yes"}

    if args.food_sensor is None:
        food_sensor_env = os.environ.get("MAZE_FOOD_SENSOR", "").lower() in {"1", "true", "yes", "on"}
        food_sensor_enabled = food_sensor_env
    else:
        food_sensor_enabled = args.food_sensor

    default_video = f"maze_run_{maze_size.name.lower()}.mp4"
    video_file = args.video_file or default_video

    visualizer = Visualizer(env, output_path=video_file, dump_debug_frames=dump_frames)
    agent = Agent(
        env,
        visualizer=visualizer,
        show_log=args.show_log,
        use_food_sensor=food_sensor_enabled,
    )
    agent.run()
    agent.save_visualization()
    agent.print_report()


if __name__ == "__main__":
    main()
