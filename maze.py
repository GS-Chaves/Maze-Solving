import random
from collections import deque
from enum import Enum


class MazeSize(Enum):
    SMALL = 9
    MEDIUM = 12
    LARGE = 15
    EXTRA_LARGE = 18


class MazeMap:
    def __init__(
        self,
        size=MazeSize.MEDIUM,
        *,
        auto_generate=False,
        filepath=None,
        loop_density=0.21,
        food_density=None,
    ):
        if not isinstance(size, MazeSize):
            raise TypeError("size deve ser uma instância de MazeSize")

        self.size = size
        self.width = size.value
        self.height = size.value
        self.loop_density = loop_density
        if food_density is None:
            density_by_size = {
                MazeSize.SMALL: 0.2,
                MazeSize.MEDIUM: 0.25,
                MazeSize.LARGE: 0.23,
                MazeSize.EXTRA_LARGE: 0.2,
            }
            self.food_density = density_by_size[size]
        else:
            self.food_density = food_density
        self.map_data = [['X' for _ in range(self.width)] for _ in range(self.height)]
        self.entry = None
        self.exit = None
        self.filepath = None
        self._main_path = []

        if auto_generate:
            self.generate_random_maze()
            self.save_to_file(filepath)

    def generate_random_maze(self):
        self._reset_grid()
        entry_y = random.randrange(1, self.height - 1, 2)
        start = (1, entry_y)
        self.map_data[entry_y][0] = 'E'
        self.map_data[entry_y][1] = '_'
        self.entry = (0, entry_y)

        self._carve_maze(start)
        self._choose_exit_and_path(start)
        protected = set(self._main_path)
        self._add_loops(protected)
        self._add_islands(protected)
        self._place_foods(protected)
        return self.map_data

    def _reset_grid(self):
        for y in range(self.height):
            for x in range(self.width):
                self.map_data[y][x] = 'X'

    def _carve_maze(self, start):
        stack = [start]
        visited = {start}

        while stack:
            x, y = stack[-1]
            self.map_data[y][x] = '_'
            neighbors = []
            for dx, dy in ((0, -2), (2, 0), (0, 2), (-2, 0)):
                nx, ny = x + dx, y + dy
                if 1 <= nx < self.width - 1 and 1 <= ny < self.height - 1:
                    if self.map_data[ny][nx] == 'X':
                        neighbors.append((nx, ny, dx, dy))

            if neighbors:
                nx, ny, dx, dy = random.choice(neighbors)
                wall_x = x + dx // 2
                wall_y = y + dy // 2
                self.map_data[wall_y][wall_x] = '_'
                self.map_data[ny][nx] = '_'
                stack.append((nx, ny))
                visited.add((nx, ny))
            else:
                stack.pop()

    def _add_loops(self, protected):
        interior_walls = [
            (x, y)
            for y in range(1, self.height - 1)
            for x in range(1, self.width - 1)
            if self.map_data[y][x] == 'X'
        ]
        random.shuffle(interior_walls)
        openings = int(len(interior_walls) * self.loop_density)

        opened = 0
        for x, y in interior_walls:
            if opened >= openings:
                break
            if (x, y) in protected:
                continue
            open_neighbors = 0
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if (x + dx, y + dy) in protected:
                    open_neighbors = 0
                    break
                if self.map_data[y + dy][x + dx] in {'_', 'S'}:
                    open_neighbors += 1
            if open_neighbors >= 2:
                self.map_data[y][x] = '_'
                opened += 1

    def _add_islands(self, protected):
        attempts = max(3, self.width // 2)
        for _ in range(attempts):
            x = random.randrange(2, self.width - 2)
            y = random.randrange(2, self.height - 2)
            cells = [
                (x, y),
                (x + 1, y),
                (x, y + 1),
                (x + 1, y + 1),
            ]
            if any(cx >= self.width - 1 or cy >= self.height - 1 for cx, cy in cells):
                continue
            if any(self.map_data[cy][cx] in {'E', 'S'} for cx, cy in cells):
                continue
            if any((cx, cy) in protected for cx, cy in cells):
                continue
            if any(self.map_data[cy][cx] == 'X' for cx, cy in cells):
                for cx, cy in cells:
                    self.map_data[cy][cx] = '_'
    def _choose_exit_and_path(self, start):
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        queue = deque([start])
        parents = {start: None}
        distances = {start: 0}
        best = start

        while queue:
            current = queue.popleft()
            cx, cy = current
            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < self.width and 0 <= ny < self.height):
                    continue
                if (nx, ny) in parents:
                    continue
                tile = self.map_data[ny][nx]
                if tile == 'X':
                    continue
                parents[(nx, ny)] = current
                distances[(nx, ny)] = distances[current] + 1
                queue.append((nx, ny))
                if distances[(nx, ny)] > distances.get(best, 0):
                    best = (nx, ny)

        if best == start and len(parents) > 1:
            candidates = [cell for cell in parents if cell != start]
            best = random.choice(candidates)
        elif best == start:
            for dx, dy in directions:
                neighbor = (start[0] + dx, start[1] + dy)
                if parents.get(neighbor) == start:
                    best = neighbor
                    break

        self.exit = best
        path = []
        node = best
        while node is not None:
            path.append(node)
            node = parents.get(node)
        path.reverse()
        self._main_path = path

        exit_x, exit_y = self.exit
        self.map_data[exit_y][exit_x] = 'S'

    def _place_foods(self, protected):
        if not self._main_path:
            return

        open_cells = [
            (x, y)
            for y in range(1, self.height - 1)
            for x in range(1, self.width - 1)
            if self.map_data[y][x] == '_'
        ]

        if not open_cells:
            return

        total_target = max(5, int(len(open_cells) * self.food_density))
        total_target = min(total_target, len(open_cells))
        if total_target <= 0:
            return

        entry_path_cell = (1, self.entry[1])
        path_cells = [
            cell
            for cell in self._main_path
            if cell not in {entry_path_cell, self.exit}
        ]
        primary_target = min(len(path_cells), max(3, int(total_target * 0.7)))
        secondary_target = max(0, total_target - primary_target)

        selected_foods = set()

        path_food = self._select_path_food_positions(path_cells, primary_target, selected_foods)
        for x, y in path_food:
            self.map_data[y][x] = 'o'
            selected_foods.add((x, y))

        if secondary_target > 0:
            branch_food = self._select_branch_food_positions(
                open_cells, selected_foods, protected, secondary_target
            )
            for x, y in branch_food:
                self.map_data[y][x] = 'o'
                selected_foods.add((x, y))

    def _select_path_food_positions(self, path_cells, count, taken):
        if count <= 0 or not path_cells:
            return []
        if count >= len(path_cells):
            return path_cells

        selected = []
        chosen = set()
        for cell in path_cells:
            if cell in taken:
                continue
            if all(not self._is_adjacent(cell, other) for other in taken) and all(
                not self._is_adjacent(cell, other) for other in chosen
            ):
                selected.append(cell)
                chosen.add(cell)
                if len(selected) == count:
                    break

        if len(selected) < count:
            for cell in path_cells:
                if cell in taken or cell in chosen:
                    continue
                if any(self._is_adjacent(cell, other) for other in taken):
                    continue
                if any(self._is_adjacent(cell, other) for other in chosen):
                    continue
                selected.append(cell)
                chosen.add(cell)
                if len(selected) == count:
                    break

        if len(selected) < count:
            for cell in path_cells:
                if cell in taken or cell in chosen:
                    continue
                selected.append(cell)
                chosen.add(cell)
                if len(selected) == count:
                    break

        return selected[:count]

    def _select_branch_food_positions(self, open_cells, selected_positions, protected, count):
        if count <= 0:
            return []

        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        candidates = []
        for x, y in open_cells:
            if (x, y) in selected_positions or (x, y) in protected:
                continue
            open_neighbors = sum(
                1
                for dx, dy in directions
                if 0 <= x + dx < self.width
                and 0 <= y + dy < self.height
                and self.map_data[y + dy][x + dx] != 'X'
            )
            if open_neighbors == 1:
                candidates.append((x, y))

        random.shuffle(candidates)
        selected = []
        chosen = set()
        for cell in candidates:
            if self._has_adjacent(cell, selected_positions) or self._has_adjacent(cell, chosen):
                continue
            selected.append(cell)
            chosen.add(cell)
            if len(selected) == count:
                break

        if len(selected) < count:
            remaining = [
                cell
                for cell in open_cells
                if cell not in selected_positions and cell not in protected and cell not in chosen
            ]
            random.shuffle(remaining)
            for cell in list(remaining):
                if len(selected) == count:
                    break
                if self._has_adjacent(cell, selected_positions) or self._has_adjacent(cell, chosen):
                    continue
                selected.append(cell)
                chosen.add(cell)

        if len(selected) < count:
            for cell in open_cells:
                if len(selected) == count:
                    break
                if cell in selected_positions or cell in protected or cell in chosen:
                    continue
                selected.append(cell)
                chosen.add(cell)

        return selected[:count]

    @staticmethod
    def _is_adjacent(a, b):
        if a is None or b is None:
            return False
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1

    def _has_adjacent(self, cell, positions):
        return any(self._is_adjacent(cell, other) for other in positions)

    def save_to_file(self, filepath=None):
        filename = filepath if filepath is not None else f"maze{self.width}.txt"
        with open(filename, "w", encoding="ascii") as maze_file:
            for row in self.map_data:
                maze_file.write(''.join(row) + '\n')
        self.filepath = filename
        return filename

    def display(self):
        for row in self.map_data:
            print(''.join(row))
